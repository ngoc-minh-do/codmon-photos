import hashlib
import pathlib
import sqlite3
import types

import faces
import numpy as np
import pytest


def unit(*values):
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


PIPE = "v2:yunet:s0.6:w600k_r50.onnx:arc112:floor20"
OTHER_PIPE = "v2:scrfd:s0.5:w600k_r50.onnx:arc112:floor20"


def add_ref(db, person="kid", emb=None, pipeline=PIPE):
    emb = unit(1, 0, 0) if emb is None else emb
    db.execute(
        "INSERT INTO refs (person, pipeline, source, crop, emb) VALUES (?,?,?,?,?)",
        (person, pipeline, "s", "c", emb.tobytes()),
    )


def test_sweep_counts_and_shares():
    rows = [
        {"score": 0.62, "faces": 2},
        {"score": 0.50, "faces": 1},
        {"score": 0.35, "faces": 3},
        {"score": 0.00, "faces": 0},  # no usable face: never kept
    ]
    got = {t: (kept, round(share)) for t, kept, share in faces.sweep(rows, edges=(0.30, 0.45, 0.60))}
    assert got == {0.30: (3, 75), 0.45: (2, 50), 0.60: (1, 25)}


def test_matched_rows_needs_a_face_and_the_threshold(tmp_path):
    rows = [
        {"path": tmp_path / "a.jpg", "score": 0.55, "faces": 2},
        {"path": tmp_path / "b.jpg", "score": 0.20, "faces": 1},
        {"path": tmp_path / "c.jpg", "score": 0.00, "faces": 0},
    ]
    assert [r["path"].name for r in faces.matched_rows(rows, 0.45)] == ["a.jpg"]


def test_export_dry_run_writes_nothing(tmp_path):
    album = tmp_path / "2026-09-25"
    album.mkdir()
    photo = album / "her.jpg"
    photo.write_bytes(b"\xff\xd8data")
    out = tmp_path / "matched"
    stats = faces.export_matches([{"album": album.name, "path": photo, "score": 0.6, "faces": 2}], "kid", out, 0.45)
    assert stats == {"copied": 1, "kept": 0, "pruned": 0}
    assert not out.exists()


def test_export_copies_into_person_and_album_dirs(tmp_path):
    album = tmp_path / "downloads" / "2026-09-25"
    album.mkdir(parents=True)
    photo = album / "her.jpg"
    photo.write_bytes(b"\xff\xd8data")
    out = tmp_path / "matched"
    rows = [{"album": album.name, "path": photo, "score": 0.6, "faces": 2}]

    stats = faces.export_matches(rows, "kid", out, 0.45, apply=True)

    dest = out / "kid" / album.name / "her.jpg"
    assert stats["copied"] == 1 and dest.read_bytes() == photo.read_bytes()
    assert photo.exists(), "album folder must stay untouched"
    assert faces.export_matches(rows, "kid", out, 0.45, apply=True)["kept"] == 1


def test_export_lower_threshold_brings_more_and_prune_drops_stale(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    both, only_low = album / "both.jpg", album / "low.jpg"
    both.write_bytes(b"\xff\xd8a")
    only_low.write_bytes(b"\xff\xd8b")
    rows = [
        {"album": "album", "path": only_low, "score": 0.40, "faces": 1},
        {"album": "album", "path": both, "score": 0.60, "faces": 1},
    ]
    out = tmp_path / "matched"

    faces.export_matches(rows, "kid", out, 0.50, apply=True)
    assert sorted(p.name for p in (out / "kid" / "album").iterdir()) == ["both.jpg"]

    stats = faces.export_matches(rows, "kid", out, 0.35, apply=True)
    assert stats["copied"] == 1 and stats["kept"] == 1

    stats = faces.export_matches(rows, "kid", out, 0.50, prune=True, apply=True)
    assert stats["pruned"] == 1
    assert sorted(p.name for p in (out / "kid" / "album").iterdir()) == ["both.jpg"]


def test_export_can_hardlink(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    photo = album / "her.jpg"
    photo.write_bytes(b"\xff\xd8data")
    out = tmp_path / "matched"
    faces.export_matches(
        [{"album": "album", "path": photo, "score": 0.6, "faces": 1}], "kid", out, 0.45, link=True, apply=True
    )
    dest = out / "kid" / "album" / "her.jpg"
    assert dest.stat().st_ino == photo.stat().st_ino


def test_score_faces_picks_best_face_and_reference():
    gallery = np.stack([unit(1, 0, 0), unit(0, 1, 0)])
    rows = [((0, 0, 10, 10), 0.9, unit(0, 0, 1)), ((0, 0, 12, 12), 0.9, unit(0.8, 0.6, 0))]
    score, idx = faces.score_faces(rows, gallery)
    assert idx == 1 and abs(score - 0.8) < 1e-6
    assert faces.score_faces([], gallery) == (0.0, -1)


def test_store_and_load_faces_roundtrip(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    emb = np.arange(8, dtype=np.float32) / 10
    faces.store_photo(db, "abc", (750, 500), PIPE, [((1.0, 2.0, 33.0, 44.0), 0.91, emb)])
    loaded = faces.load_faces(db, "abc", PIPE)
    assert len(loaded) == 1
    assert loaded[0][0] == (1.0, 2.0, 33.0, 44.0)
    assert loaded[0][1] == 0.91
    assert np.allclose(loaded[0][2], emb)


def test_cache_is_keyed_by_content_not_path(tmp_path):
    one, two = tmp_path / "one.jpg", tmp_path / "elsewhere" / "two.jpg"
    two.parent.mkdir()
    one.write_bytes(b"\xff\xd8same")
    two.write_bytes(b"\xff\xd8same")
    db = faces.open_db(tmp_path / "faces.db")
    sha1 = faces.file_sha1(one)
    faces.store_photo(db, sha1, (10, 10), PIPE, [])
    assert faces.cached_photo(db, faces.file_sha1(two), PIPE), "same bytes must hit the cache"
    assert not faces.cached_photo(db, sha1, OTHER_PIPE), "another detector is a different cache slot"
    two.write_bytes(b"\xff\xd8differs")
    assert not faces.cached_photo(db, faces.file_sha1(two), PIPE)


def test_collect_scores_albums_from_the_cache(tmp_path):
    album = tmp_path / "downloads" / "2026-09-25"
    album.mkdir(parents=True)
    her, other = album / "her.jpg", album / "other.jpg"
    her.write_bytes(b"\xff\xd8her")
    other.write_bytes(b"\xff\xd8other")
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    faces.store_photo(db, faces.file_sha1(her), (750, 500), PIPE, [((0, 0, 60, 60), 0.9, unit(1, 0, 0))])
    faces.store_photo(db, faces.file_sha1(other), (750, 500), PIPE, [((0, 0, 30, 30), 0.9, unit(0, 1, 0))])

    rows = faces.collect(db, "kid", [tmp_path / "downloads"], min_face=40, pipeline=PIPE)

    by_name = {r["path"].name: r for r in rows}
    assert by_name["her.jpg"]["score"] == pytest.approx(1.0)
    assert by_name["her.jpg"]["album"] == "2026-09-25"
    assert by_name["her.jpg"]["root"] == tmp_path / "downloads"
    assert by_name["other.jpg"]["score"] == pytest.approx(0.0)
    assert by_name["other.jpg"]["faces"] == 0, "a 30px face is below --min-face 40"


def test_gallery_requires_references(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    with pytest.raises(SystemExit, match="enroll"):
        faces.gallery(db, "nobody", PIPE)


def test_open_db_is_idempotent(tmp_path):
    path = tmp_path / "faces.db"
    faces.open_db(path).close()
    db = sqlite3.connect(path)
    assert db.execute("SELECT COUNT(*) FROM photos").fetchone() == (0,)
    assert db.execute("PRAGMA user_version").fetchone()[0] == faces.SCHEMA_VERSION


def test_open_db_rebuilds_an_older_cache(tmp_path, capsys):
    path = tmp_path / "faces.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        faces.SCHEMA.replace("pipeline TEXT NOT NULL,", "")
        .replace("PRIMARY KEY (sha1, pipeline)", "PRIMARY KEY (sha1)")
        .replace("PRIMARY KEY (sha1, pipeline, idx)", "PRIMARY KEY (sha1, idx)")
        .replace("PRIMARY KEY (person, pipeline, source)", "PRIMARY KEY (person, source)")
    )
    legacy.execute("INSERT INTO refs (person, source, crop, emb) VALUES ('kid','s','c',?)", (b"old",))
    legacy.commit()
    legacy.close()

    db = faces.open_db(path)  # must not raise "no such column: pipeline"

    assert "rebuilding" in capsys.readouterr().out
    assert db.execute("SELECT COUNT(*) FROM refs").fetchone() == (0,), "stale rows are dropped"
    assert db.execute("PRAGMA user_version").fetchone()[0] == faces.SCHEMA_VERSION
    add_ref(db, person="kid")  # the pipeline column exists now
    assert faces.detector_of(db.execute("SELECT pipeline FROM refs").fetchone()[0]) == "yunet"


def test_pipeline_id_separates_detectors_score_and_floor():
    yunet = types.SimpleNamespace(name="yunet", score=0.6)
    scrfd = types.SimpleNamespace(name="scrfd", score=0.5)
    assert faces.pipeline_id(yunet) != faces.pipeline_id(scrfd)
    assert faces.pipeline_id(yunet, floor=20) != faces.pipeline_id(yunet, floor=40)
    assert faces.pipeline_id(yunet) != faces.pipeline_id(types.SimpleNamespace(name="yunet", score=0.9))
    assert faces.pipeline_id(yunet) == PIPE
    assert faces.detector_of(faces.pipeline_id(scrfd)) == "scrfd"


def test_two_pipelines_coexist_for_the_same_photo(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"\xff\xd8x")
    sha1 = faces.file_sha1(photo)
    db = faces.open_db(tmp_path / "faces.db")
    faces.store_photo(db, sha1, (750, 500), PIPE, [((0, 0, 40, 40), 0.9, unit(1, 0, 0))])
    faces.store_photo(db, sha1, (750, 500), OTHER_PIPE, [((0, 0, 40, 40), 0.8, unit(0, 0.6, 0.8))])

    assert faces.cached_photo(db, sha1, PIPE) and faces.cached_photo(db, sha1, OTHER_PIPE)
    assert sorted(faces.pipelines(db)) == sorted([PIPE, OTHER_PIPE])
    assert np.allclose(faces.load_faces(db, sha1, PIPE)[0][2], unit(1, 0, 0))
    assert np.allclose(faces.load_faces(db, sha1, OTHER_PIPE)[0][2], unit(0, 0.6, 0.8))


def test_pipeline_for_finds_the_cached_id_else_predicts_one(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    faces.store_photo(db, "abc", (10, 10), PIPE, [])
    assert faces.pipeline_for(db, "yunet") == PIPE
    assert faces.pipeline_for(db, "scrfd").startswith("v2:scrfd:")


def test_compare_counts_agreement_and_only_buckets(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    names = ["both.jpg", "both2.jpg", "a.jpg", "b.jpg", "none.jpg"]

    def rows(pipeline, scores):
        return [
            {
                "album": "album",
                "path": album / n,
                "score": s,
                "faces": f,
                "face_width": 50.0,
                "pipeline": pipeline,
                "root": tmp_path,
            }
            for n, (s, f) in zip(names, scores, strict=True)
        ]

    a = rows(PIPE, [(0.60, 1), (0.70, 1), (0.55, 1), (0.10, 1), (0.0, 0)])
    b = rows(OTHER_PIPE, [(0.50, 1), (0.40, 1), (0.20, 1), (0.65, 1), (0.0, 0)])
    stats = faces.compare({PIPE: a, OTHER_PIPE: b}, 0.45)

    assert stats["shared"] == 5
    # both.jpg hit by both, both2.jpg and a.jpg only by the first, b.jpg only by the second
    assert (stats["both"], stats["only_a"], stats["only_b"], stats["neither"]) == (1, 2, 1, 1)
    assert stats["deltas"].round(2).tolist() == [-0.10]
    # the four buckets are disjoint and cover every shared photo
    assert stats["both"] + stats["only_a"] + stats["only_b"] + stats["neither"] == stats["shared"]
    # per-detector totals and the union are derived, not the sum of the "only" buckets
    assert stats["hit_a"] == stats["both"] + stats["only_a"] == 3
    assert stats["hit_b"] == stats["both"] + stats["only_b"] == 2
    assert stats["union"] == stats["both"] + stats["only_a"] + stats["only_b"] == 4
    assert stats["agree"] == stats["both"] + stats["neither"] == 2


def test_compare_report_shows_the_contingency_table():
    album = pathlib.Path("/albums/x")
    a = [
        {"album": "x", "path": album / n, "score": s, "faces": 1, "face_width": 50.0, "pipeline": PIPE}
        for n, s in (("both.jpg", 0.6), ("a.jpg", 0.6), ("b.jpg", 0.1))
    ]
    b = [
        {"album": "x", "path": album / n, "score": s, "faces": 1, "face_width": 50.0, "pipeline": OTHER_PIPE}
        for n, s in (("both.jpg", 0.6), ("a.jpg", 0.1), ("b.jpg", 0.6))
    ]
    stats = faces.compare({PIPE: a, OTHER_PIPE: b}, 0.45)
    text = faces.compare_report(stats, [], 0.45)
    assert "both 1" in text and "only yunet 1" in text and "only scrfd 1" in text
    assert "either 3" in text
    # nobody is missed by both in this fixture, so agreement is just "both"
    assert "agree 1/3" in text
    assert "yunet hit" in text and "scrfd miss" in text


def test_disagreements_lists_the_split_photos():
    album = pathlib.Path("/albums/x")
    a = [{"album": "x", "path": album / "one.jpg", "score": 0.60, "faces": 1, "face_width": 80.0, "pipeline": PIPE}]
    b = [
        {"album": "x", "path": album / "one.jpg", "score": 0.20, "faces": 0, "face_width": 0.0, "pipeline": OTHER_PIPE}
    ]
    found = faces.disagreements({PIPE: a, OTHER_PIPE: b}, 0.45)
    assert len(found) == 1
    assert found[0]["winner"] == PIPE
    assert found[0]["width_delta"] == -80.0
    assert "yunet" in faces.compare_report(faces.compare({PIPE: a, OTHER_PIPE: b}, 0.45), found, 0.45)


def test_scrfd_decode_helpers():
    centres = faces.scrfd_anchors(2, 2, 8, num_anchors=1)
    assert centres.tolist() == [[0, 0], [8, 0], [0, 8], [8, 8]]
    assert faces.scrfd_anchors(1, 1, 8, num_anchors=2).tolist() == [[0, 0], [0, 0]]

    boxes = faces.distance2bbox(np.array([[10.0, 10.0]]), np.array([[2.0, 3.0, 4.0, 5.0]]))
    assert boxes.tolist() == [[8.0, 7.0, 14.0, 15.0]]

    kps = faces.distance2kps(np.array([[10.0, 10.0]]), np.array([[1.0, 2.0, 3.0, 4.0]]))
    assert kps.tolist() == [[11.0, 12.0, 13.0, 14.0]]

    dets = np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]], dtype=np.float32)
    assert faces.nms(dets, 0.4) == [0, 2]


def test_scan_skips_files_that_vanished(tmp_path, monkeypatch):
    import cv2

    root = tmp_path / "downloads" / "album"
    root.mkdir(parents=True)
    keep, doomed = root / "keep.jpg", root / "doomed.jpg"
    cv2.imwrite(str(keep), np.zeros((32, 32, 3), dtype=np.uint8))
    cv2.imwrite(str(doomed), np.zeros((32, 32, 3), dtype=np.uint8))
    db = faces.open_db(tmp_path / "faces.db")

    real_sha1 = faces.file_sha1

    def sha1_or_race(path):
        if path.name == "doomed.jpg":
            raise FileNotFoundError(path)  # deleted/renamed mid-scan
        return real_sha1(path)

    monkeypatch.setattr(faces, "file_sha1", sha1_or_race)
    stub = types.SimpleNamespace(name="stub", faces=lambda img: [])

    done, skipped = faces.scan(db, [tmp_path / "downloads"], stub, None, floor=20)

    assert (done, skipped) == (1, 0)
    assert faces.cached_photo(db, real_sha1(keep), faces.pipeline_id(stub, 20))


def test_enroll_guard_is_per_detector(tmp_path):
    import face_filter

    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db, person="kid", pipeline=PIPE)

    assert face_filter.enrolled_pipelines(db, "kid", "yunet") == [PIPE]
    assert face_filter.enrolled_pipelines(db, "kid", "scrfd") == [], "a second detector must not need --force"

    add_ref(db, person="kid", pipeline=OTHER_PIPE)
    assert face_filter.enrolled_pipelines(db, "kid", "scrfd") == [OTHER_PIPE]
    assert face_filter.enrolled_pipelines(db, "someoneelse", "yunet") == []


def _enroll_fixture(tmp_path, monkeypatch):
    """Two photos: one with a single face, one with a big impostor + the real face."""
    import cv2

    refs = tmp_path / "refs" / "kid" / "face"
    refs.mkdir(parents=True)
    real = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    impostor = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for name, colour in (("single.jpg", (10, 10, 10)), ("group.jpg", (20, 20, 20))):
        cv2.imwrite(str(refs / name), np.full((64, 64, 3), colour, dtype=np.uint8))

    def fake_detect(img, det, min_width=20):
        if int(img[0, 0, 0]) == 10:  # single.jpg
            return [((0, 0, 50, 50), np.array([[1.0, 1.0]] * 5), 0.9)]
        # group.jpg: the impostor is bigger, so "largest face" would pick it
        return [
            ((0, 0, 60, 60), np.array([[9.0, 9.0]] * 5), 0.8),
            ((0, 0, 20, 20), np.array([[1.0, 1.0]] * 5), 0.7),
        ]

    def fake_embed(img, kps, rec):
        return real if kps[0][0] == 1.0 else impostor  # landmark x picks the identity

    monkeypatch.setattr(faces, "detect", fake_detect)
    monkeypatch.setattr(faces, "embed", fake_embed)
    monkeypatch.setattr(faces, "align", lambda img, kps, size=112: np.zeros((size, size, 3), np.uint8))
    return refs


def test_enroll_prefers_the_face_matching_the_gallery_over_the_largest(tmp_path, monkeypatch):
    refs = _enroll_fixture(tmp_path, monkeypatch)
    db = faces.open_db(tmp_path / "faces.db")
    det = types.SimpleNamespace(name="yunet", score=0.6, faces=lambda img: [])

    warnings = faces.enroll(db, "kid", refs, det, None, tmp_path / ".cache")

    embs = [np.frombuffer(r[0], dtype=np.float32) for r in db.execute("SELECT emb FROM refs ORDER BY source")]
    assert len(embs) == 2
    assert all(np.allclose(e, [1.0, 0.0, 0.0]) for e in embs), "the 60px impostor must not be enrolled"
    assert any("matching the gallery" in w for w in warnings)


def test_enroll_refuses_to_guess_without_a_single_face_photo(tmp_path, monkeypatch):
    """A gallery of only group photos has no seed: nothing is enrolled, nothing guessed."""
    import cv2

    refs = tmp_path / "refs" / "kid" / "face"
    refs.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(refs / name), np.zeros((32, 32, 3), dtype=np.uint8))

    monkeypatch.setattr(
        faces,
        "detect",
        lambda img, det, min_width=20: [
            ((0, 0, 60, 60), np.array([[9.0, 9.0]] * 5), 0.8),
            ((0, 0, 20, 20), np.array([[1.0, 1.0]] * 5), 0.7),
        ],
    )
    monkeypatch.setattr(faces, "embed", lambda img, kps, rec: np.array([1.0, 0.0, 0.0], np.float32))
    monkeypatch.setattr(faces, "align", lambda img, kps, size=112: np.zeros((size, size, 3), np.uint8))
    db = faces.open_db(tmp_path / "faces.db")
    det = types.SimpleNamespace(name="yunet", score=0.6, faces=lambda img: [])

    with pytest.raises(SystemExit, match="only her face"):
        faces.enroll(db, "kid", refs, det, None, tmp_path / ".cache")

    assert db.execute("SELECT COUNT(*) FROM refs").fetchone() == (0,)


def test_pick_reference_needs_a_seeded_gallery():
    with pytest.raises(ValueError, match="seeded gallery"):
        two_faces = [((0, 0, 60, 60), np.zeros((5, 2)), 0.9), ((0, 0, 20, 20), np.zeros((5, 2)), 0.8)]
        faces._pick_reference(two_faces, [], None, None)


def test_re_enroll_leaves_no_stale_crop_files(tmp_path, monkeypatch):
    refs = _enroll_fixture(tmp_path, monkeypatch)
    db = faces.open_db(tmp_path / "faces.db")
    det = types.SimpleNamespace(name="yunet", score=0.6, faces=lambda img: [])
    crop_root = tmp_path / ".cache"

    faces.enroll(db, "kid", refs, det, None, crop_root)
    first = sorted(p.name for p in (crop_root / "kid" / "yunet").iterdir())
    assert len(first) == 2

    (refs / "group.jpg").unlink()  # fewer references the second time around
    faces.enroll(db, "kid", refs, det, None, crop_root)
    second = sorted(p.name for p in (crop_root / "kid" / "yunet").iterdir())

    assert len(second) == 1, f"stale crops survived: {second}"
    assert second == ["00_single.jpg"]


def test_failed_re_enroll_keeps_the_previous_gallery(tmp_path, monkeypatch):
    """A run that enrols nothing must not leave the person with an empty gallery."""
    refs = _enroll_fixture(tmp_path, monkeypatch)
    db = faces.open_db(tmp_path / "faces.db")
    det = types.SimpleNamespace(name="yunet", score=0.6, faces=lambda img: [])
    crop_root = tmp_path / ".cache"
    faces.enroll(db, "kid", refs, det, None, crop_root)
    before = db.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
    crops_before = sorted(p.name for p in (crop_root / "kid" / "yunet").iterdir())
    assert before == 2

    # now every remaining reference has two faces: nothing can be enrolled
    (refs / "single.jpg").unlink()
    monkeypatch.setattr(
        faces,
        "detect",
        lambda img, det, min_width=20: [
            ((0, 0, 60, 60), np.array([[9.0, 9.0]] * 5), 0.8),
            ((0, 0, 20, 20), np.array([[1.0, 1.0]] * 5), 0.7),
        ],
    )
    with pytest.raises(SystemExit, match="only her face"):
        faces.enroll(db, "kid", refs, det, None, crop_root)

    assert db.execute("SELECT COUNT(*) FROM refs").fetchone()[0] == before
    assert sorted(p.name for p in (crop_root / "kid" / "yunet").iterdir()) == crops_before


def test_verdict_and_badges():
    assert faces.verdict({"faces": 2, "score": 0.50}, 0.45) == "matched"
    assert faces.verdict({"faces": 2, "score": 0.44}, 0.45) == "below"
    assert faces.verdict({"faces": 0, "score": 0.00}, 0.45) == "unknown"
    assert faces.BADGES["matched"] == ("ok", "MATCH")
    assert faces.BADGES["unknown"] == ("warn", "no face")


def test_face_box_style_is_percentage_of_the_image():
    style = faces.face_box_style((150, 50, 75, 100), 750, 500)
    assert style == "left:20.00%;top:10.00%;width:10.00%;height:20.00%"
    assert faces.face_box_style(None, 750, 500) == ""
    assert faces.face_box_style((1, 2, 3, 4), 0, 0) == ""


def test_image_src_encodes_for_urls_and_decodes_back_to_the_file(tmp_path):
    import urllib.parse

    sheet = tmp_path / ".cache" / "report.html"
    photo = tmp_path / "downloads" / "2026-05-11_〈全クラス〉春" / "IMG_5706%281%29.JPG"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"x")

    src = faces.image_src(photo, sheet)

    # a literal % in a filename must be %25 in the URL, or the browser reads %28 as "("
    assert "IMG_5706%25281%2529.JPG" in src
    assert "〈" not in src, "non-ASCII is percent-encoded for the URL"
    assert urllib.parse.unquote(src) == "../downloads/2026-05-11_〈全クラス〉春/IMG_5706%281%29.JPG"
    assert (sheet.parent / urllib.parse.unquote(src)).resolve() == photo.resolve(), "URL resolves to the file"


def _report_rows(tmp_path, scores=(0.6, 0.4, 0.0)):
    album = tmp_path / "downloads" / "album"
    album.mkdir(parents=True)
    rows = []
    for i, score in enumerate(scores):
        path = album / f"p{i}.jpg"
        path.write_bytes(b"x")
        rows.append(
            {
                "album": "album",
                "pipeline": PIPE,
                "path": path,
                "root": tmp_path / "downloads",
                "rel": f"album/{path.name}",
                "override": None,
                "score": score,
                "faces": 0 if score == 0.0 else 1,
                "face_width": 60.0 if score else 0.0,
                "best_box": (100, 50, 60, 80) if score else None,
                "width": 750,
                "height": 500,
            }
        )
    return rows


def test_report_links_photos_and_badges_every_state(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    out = tmp_path / ".cache" / "report-kid-yunet.html"

    faces.write_report(db, "kid", _report_rows(tmp_path), out, 0.45, 40)
    html_text = out.read_text()

    assert "data:image" not in html_text, "images must be referenced, not embedded"
    assert 'src=""' not in html_text, "no broken image for photos without a face"
    assert "../downloads/album/p0.jpg" in html_text
    assert ">MATCH<" in html_text and ">below<" in html_text and ">no face<" in html_text
    assert 'class="box" style="left:13.33%;top:10.00%;width:8.00%;height:16.00%"' in html_text
    assert out.stat().st_size < 20_000, "sheet stays small"

    # the verdict is carried by the card border, not only by the badge text
    assert 'class="card matched"' in html_text
    assert 'class="card below"' in html_text
    assert 'class="card unknown"' in html_text
    assert ".card.matched{border-left-color:#5ec26a}" in html_text
    assert "font:700 13px" in html_text, "badge is 13px bold"
    # the band label is gone; the path is a tooltip, not a visible line
    assert "0.40-0.45" not in html_text and "0.60-0.65" not in html_text
    assert 'class="path"' not in html_text
    assert 'title="album/p0.jpg"' in html_text


def test_dispute_kind_separates_detection_from_boundary_noise():
    same_two = {"faces": 2, "score": 0.46}
    assert faces.dispute_kind({"faces": 2, "score": 0.44}, same_two) == "boundary"
    assert faces.dispute_kind({"faces": 0, "score": 0.0}, {"faces": 1, "score": 0.6}) == "detection"
    assert faces.dispute_kind({"faces": 5, "score": 0.29}, {"faces": 7, "score": 0.45}) == "detection"
    assert faces.dispute_kind({"faces": 2, "score": 0.30}, same_two) == "score"


def _pair_rows(tmp_path, specs):
    """specs: list of (name, score_a, faces_a, score_b, faces_b)."""
    album = tmp_path / "downloads" / "album"
    album.mkdir(parents=True, exist_ok=True)
    a_rows, b_rows = [], []
    for name, sa, fa, sb, fb in specs:
        path = album / name
        path.write_bytes(b"x")
        for rows, score, n in ((a_rows, sa, fa), (b_rows, sb, fb)):
            rows.append(
                {
                    "album": "album",
                    "pipeline": PIPE if rows is a_rows else OTHER_PIPE,
                    "path": path,
                    "root": tmp_path / "downloads",
                    "score": score,
                    "faces": n,
                    "face_width": 60.0 if n else 0.0,
                    "best_box": (100, 50, 60, 80) if n else None,
                    "width": 750,
                    "height": 500,
                }
            )
    return {PIPE: a_rows, OTHER_PIPE: b_rows}


def test_disagreements_carry_their_kind(tmp_path):
    rows = _pair_rows(
        tmp_path,
        [
            ("detect.jpg", 0.20, 2, 0.60, 1),  # counts differ
            ("noise.jpg", 0.44, 2, 0.46, 2),  # within 0.05
            ("gap.jpg", 0.30, 2, 0.70, 2),  # same faces, big gap
            ("agree.jpg", 0.60, 2, 0.62, 2),  # no dispute
        ],
    )
    found = {d["name"]: d["kind"] for d in faces.disagreements(rows, 0.45)}
    assert found == {"detect.jpg": "detection", "noise.jpg": "boundary", "gap.jpg": "score"}


def test_comparison_groups_counts_and_seeded_samples(tmp_path):
    specs = [("d.jpg", 0.2, 2, 0.6, 1), ("n.jpg", 0.44, 2, 0.46, 2), ("g.jpg", 0.3, 2, 0.7, 2)]
    specs += [(f"both{i}.jpg", 0.6, 1, 0.6, 1) for i in range(5)]
    specs += [(f"none{i}.jpg", 0.1, 1, 0.1, 1) for i in range(5)]
    rows = _pair_rows(tmp_path, specs)

    groups, sampled, totals = faces.comparison_groups(rows, 0.45, sample=3, seed=7)

    assert {k: len(v) for k, v in groups.items()} == {"detection": 1, "score": 1, "boundary": 1}
    assert totals == {"both": 5, "neither": 5}
    assert len(sampled["both"]) == 3 and len(sampled["neither"]) == 3
    again = faces.comparison_groups(rows, 0.45, sample=3, seed=7)
    assert [k for k, _a, _b in sampled["both"]] == [k for k, _a, _b in again[1]["both"]], "sampling is seeded"


def test_comparison_sheet_renders_two_panes_and_no_embedded_images(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    rows = _pair_rows(tmp_path, [("one.jpg", 0.2, 2, 0.6, 1), ("two.jpg", 0.6, 1, 0.6, 1)])
    for pipeline, pipeline_rows in rows.items():
        for row in pipeline_rows:
            faces.store_photo(
                db,
                faces.file_sha1(row["path"]),
                (750, 500),
                pipeline,
                [((100, 50, 60, 80), 0.9, np.ones(4, dtype=np.float32))],
            )
    stats = faces.compare(rows, 0.45)
    out = tmp_path / ".cache" / "compare-kid-yunet-scrfd.html"

    faces.write_comparison(db, rows, stats, out, 0.45, 40, sample=1)
    html_text = out.read_text()

    assert html_text.count('class="pane-a"') == 2 and html_text.count('class="pane-b"') == 2
    assert html_text.count('class="box best"') == 4, "each pane draws its best face"
    assert "data:image" not in html_text and 'src=""' not in html_text
    assert "../downloads/album/one.jpg" in html_text
    assert ">detection<" in html_text or 'class="kind">detection<' in html_text
    assert "<details>" in html_text, "agreed buckets are folded away"
    assert "Detection differences — the detectors saw a different number of faces" in html_text


def test_ensure_model_extracts_buffalo_members_into_the_cache_only(tmp_path, monkeypatch):
    """No path outside the models dir is consulted: archive -> extract -> verify."""
    import io
    import zipfile

    payload = b"fake onnx bytes"
    digest = hashlib.sha256(payload).hexdigest()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("w600k_r50.onnx", payload)
    archive_bytes = buffer.getvalue()

    monkeypatch.setattr(faces, "MODELS", {**faces.MODELS, "arcface": ("http://unused", digest)})
    monkeypatch.setattr(faces, "BUFFALO_MEMBERS", {"arcface": "w600k_r50.onnx"})
    monkeypatch.setattr(faces, "MODEL_FILES", {**faces.MODEL_FILES, "arcface": "arcface_r50.onnx"})
    monkeypatch.setattr(faces.Path, "home", classmethod(lambda cls: tmp_path / "nohome"))

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive_bytes)

    monkeypatch.setattr(faces, "_download", fake_download)

    models_dir = tmp_path / ".cache/models"
    got = faces.ensure_model("arcface", models_dir)

    assert got == models_dir / "arcface_r50.onnx"
    assert got.read_bytes() == payload
    assert not (models_dir / "buffalo_l.zip").exists(), "the archive is cleaned up"
    assert not (tmp_path / "nohome").exists(), "no external model directory is touched"


def test_report_collapses_albums_and_counts_them(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    rows = _report_rows(tmp_path)  # one matched (0.6), one below (0.4), one unusable
    out = tmp_path / ".cache" / "album.html"

    faces.write_report(db, "kid", rows, out, 0.45, 40)
    html_text = out.read_text()

    assert html_text.count("<details class='album'") == 1
    assert "<details class='album' open>" in html_text, "an album with a match starts open"
    assert "3 photos · 1 matched · 1 below (1 within 0.10) · 1 unusable" in html_text
    assert "@media print{details{display:block}" in html_text, "collapsed albums still print"


def test_report_album_without_matches_starts_collapsed(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    rows = _report_rows(tmp_path)
    out = tmp_path / ".cache" / "closed.html"

    faces.write_report(db, "kid", rows, out, 0.95, 40)  # nothing reaches 0.95
    html_text = out.read_text()

    assert "<details class='album' open>" not in html_text
    assert "<details class='album'>" in html_text


def test_overrides_roundtrip_and_clear(tmp_path):
    path = tmp_path / "refs" / "kid" / "overrides.json"
    assert faces.load_overrides(path) == {}, "a missing file means no decisions"

    faces.apply_decision(path, "album/a.jpg", "keep")
    faces.apply_decision(path, "album/b.jpg", "drop")
    assert faces.load_overrides(path) == {"album/a.jpg": "keep", "album/b.jpg": "drop"}
    assert "album/a.jpg" in path.read_text(), "the key is the readable path, not a hash"

    faces.apply_decision(path, "album/a.jpg", "clear")
    assert faces.load_overrides(path) == {"album/b.jpg": "drop"}
    with pytest.raises(ValueError, match="unknown action"):
        faces.apply_decision(path, "album/b.jpg", "maybe")


def test_override_changes_the_verdict_both_ways():
    low = {"faces": 1, "score": 0.10, "override": "keep"}
    high = {"faces": 3, "score": 0.90, "override": "drop"}
    plain = {"faces": 3, "score": 0.90, "override": None}
    noface = {"faces": 0, "score": 0.0, "override": "keep"}

    assert faces.verdict(low, 0.45) == "included"
    assert faces.verdict(high, 0.45) == "excluded"
    assert faces.verdict(plain, 0.45) == "matched"
    assert faces.verdict(noface, 0.45) == "included", "a kept photo with no detectable face still exports"


def test_matched_rows_applies_overrides(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    rows = []
    for name, score, override in (("keep.jpg", 0.05, "keep"), ("drop.jpg", 0.95, "drop"), ("auto.jpg", 0.60, None)):
        path = album / name
        path.write_bytes(b"x")
        rows.append(
            {
                "album": "album",
                "path": path,
                "root": tmp_path,
                "rel": f"album/{name}",
                "score": score,
                "faces": 1,
                "override": override,
            }
        )

    picked = [r["rel"] for r in faces.matched_rows(rows, 0.45)]
    assert picked == ["album/keep.jpg", "album/auto.jpg"]


def test_unused_overrides_flags_renamed_paths(tmp_path):
    rows = [{"rel": "album/here.jpg"}]
    overrides = {"album/here.jpg": "keep", "album/gone.jpg": "drop"}
    assert faces.unused_overrides(overrides, rows) == ["album/gone.jpg"]


def test_report_marks_manual_decisions(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    rows = _report_rows(tmp_path)
    rows[0]["override"] = "drop"  # the 0.6 match
    rows[2]["override"] = "keep"  # the no-face photo
    out = tmp_path / ".cache" / "manual.html"

    faces.write_report(db, "kid", rows, out, 0.45, 40)
    html_text = out.read_text()

    assert ">DROPPED<" in html_text and ">ADDED<" in html_text
    assert 'class="card excluded"' in html_text
    assert 'class="card included"' in html_text
    assert ".card.included{border-left-color:#6aa9e0}" in html_text


def test_report_can_render_action_buttons(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    out = tmp_path / ".cache" / "interactive.html"

    faces.write_report(db, "kid", _report_rows(tmp_path), out, 0.45, 40, interactive=True)
    html_text = out.read_text()

    assert html_text.count('class="actions"') == 3, "one action row per photo card"
    assert 'fetch("/decide"' in html_text
    assert "album/p0.jpg" in html_text, "the button payload is the readable key"


def test_render_report_can_serve_images_from_the_server(tmp_path):
    """The review server passes src_for, so the page never emits a filesystem path."""
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    out = tmp_path / ".cache" / "served.html"

    faces.write_report(
        db,
        "kid",
        _report_rows(tmp_path),
        out,
        0.45,
        40,
        interactive=True,
        src_for=lambda path: "/photos/" + path.name,
    )
    html_text = out.read_text()

    assert 'src="/photos/p0.jpg"' in html_text
    assert "../downloads/album" not in html_text, "no relative paths when served over http"


def _reviewed_rows():
    """Three keeps (two below any cutoff), three drops (two above it)."""

    def row(name, score, override):
        return {"rel": f"album/{name}", "album": "album", "score": score, "faces": 1, "override": override}

    return [
        row("k1.jpg", 0.20, "keep"),
        row("k2.jpg", 0.40, "keep"),
        row("k3.jpg", 0.70, "keep"),
        row("d1.jpg", 0.30, "drop"),
        row("d2.jpg", 0.60, "drop"),
        row("d3.jpg", 0.90, "drop"),
        row("u1.jpg", 0.80, None),  # unreviewed: not counted as ground truth
    ]


def test_evaluate_counts_what_each_threshold_agrees_with():
    table = {row["threshold"]: row for row in faces.evaluate(_reviewed_rows(), edges=(0.45, 0.55))}

    assert table[0.45]["keeps"] == 3 and table[0.45]["drops"] == 3
    assert table[0.45]["keeps_matched"] == 1, "only k3 clears 0.45 without a pin"
    assert table[0.45]["drops_matched"] == 2, "d2 and d3 would leak in at 0.45"
    assert table[0.55]["keeps_matched"] == 1
    assert table[0.55]["drops_matched"] == 2


def test_evaluate_report_states_reviewed_and_pinned_counts():
    text = faces.evaluate_report(_reviewed_rows(), edges=(0.45,))
    assert "6 reviewed of 7 photos (3 keep, 3 drop)" in text, "unreviewed rows are excluded"
    assert "keeps found" in text and "drops leaked" in text
    assert "only your keep decision saves them" in text
    assert "only your drop keeps them out" in text
    assert "keeps found + keeps pinned" not in text, "the legend explains, it does not restate the formula twice"
    assert "0.45" in text
