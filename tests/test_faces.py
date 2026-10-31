import pathlib
import sqlite3
import types

import faces
import numpy as np
import pytest


def unit(*values):
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


PIPE = "v2:yunet:w600k_r50.onnx:arc112:floor20"
OTHER_PIPE = "v2:scrfd:w600k_r50.onnx:arc112:floor20"


def add_ref(db, person="kid", emb=None, pipeline=PIPE):
    emb = unit(1, 0, 0) if emb is None else emb
    db.execute(
        "INSERT INTO refs (person, pipeline, source, crop, emb) VALUES (?,?,?,?,?)",
        (person, pipeline, "s", "c", emb.tobytes()),
    )


def test_band_of_boundaries():
    assert faces.band_of(0.10) == "<0.35"
    assert faces.band_of(0.35) == "0.35-0.40"
    assert faces.band_of(0.449) == "0.40-0.45"
    assert faces.band_of(0.70) == ">=0.65"


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


def test_pipeline_id_separates_detectors_and_floor():
    yunet = types.SimpleNamespace(name="yunet")
    scrfd = types.SimpleNamespace(name="scrfd")
    assert faces.pipeline_id(yunet) != faces.pipeline_id(scrfd)
    assert faces.pipeline_id(yunet, floor=20) != faces.pipeline_id(yunet, floor=40)
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


def test_enroll_bootstraps_from_the_largest_face_then_flags_disagreement(tmp_path, monkeypatch):
    """With no single-face photo, the largest seeds the gallery; a total mismatch is skipped."""
    import cv2

    refs = tmp_path / "refs" / "kid" / "face"
    refs.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(refs / name), np.zeros((32, 32, 3), dtype=np.uint8))

    faces_for_calls = iter(
        [
            [((0, 0, 60, 60), np.array([[9.0, 9.0]] * 5), 0.8), ((0, 0, 20, 20), np.array([[1.0, 1.0]] * 5), 0.7)],
            [((0, 0, 30, 30), np.array([[2.0, 2.0]] * 5), 0.9), ((0, 0, 25, 25), np.array([[3.0, 3.0]] * 5), 0.9)],
        ]
    )
    monkeypatch.setattr(faces, "detect", lambda img, det, min_width=20: next(faces_for_calls))

    def embed_by_identity(img, kps, rec):
        # face x=9 is the bootstrap kid, x=2/3 belong to someone else (orthogonal)
        return np.array([0.0, 1.0, 0.0] if kps[0][0] == 9.0 else [1.0, 0.0, 0.0], np.float32)

    monkeypatch.setattr(faces, "embed", embed_by_identity)
    monkeypatch.setattr(faces, "align", lambda img, kps, size=112: np.zeros((size, size, 3), np.uint8))
    db = faces.open_db(tmp_path / "faces.db")
    det = types.SimpleNamespace(name="yunet", score=0.6, faces=lambda img: [])

    warnings = faces.enroll(db, "kid", refs, det, None, tmp_path / ".cache")

    assert any("no gallery yet" in w for w in warnings)
    assert any("no face there resembles the others" in w for w in warnings)
    assert db.execute("SELECT COUNT(*) FROM refs").fetchone() == (1,)


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
