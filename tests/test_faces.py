import sqlite3

import faces
import numpy as np
import pytest


def unit(*values):
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def add_ref(db, person="kid", emb=None):
    emb = unit(1, 0, 0) if emb is None else emb
    db.execute("INSERT INTO refs (person, source, crop, emb) VALUES (?,?,?,?)", (person, "s", "c", emb.tobytes()))


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
    faces.store_photo(db, "abc", (750, 500), "w600k_r50", [((1.0, 2.0, 33.0, 44.0), 0.91, emb)])
    loaded = faces.load_faces(db, "abc")
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
    faces.store_photo(db, sha1, (10, 10), "w600k_r50", [])
    assert faces.cached_photo(db, faces.file_sha1(two), "w600k_r50"), "same bytes must hit the cache"
    assert not faces.cached_photo(db, sha1, "other-model")
    two.write_bytes(b"\xff\xd8differs")
    assert not faces.cached_photo(db, faces.file_sha1(two), "w600k_r50")


def test_collect_scores_albums_from_the_cache(tmp_path):
    album = tmp_path / "downloads" / "2026-09-25"
    album.mkdir(parents=True)
    her, other = album / "her.jpg", album / "other.jpg"
    her.write_bytes(b"\xff\xd8her")
    other.write_bytes(b"\xff\xd8other")
    db = faces.open_db(tmp_path / "faces.db")
    add_ref(db)
    faces.store_photo(db, faces.file_sha1(her), (750, 500), "w600k_r50", [((0, 0, 60, 60), 0.9, unit(1, 0, 0))])
    faces.store_photo(db, faces.file_sha1(other), (750, 500), "w600k_r50", [((0, 0, 30, 30), 0.9, unit(0, 1, 0))])

    rows = faces.collect(db, "kid", [tmp_path / "downloads"], min_face=40)

    by_name = {r["path"].name: r for r in rows}
    assert by_name["her.jpg"]["score"] == pytest.approx(1.0)
    assert by_name["her.jpg"]["album"] == "2026-09-25"
    assert by_name["her.jpg"]["root"] == tmp_path / "downloads"
    assert by_name["other.jpg"]["score"] == pytest.approx(0.0)
    assert by_name["other.jpg"]["faces"] == 0, "a 30px face is below --min-face 40"


def test_gallery_requires_references(tmp_path):
    db = faces.open_db(tmp_path / "faces.db")
    with pytest.raises(SystemExit, match="enroll"):
        faces.gallery(db, "nobody")


def test_open_db_is_idempotent(tmp_path):
    path = tmp_path / "faces.db"
    faces.open_db(path).close()
    db = sqlite3.connect(path)
    assert db.execute("SELECT COUNT(*) FROM photos").fetchone() == (0,)
