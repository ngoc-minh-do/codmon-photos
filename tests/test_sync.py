import struct

import codmon_sync as m
import piexif


def make_jpeg(width, height):
    # JPEG segment lengths include their own 2-byte length field.
    sof_body = struct.pack(">BHHB", 8, height, width, 3) + (b"\x01\x11\x00" * 3)
    sof = b"\xff\xc0" + struct.pack(">H", len(sof_body) + 2) + sof_body
    sos = b"\xff\xda" + struct.pack(">H", 12) + b"\x03" + (b"\x01\x00" * 3) + b"\x00\x3f\x00"
    return b"\xff\xd8" + sof + sos + b"\xff\xd9"


def test_jpeg_dim():
    assert m.jpeg_dim(make_jpeg(667, 500)) == (667, 500)
    assert m.jpeg_dim(b"") is None
    assert m.jpeg_dim(b"\xff\xd8\xff\xd9") is None


def test_safe_title():
    assert m.safe_title("〈０歳児〉9月の活動写真") == "〈０歳児〉9月の活動写真"
    assert m.safe_title("a<b:c*d?") == "a_b_c_d"
    assert m.safe_title("   ") == "album"
    assert m.safe_title("") == "album"


def test_to_exif_dt():
    assert m.to_exif_dt("2026-10-05 11:39:12") == "2026:10:05 11:39:12"
    assert m.to_exif_dt("") is None
    assert m.to_exif_dt(None) is None


def test_plan_names_disambiguates_duplicates(tmp_path):
    entries = [
        ("1", "https://image.codmon.com/codmon/1/albums/IMG_A.JPG?w", (10, 10)),
        ("2", "https://image.codmon.com/codmon/1/albums/IMG_A.JPG?w", (10, 10)),
    ]
    planned, dups = m.plan_names(entries)
    assert dups == ["IMG_A.JPG"]
    assert planned[0][2] != planned[1][2]
    assert {p[2] for p in planned} == {"1_IMG_A.JPG", "2_IMG_A.JPG"}


def test_target_has_files_local(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    full = tmp_path / "full"
    full.mkdir()
    (full / "a.jpg").write_bytes(b"x")
    assert not m.target_has_files(str(empty), False)
    assert m.target_has_files(str(full), False)
    assert not m.target_has_files(str(tmp_path / "missing"), False)


def test_smb_config(monkeypatch):
    monkeypatch.delenv("SMB_HOST", raising=False)
    monkeypatch.delenv("SMB_SHARE", raising=False)
    monkeypatch.delenv("SMB_USER", raising=False)
    monkeypatch.delenv("SMB_PASSWORD", raising=False)
    assert m.smb_config({}) is None
    try:
        m.smb_config({"SMB_HOST": "x"})
    except SystemExit:
        pass
    else:
        raise AssertionError("partial SMB config should fail")


def test_stamp_exif(tmp_path):
    path = tmp_path / "stamped.jpg"
    path.write_bytes(make_jpeg(500, 667))
    m.stamp_exif(str(path), "2026:10:05 11:39:12", "+09:00")
    exif = piexif.load(str(path))
    assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:05 11:39:12"
    assert exif["Exif"][piexif.ExifIFD.OffsetTimeOriginal] == b"+09:00"
