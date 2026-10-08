#!/usr/bin/env python3
"""Download Codmon photo albums at the best resolution the CDN allows.

Albums are fetched straight from the Codmon parent API (no response.json) and
downloaded into ./downloads/<display_date>_<album_title>. See AGENTS.md for the
calibration pipeline and CDN rules.
"""
import argparse
import os
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import piexif
import smbclient
from codmon_api import CodmonClient, CodmonError
from dotenv import load_dotenv

MAX_AXIS = 500
COMBO_H = f"&width=0&height={MAX_AXIS}"  # landscape best candidate
COMBO_W = f"&width={MAX_AXIS}&height=0"  # portrait best candidate
PROBE = "&width=10&height=0"
DEFAULT_TZ = "+09:00"  # insert_datetime is Tokyo time


def jpeg_dim(data):
    i, t = 2, len(data)
    while i + 9 < t:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xD8 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        ln = int.from_bytes(data[i + 2 : i + 4], "big")
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return int.from_bytes(data[i + 7 : i + 9], "big"), int.from_bytes(
                data[i + 5 : i + 7], "big"
            )
        i += 2 + ln
    return None


def fetch(url, retries=3):
    for a in range(retries):
        r = subprocess.run(["curl", "-sS", "-m", "60", url], capture_output=True)
        if r.returncode == 0 and r.stdout[:2] == b"\xff\xd8":
            return r.stdout
        time.sleep(1 + a)
    return None


def best_url(url):
    return url.split("&width=")[0]


def classify(items, workers):
    """Tiny probe each photo for orientation. -> list of (pid, url, (w,h) or None)"""
    def probe(item):
        pid, url = item
        d = fetch(best_url(url) + PROBE)
        return pid, url, (jpeg_dim(d) if d else None)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(probe, items))


def probe_combo(base, combo):
    d = fetch(base + combo)
    dim = jpeg_dim(d) if d else None
    return dim, len(d) if d else 0


def calibrate(entries):
    """Find the best combo for each orientation present; assume the default
    combo for orientations missing from the album (single-orientation albums
    are valid - every photo just shares one combo)."""
    lands = [e for e in entries if e[2] and e[2][0] > e[2][1]]
    ports = [e for e in entries if e[2] and e[2][0] < e[2][1]]
    if not lands and not ports:
        raise SystemExit("no photos classified (signed URLs expired?); re-run to get fresh URLs")

    def best(entry):
        base = best_url(entry[1])
        hw, hb = probe_combo(base, COMBO_H)
        ww, wb = probe_combo(base, COMBO_W)
        h_area = (hw[0] * hw[1]) if hw else 0
        w_area = (ww[0] * ww[1]) if ww else 0
        return COMBO_H if h_area >= w_area else COMBO_W, hw, ww

    if lands:
        lc, lh, lw = best(lands[0])
        print(f"calibrate landscape {lands[0][0]}: H={lh} W={lw} -> {lc}")
    else:
        lc = COMBO_H
        print(f"no landscape photos; assuming {lc}")
    if ports:
        pc, ph, pw = best(ports[0])
        print(f"calibrate portrait  {ports[0][0]}: H={ph} W={pw} -> {pc}")
    else:
        pc = COMBO_W
        print(f"no portrait photos; assuming {pc}")
    return lc, pc


def name_of(url):
    return f"{url.split('/albums/')[-1].split('?')[0]}"


def plan_names(entries):
    """Filename per photo; detect duplicate basenames BEFORE download and
    disambiguate by prefixing the id so no file silently overwrites another."""
    names = []
    for _pid, url, _dim in entries:
        names.append(name_of(url).replace("/", "_"))
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    dups = sorted(n for n, c in counts.items() if c > 1)
    if dups:
        print(f"duplicate basenames detected, prefixing id: {dups}")
    out, seen = [], set()
    for (pid, url, _dim), base in zip(entries, names, strict=True):
        final = (f"{pid}_{base}" if counts[base] > 1 else base)
        n = 1
        while final in seen:
            n += 1
            final = f"{pid}_{n}_{base}"
        seen.add(final)
        out.append((pid, url, final))
    return out, dups


def download_one(task, combo, outdir, exif_dt, tz):
    pid, url, final = task
    base = best_url(url)
    d = fetch(base + combo)
    if not d:
        return pid, None, None, None
    dim = jpeg_dim(d)
    path = os.path.join(outdir, final)
    with open(path, "wb") as f:
        f.write(d)
    if exif_dt:
        stamp_exif(path, exif_dt, tz)
    return pid, f"{dim[0]}x{dim[1]}" if dim else "?", len(d), path


def stamp_exif(path, dt, tz):
    """Embed DateTimeOriginal (+OffsetTimeOriginal) into the saved JPEG so
    apps that export/sub-group by capture date (Immich, Windows Explorer, ...)
    show the album's insert_datetime instead of the download time."""
    exif = {
        "0th": {},
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: dt,
            piexif.ExifIFD.DateTimeDigitized: dt,
            piexif.ExifIFD.OffsetTimeOriginal: tz,
        },
        "GPS": {},
        "Interop": {},
        "1st": {},
    }
    piexif.insert(piexif.dump(exif), path)


def to_exif_dt(raw):
    """'YYYY-MM-DD HH:MM:SS' -> EXIF 'YYYY:MM:DD HH:MM:SS', or None."""
    if not raw or len(raw) < 10:
        return None
    return raw.replace("-", ":", 2)


def timezone():
    name = (os.environ.get("TZ") or "Asia/Tokyo").lstrip(":")
    return ZoneInfo(name) if name else ZoneInfo("Asia/Tokyo")


def safe_title(title):
    """Albums name directories; strip characters that break cross-platform
    filesystems (Windows forbids < > : \" / \\ | ? *)."""
    keep = []
    for ch in str(title or "").strip():
        keep.append("_" if ch in '<>:"/\\|?*' else ch)
    name = "".join(keep).strip(" _.　")
    return name or "album"


SMB_ENV = ("SMB_HOST", "SMB_SHARE", "SMB_USER", "SMB_PASSWORD")


def smb_config(cfg):
    """Resolve the SMB_* env vars. Returns {host, share, user, password} or
    None when none are set. Errors out if only some are set."""
    values = {key: cfg.get(key, "").strip() for key in SMB_ENV}
    if not any(values.values()):
        return None
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise SystemExit(
            f"SMB_* partially configured; missing {', '.join(missing)} "
            "(set all four in .env or leave all unset to skip uploads)"
        )
    return values


def smb_upload(local_dir, folder, cfg):
    """Mirror an album folder to //HOST/SHARE/<folder>, preserving local copy."""
    host = cfg["SMB_HOST"]
    target = f"//{host}/{cfg['SMB_SHARE']}/{folder}"
    smbclient.register_session(host, username=cfg["SMB_USER"], password=cfg["SMB_PASSWORD"])
    smbclient.makedirs(target, exist_ok=True)
    uploaded = 0
    for name in sorted(os.listdir(local_dir)):
        local = os.path.join(local_dir, name)
        if not os.path.isfile(local):
            continue
        with open(local, "rb") as src, smbclient.open_file(f"{target}/{name}", "wb") as dst:
            shutil.copyfileobj(src, dst)
        uploaded += 1
    return uploaded


def run_album(photos, folder, exif_dt, out, workers, tz):
    """Run the 3-phase pipeline for one album into downloads/<folder>; returns
    the number of downloaded photos."""
    outdir = os.path.join(out, folder)
    os.makedirs(outdir, exist_ok=True)

    print("1/3 classifying orientations (tiny probes)...")
    entries = classify(photos, workers)

    print("2/3 calibrating best combo on 1 landscape + 1 portrait...")
    lc, pc = calibrate(entries)

    print("3/3 downloading all with calibrated combos...")
    planned, dups = plan_names(entries)
    choice = {True: lc, False: pc}
    tasks = [
        ((pid, url, final), choice[bool(dim and dim[0] > dim[1])])
        for (pid, url, dim), (_, _, final) in zip(entries, planned, strict=True)
    ]
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, dim, nbytes, path in ex.map(
            lambda t: download_one(t[0], t[1], outdir, exif_dt, tz), tasks
        ):
            print(f"{'OK' if path else 'FAIL'} {dim} {nbytes // 1024 if nbytes else 0}k {os.path.basename(path) if path else pid}")
            if path:
                ok += 1
    print(f"--- downloaded {ok}/{len(photos)} to {outdir}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  uv run scripts/codmon_sync.py                      # last ~6 weeks\n"
            "  uv run scripts/codmon_sync.py --date 2026-10-05    # one day\n"
            "  uv run scripts/codmon_sync.py --lookback 220       # ~since March\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default="downloads")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default scans the last --lookback days for albums")
    ap.add_argument("--lookback", type=int, default=45,
                    help="days back from today to scan when --date is omitted (default 45)")
    ap.add_argument("--tz", default=DEFAULT_TZ,
                    help="timezone of insert_datetime, as EXIF OffsetTimeOriginal (default: +09:00 Tokyo)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    load_dotenv()
    email = os.environ.get("CODMON_EMAIL", "").strip()
    password = os.environ.get("CODMON_PASSWORD", "").strip()
    if not email or not password:
        raise SystemExit("set CODMON_EMAIL and CODMON_PASSWORD (see .env.example)")

    if args.date:
        start = end = args.date
    else:
        today = datetime.now(timezone()).date()
        start = (today - timedelta(days=args.lookback)).isoformat()
        end = today.isoformat()

    print(f"logging in and scanning timeline {start}..{end}...")
    try:
        albums = CodmonClient(email, password).albums(start, end)
    except CodmonError as exc:
        raise SystemExit(f"Codmon API error: {exc}") from exc
    if not albums:
        raise SystemExit(f"no photo albums found in {start}..{end}")

    folders = [f"{a.display_date}_{safe_title(a.title)}" for a in albums]
    name_counts = Counter(folders)
    smb = smb_config(os.environ)
    if smb:
        print(f"SMB mirror enabled -> //{smb['SMB_HOST']}/{smb['SMB_SHARE']}/<album>")
    total = 0
    for album, folder in zip(albums, folders, strict=True):
        if name_counts[folder] > 1:
            folder = f"{folder}_{album.album_id}"
        exif_dt = to_exif_dt(album.insert_datetime)
        print(
            f"album {album.album_id} {album.title!r} ({len(album.photos)} photos) "
            f"display={album.display_date} insert={album.insert_datetime}"
        )
        if exif_dt:
            print(f"   -> EXIF DateTimeOriginal {exif_dt} (tz {args.tz})")
        total += run_album(album.photos, folder, exif_dt, args.out, args.workers, args.tz)
        if smb:
            local = os.path.join(args.out, folder)
            try:
                uploaded = smb_upload(local, folder, smb)
                print(f"   -> uploaded {uploaded} files to //{smb['SMB_HOST']}/{smb['SMB_SHARE']}/{folder}")
            except Exception as exc:
                print(f"   -> SMB upload FAILED for {folder}: {exc}")
    print(f"done: {total} photos across {len(albums)} album(s) under {args.out}/")


if __name__ == "__main__":
    main()