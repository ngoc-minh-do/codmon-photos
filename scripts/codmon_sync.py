#!/usr/bin/env python3
"""Download photos from Codmon at the best resolution the CDN allows into
./downloads/<date>, fetching albums directly from the Codmon API (no
response.json needed).

Usage:
  uv run scripts/codmon_sync.py [--out DIR] [--date YYYY-MM-DD]
                                [--lookback N] [--workers N] [--tz +09:00]

Credentials come from CODMON_EMAIL / CODMON_PASSWORD (a .env is auto-loaded,
see .env.example). Without --date, the timeline is scanned back --lookback days
(default 45) and every photo album found is downloaded into a folder named by
its display_date.

Pipeline (does not search best resolution on every photo):
  1. Classify: one tiny probe fetch (`width=10`) per photo to learn its
     orientation cheaply (aspect is preserved at any size).
  2. Calibrate: take ONE landscape and ONE portrait photo, fetch both size
     combos (`width=0&height=500` vs `width=500&height=0`) and pick the combo
     that yields the larger image for each orientation. Single-orientation
     albums calibrate the present orientation and assume the default combo for
     the missing one.
  3. Download: fetch every photo once, using the calibrated best combo for its
     orientation.

The CDN strips all EXIF on re-encode, so after download each photo is stamped
with DateTimeOriginal + OffsetTimeOriginal from the album's insert_datetime
(Tokyo time by default) so apps (Windows Explorer, Immich, ...) show the right
capture date.

Background / best-resolution rules (from live probing):
  - Photo base URLs are signed CloudFront URLs; the *width*/*height* query
    params ride inside the signed resource. Truncate the URL at "&width=" to
    retune size while keeping Policy/Signature/Key-Pair-Id intact.
  - Per-axis cap is 500 (a param >500 or 0&0 -> HTTP 400; omitting the params
    -> HTTP 403). Supplying BOTH params boxes to the smaller — never helps.
  - "width=X&height=0" constrains width, auto height; "width=0&height=Y"
    constrains height, auto width. Constraining the axis that is SMALLER in
    the photo's aspect lets the auto side grow past 500 (e.g. landscape
    4:3 -> 667x500, 3:2 -> 750x500, 16:9 -> 1110x500; portrait -> 500x667 or
    500x890).
"""
import argparse
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import piexif
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
    ap = argparse.ArgumentParser(description=__doc__)
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

    dates = Counter(a.display_date for a in albums)
    total = 0
    for album in albums:
        folder = album.display_date
        if dates[folder] > 1:
            folder = f"{folder}_{album.album_id}"
        exif_dt = to_exif_dt(album.insert_datetime)
        print(
            f"album {album.album_id} {album.title!r} ({len(album.photos)} photos) "
            f"display={album.display_date} insert={album.insert_datetime}"
        )
        if exif_dt:
            print(f"   -> EXIF DateTimeOriginal {exif_dt} (tz {args.tz})")
        total += run_album(album.photos, folder, exif_dt, args.out, args.workers, args.tz)
    print(f"done: {total} photos across {len(albums)} album(s) under {args.out}/")


if __name__ == "__main__":
    main()