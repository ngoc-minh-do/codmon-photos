#!/usr/bin/env python3
"""Fetch best-resolution photos from a Codmon API response JSON into
./downloads/<date>.

Usage:
  uv run scripts/codmon_sync.py <response.json> [--out DIR] [--date YYYY-MM-DD]
                                [--workers N]

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
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MAX_AXIS = 500
COMBO_H = f"&width=0&height={MAX_AXIS}"  # landscape best candidate
COMBO_W = f"&width={MAX_AXIS}&height=0"  # portrait best candidate
PROBE = f"&width=10&height=0"


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


def collect_photos(obj, out):
    if isinstance(obj, dict):
        if isinstance(obj.get("url"), str) and "codmon.com" in obj["url"]:
            out.append((obj.get("id"), obj["url"]))
        for v in obj.values():
            collect_photos(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_photos(v, out)
    return out


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
        raise SystemExit("no photos classified (signed URLs expired?); need a fresh response.json")

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
    for pid, url, _dim in entries:
        names.append(name_of(url).replace("/", "_"))
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    dups = sorted(n for n, c in counts.items() if c > 1)
    if dups:
        print(f"duplicate basenames detected, prefixing id: {dups}")
    out, seen = [], set()
    for (pid, url, _dim), base in zip(entries, names):
        final = (f"{pid}_{base}" if counts[base] > 1 else base)
        n = 1
        while final in seen:
            n += 1
            final = f"{pid}_{n}_{base}"
        seen.add(final)
        out.append((pid, url, final))
    return out, dups


def download_one(task, combo, outdir):
    pid, url, final = task
    base = best_url(url)
    d = fetch(base + combo)
    if not d:
        return pid, None, None, None
    dim = jpeg_dim(d)
    path = os.path.join(outdir, final)
    with open(path, "wb") as f:
        f.write(d)
    return pid, f"{dim[0]}x{dim[1]}" if dim else "?", len(d), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("response_json")
    ap.add_argument("--out", default="downloads")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    photos = collect_photos(json.load(open(args.response_json)), [])
    if not photos:
        raise SystemExit("no Codmon photo URLs found in the JSON")
    outdir = os.path.join(args.out, args.date)
    os.makedirs(outdir, exist_ok=True)

    print("1/3 classifying orientations (tiny probes)...")
    entries = classify(photos, args.workers)

    print("2/3 calibrating best combo on 1 landscape + 1 portrait...")
    lc, pc = calibrate(entries)

    print("3/3 downloading all with calibrated combos...")
    planned, dups = plan_names(entries)
    choice = {True: lc, False: pc}
    tasks = [
        ((pid, url, final), choice[bool(dim and dim[0] > dim[1])])
        for (pid, url, dim), (_, _, final) in zip(entries, planned)
    ]
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pid, dim, nbytes, path in ex.map(
            lambda t: download_one(t[0], t[1], outdir), tasks
        ):
            print(f"{'OK' if path else 'FAIL'} {dim} {nbytes // 1024 if nbytes else 0}k {os.path.basename(path) if path else pid}")
            if path:
                ok += 1
    print(f"--- downloaded {ok}/{len(photos)} to {outdir}")


if __name__ == "__main__":
    main()