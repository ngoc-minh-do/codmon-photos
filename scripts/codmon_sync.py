#!/usr/bin/env python3
"""Fetch best-resolution photos from a Codmon API response JSON and
optionally upload them to an anonymous SMB share under a date folder.

Usage:
  uv run scripts/codmon_sync.py <response.json> [--out DIR] [--host HOST] [--share SHARE]
                                [--date YYYY-MM-DD] [--no-upload] [--workers N]

Pipeline (does not search best resolution on every photo):
  1. Classify: one tiny probe fetch (`width=10`) per photo to learn its
     orientation cheaply (aspect is preserved at any size).
  2. Calibrate: take ONE landscape and ONE portrait photo, fetch both size
     combos (`width=0&height=500` vs `width=500&height=0`) and pick the combo
     that yields the larger image for each orientation.
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

from smb.SMBConnection import SMBConnection  # provided by uv venv (.venv)

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
    """Pick one landscape + one portrait, find the best combo for each."""
    lands = [e for e in entries if e[2] and e[2][0] > e[2][1]]
    ports = [e for e in entries if e[2] and e[2][0] < e[2][1]]
    if not lands or not ports:
        raise SystemExit("need at least one landscape and one portrait photo to calibrate")
    land = lands[0]
    port = ports[0]

    def best(entry):
        base = best_url(entry[1])
        hw, hb = probe_combo(base, COMBO_H)
        ww, wb = probe_combo(base, COMBO_W)
        h_area = (hw[0] * hw[1]) if hw else 0
        w_area = (ww[0] * ww[1]) if ww else 0
        return COMBO_H if h_area >= w_area else COMBO_W, hw, ww

    lc, lh, lw = best(lands[0])
    pc, ph, pw = best(ports[0])
    print(f"calibrate landscape {lands[0][0]}: H={lh} W={lw} -> {lc}")
    print(f"calibrate portrait  {ports[0][0]}: H={ph} W={pw} -> {pc}")
    return lc, pc


def name_of(url):
    return f"{url.split('/albums/')[-1].split('?')[0]}"


def download_one(item, combo, outdir):
    pid, url = item
    base = best_url(url)
    d = fetch(base + combo)
    if not d:
        return pid, None, None, None
    dim = jpeg_dim(d)
    path = os.path.join(outdir, f"{pid}_{name_of(url)}".replace("/", "_"))
    with open(path, "wb") as f:
        f.write(d)
    return pid, f"{dim[0]}x{dim[1]}" if dim else "?", len(d), path


def smb_put(host, port, share, date, files):
    c = SMBConnection("", "", "lab", "host", use_ntlm_v2=True, is_direct_tcp=True)
    if not c.connect(host, port, timeout=15):
        raise SystemExit(f"cannot connect to SMB {host}:{port}")
    try:
        c.createDirectory(share, f"/{date}")
    except Exception:
        pass
    n = 0
    for path in files:
        with open(path, "rb") as f:
            c.storeFile(share, f"/{date}/{os.path.basename(path)}", f, timeout=120)
        n += 1
    c.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("response_json")
    ap.add_argument("--out", default="downloads")
    ap.add_argument("--host", default="192.168.0.100")
    ap.add_argument("--port", type=int, default=445)
    ap.add_argument("--share", default="share")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    photos = collect_photos(json.load(open(args.response_json)), [])
    if not photos:
        raise SystemExit("no Codmon photo URLs found in the JSON")
    os.makedirs(args.out, exist_ok=True)

    print("1/3 classifying orientations (tiny probes)...")
    entries = classify(photos, args.workers)

    print("2/3 calibrating best combo on 1 landscape + 1 portrait...")
    lc, pc = calibrate(entries)

    print("3/3 downloading all with calibrated combos...")
    paths, ok = [], 0
    tasks = [(pid, url, (lc if (dim and dim[0] > dim[1]) else pc)) for pid, url, dim in entries]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pid, dim, nbytes, path in ex.map(
            lambda t: download_one((t[0], t[1]), t[2], args.out), tasks
        ):
            print(f"{'OK' if path else 'FAIL'} {dim} {nbytes // 1024 if nbytes else 0}k {os.path.basename(path) if path else pid}")
            if path:
                ok += 1
                paths.append(path)
    print(f"--- downloaded {ok}/{len(photos)}")

    if ok and not args.no_upload:
        n = smb_put(args.host, args.port, args.share, args.date, paths)
        print(f"--- uploaded {n} files to //{args.host}/{args.share}/{args.date}")
    elif ok:
        print(f"--- upload skipped (--no-upload); files in {args.out}")


if __name__ == "__main__":
    main()