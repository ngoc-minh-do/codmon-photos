#!/usr/bin/env python3
"""Score downloaded albums by whether your child's face is in them.

    uv run scripts/face_filter.py enroll <name>        # refs/<name>/face/*.jpg -> gallery
    uv run scripts/face_filter.py scan                 # detect + embed every photo (cached)
    uv run scripts/face_filter.py report <name>        # review sheet + threshold sweep
    uv run scripts/face_filter.py export <name> --threshold 0.50
    uv run scripts/face_filter.py upload <name>            # -> SMB share

References live in ``refs/<name>/face/``; embeddings and scores in
``.cache/faces.db``; models in ``.cache/models``. ``export`` copies matches to
``matched/<name>/<album>/`` and never modifies the album folders; ``upload``
mirrors that folder to ``//<host>/<share>/<name>/<album>/``. See scripts/faces.py
for the pipeline and AGENTS.md for the resolution limits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import faces
import notify
import smb_sync
from dotenv import load_dotenv


def enrolled_pipelines(db, person: str, detector: str) -> list[str]:
    """Cached pipelines this person is already enrolled for on ``detector``."""
    rows = db.execute("SELECT DISTINCT pipeline FROM refs WHERE person = ?", (person,))
    return [row[0] for row in rows if faces.detector_of(row[0]) == detector]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run scripts/face_filter.py enroll kid\n"
            "  uv run scripts/face_filter.py scan\n"
            "  uv run scripts/face_filter.py report kid --threshold 0.45\n"
            "  uv run scripts/face_filter.py export kid --threshold 0.45 --yes\n"
        ),
    )
    ap.add_argument(
        "--photos",
        type=Path,
        nargs="+",
        default=[faces.DEFAULT_PHOTOS],
        help="album root(s) to scan (default: downloads)",
    )
    ap.add_argument("--db", type=Path, default=faces.DEFAULT_DB, help="sqlite cache (default: .cache/faces.db)")
    ap.add_argument("--models", type=Path, default=faces.DEFAULT_MODELS, help="ONNX model dir (default: .cache/models)")
    ap.add_argument("--refs", type=Path, default=faces.DEFAULT_REFS, help="reference root (default: refs)")
    ap.add_argument(
        "--min-face",
        type=float,
        default=faces.DEFAULT_MIN_FACE,
        help=f"ignore faces narrower than this (default {faces.DEFAULT_MIN_FACE}px)",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--detector",
        choices=faces.DETECTORS,
        default=faces.DEFAULT_DETECTOR,
        help=f"face detector to use; scans are cached per detector (default: {faces.DEFAULT_DETECTOR})",
    )
    common.add_argument(
        "--det-score",
        type=float,
        default=None,
        help="detector confidence threshold (default: 0.5 scrfd, 0.6 yunet); part of the cache key",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("enroll", parents=[common], help="embed refs/<name>/face/*.jpg into the gallery")
    p.add_argument("person")
    p.add_argument("--force", action="store_true", help="re-enroll even if references already exist")

    sub.add_parser("scan", parents=[common], help="detect + embed every photo (cached per content+detector)")

    p = sub.add_parser(
        "report", parents=[common], help="write .cache/report-<name>-<detector>.html and print the threshold sweep"
    )
    p.add_argument("person")
    p.add_argument("--threshold", type=float, default=faces.DEFAULT_THRESHOLD)
    p.add_argument("--out", type=Path, default=None, help="sheet path (default: .cache/report-<name>-<detector>.html)")

    p = sub.add_parser("export", parents=[common], help="copy matching photos to matched/<name>/<album>/")
    p.add_argument("person")
    p.add_argument("--threshold", type=float, default=faces.DEFAULT_THRESHOLD)
    p.add_argument("--matched", type=Path, default=faces.DEFAULT_MATCHED, help="output root (default: matched)")
    p.add_argument("--link", action="store_true", help="hardlink instead of copy")
    p.add_argument("--prune", action="store_true", help="delete earlier copies that no longer match")
    p.add_argument("--yes", action="store_true", help="actually write files (default: dry run)")

    p = sub.add_parser(
        "compare",
        help="compare two cached detectors: terminal table + .cache/compare-<name>-<a>-<b>.html",
    )
    p.add_argument("person")
    p.add_argument("--threshold", type=float, default=faces.DEFAULT_THRESHOLD)
    p.add_argument(
        "--detectors",
        nargs=2,
        choices=faces.DETECTORS,
        default=None,
        help="pair to compare (default: whatever two pipelines are cached)",
    )
    p.add_argument("--sample", type=int, default=20, help="photos sampled from the agreed buckets (default 20)")
    p.add_argument("--list-noise", action="store_true", help="also list boundary-noise rows in the terminal")
    p.add_argument("--out", type=Path, default=None, help="sheet path (default: .cache/compare-<name>-<a>-<b>.html)")

    p = sub.add_parser("upload", help="mirror matched/<name>/ to the SMB share (<share>/<name>/<album>/)")
    p.add_argument("person")
    p.add_argument("--matched", type=Path, default=faces.DEFAULT_MATCHED, help="local root (default: matched)")
    p.add_argument("--delete", action="store_true", help="remove remote photos that are no longer in matched/")
    p.add_argument("--yes", action="store_true", help="actually upload (default: dry run)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    db = faces.open_db(args.db)

    if args.cmd == "enroll":
        enrolled = enrolled_pipelines(db, args.person, args.detector)
        if enrolled and not args.force:
            print(
                f"{args.person!r} already enrolled for {args.detector} ({len(enrolled)} pipeline); use --force to redo"
            )
            return 0
        det, rec = faces.load_models(args.detector, args.models, args.det_score)
        warnings = faces.enroll(db, args.person, args.refs / args.person / "face", det, rec, args.db.parent / "refs")
        count = db.execute(
            "SELECT COUNT(*) FROM refs WHERE person = ? AND pipeline = ?", (args.person, faces.pipeline_id(det))
        ).fetchone()[0]
        print(f"enrolled {count} reference photo(s) for {args.person!r} [{faces.pipeline_id(det)}]")
        for warning in warnings:
            print(f"   ! {warning}")
        print(f"   aligned crops: {args.db.parent / 'refs' / args.person}/ (check these are all her)")
        return 0

    if args.cmd == "scan":
        det, rec = faces.load_models(args.detector, args.models, args.det_score)
        done, skipped = faces.scan(db, args.photos, det, rec, faces.DEFAULT_FLOOR)
        print(f"scanned {done} photo(s), {skipped} already cached [{faces.pipeline_id(det)}]")
        return 0

    if args.cmd == "upload":
        cfg = smb_sync.smb_config(os.environ)
        if not cfg:
            raise SystemExit("set SMB_HOST / SMB_SHARE / SMB_USER / SMB_PASSWORD in .env to upload")
        client = smb_sync.client_for(cfg)
        stats = smb_sync.plan_uploads(args.matched, args.person, cfg, client)
        prunable = smb_sync.prune_targets(args.matched, args.person, cfg, client) if args.delete else []
        print(f"{smb_sync.remote_base(cfg, args.person)}/ <- {args.matched / args.person}/")
        print(smb_sync.upload_report(stats, prunable))
        if not args.yes:
            return 0
        uploaded = smb_sync.upload_files(stats["uploads"], client)
        removed = 0
        for remote in prunable:
            client.remove(remote)
            removed += 1
        print(f"uploaded {uploaded}, removed {removed}")
        if uploaded or removed:
            notify.notify(
                os.environ.get("APPRISE_URL", ""),
                "# 👍 codmon photos uploaded",
                smb_sync.upload_body(
                    stats["uploads"], len(stats["skipped"]), removed, smb_sync.remote_base(cfg, args.person)
                ),
            )
        return 0

    if args.cmd == "compare":
        chosen = args.detectors or [faces.detector_of(p) for p in faces.pipelines(db)]
        pair = [p for p in faces.pipelines(db) if faces.detector_of(p) in chosen]
        if len(pair) < 2:
            raise SystemExit(
                f"need two scanned detectors to compare; cached: {[faces.detector_of(p) for p in faces.pipelines(db)]} "
                "— run scan --detector scrfd and scan --detector yunet first"
            )
        rows_by_pipeline = {p: faces.collect(db, args.person, args.photos, args.min_face, p) for p in pair[:2]}
        stats = faces.compare(rows_by_pipeline, args.threshold)
        found = faces.disagreements(rows_by_pipeline, args.threshold)
        print(faces.compare_report(stats, found, args.threshold, args.list_noise))
        a, b = stats["pipelines"]
        out = args.out or args.db.parent / (f"compare-{args.person}-{faces.detector_of(a)}-{faces.detector_of(b)}.html")
        sheet = faces.write_comparison(db, rows_by_pipeline, stats, out, args.threshold, args.min_face, args.sample)
        print(f"wrote {sheet}")
        return 0

    rows = faces.collect(db, args.person, args.photos, args.min_face, faces.pipeline_for(db, args.detector))
    if args.cmd == "report":
        out = args.out or args.db.parent / f"report-{args.person}-{args.detector}.html"
        print("threshold  photos kept  share")
        for threshold, kept, share in faces.sweep(rows):
            mark = "  <- current" if abs(threshold - args.threshold) < 1e-9 else ""
            print(f"   {threshold:.2f}      {kept:5d}     {share:3.0f}%{mark}")
        unknown = sum(1 for row in rows if not row["faces"])
        if unknown:
            print(f"{unknown} photo(s) have no face wider than {args.min_face:.0f}px and are never counted as hers")
        sheet = faces.write_report(db, args.person, rows, out, args.threshold, args.min_face)
        print(f"wrote {sheet}")
        return 0

    stats = faces.export_matches(
        rows, args.person, args.matched, args.threshold, link=args.link, prune=args.prune, apply=args.yes
    )
    target = args.matched / args.person
    for row in faces.matched_rows(rows, args.threshold):
        print(f"   {row['album']}/{row['path'].name} ({row['score']:.2f})")
    tail = f" (+{stats['pruned']} pruned)" if stats["pruned"] else ""
    if not args.yes:
        print(f"dry run: would copy {stats['copied']} photo(s) to {target}/, {stats['kept']} already there{tail}")
        return 0
    print(f"copied {stats['copied']} photo(s) to {target}/, {stats['kept']} already there{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
