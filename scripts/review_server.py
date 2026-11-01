#!/usr/bin/env python3
"""Mark photos keep/drop from the browser instead of editing files.

    uv run scripts/review_server.py kid [--threshold 0.45] [--port 8765]

Serves the review sheet on http://127.0.0.1:8765 with keep / drop / clear buttons.
A click writes the decision to ``refs/<person>/overrides.json`` (keyed by
``album/file.JPG``) and updates that card in place, so nothing is copied, renamed
or typed.

Decisions win over the score everywhere: "keep" exports a photo that scored low or
had no detectable face, "drop" removes one that scored well. The server binds to
localhost, serves only its own page, and writes nothing but that one json file.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import faces


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Then open http://127.0.0.1:<port>/ and click keep / drop per photo.",
    )
    ap.add_argument("person")
    ap.add_argument("--photos", type=Path, nargs="+", default=[faces.DEFAULT_PHOTOS])
    ap.add_argument("--db", type=Path, default=faces.DEFAULT_DB)
    ap.add_argument("--refs", type=Path, default=faces.DEFAULT_REFS)
    ap.add_argument("--detector", choices=faces.DETECTORS, default=faces.DEFAULT_DETECTOR)
    ap.add_argument("--threshold", type=float, default=faces.DEFAULT_THRESHOLD)
    ap.add_argument("--min-face", type=float, default=faces.DEFAULT_MIN_FACE)
    ap.add_argument("--port", type=int, default=8765)
    return ap


def build_state(args) -> dict:
    """Score every photo once; decisions are re-read per request on top of that."""
    db = faces.open_db(args.db)
    pipeline = faces.pipeline_for(db, args.detector)
    rows = faces.collect(db, args.person, args.photos, args.min_face, pipeline)
    print(f"scored {len(rows)} photos for {args.person!r}", flush=True)
    return {
        "db": db,
        "args": args,
        "pipeline": pipeline,
        "rows": rows,
        "overrides_file": faces.overrides_path(args.refs, args.person),
    }


def refresh(state: dict) -> list[dict]:
    """Apply the current json decisions to the already-scored rows."""
    overrides = faces.load_overrides(state["overrides_file"])
    for row in state["rows"]:
        row["override"] = overrides.get(row["rel"])
    return state["rows"]


def render(state: dict) -> str:
    args = state["args"]
    rows = refresh(state)
    unused = faces.unused_overrides(faces.load_overrides(state["overrides_file"]), rows)
    if unused:
        print(f"   ! {len(unused)} override entr(ies) match no photo: {', '.join(unused[:3])}", flush=True)
    return faces.render_report(
        state["db"],
        args.person,
        rows,
        state["overrides_file"],
        args.threshold,
        args.min_face,
        interactive=True,
        subtitle="click keep / drop",
    )


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            page = render(state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            if self.path != "/decide":
                self.send_error(404)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                faces.apply_decision(state["overrides_file"], payload["key"], payload["action"])
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))
                return
            print(f"   {payload['action']:5s} {payload['key']}", flush=True)
            self.send_response(204)
            self.end_headers()

        def log_message(self, fmt, *args) -> None:
            pass  # the decide lines above are enough

    return Handler


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = build_state(args)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"serving on http://127.0.0.1:{args.port}/", flush=True)
    print(f"decisions -> {state['overrides_file']}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
