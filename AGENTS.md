# codmon-photos — AGENTS.md

## Project

Download photos from Codmon at the best resolution the CDN allows into
`./downloads/<display_date>_<album_title>` (e.g.
`downloads/2026-10-05_sports-day`; a `_<album_id>` suffix is added
only when two albums share the same date+title). The date comes from each
album's `insert_datetime` / `display_date` (Tokyo time). Albums always land in
`./downloads/` and are never uploaded.

If `SMB_HOST` / `SMB_SHARE` / `SMB_USER` / `SMB_PASSWORD` are all set in `.env`,
only the face filter's output (`matched/<name>/`) is mirrored to the share as
`//<host>/<share>/<name>/<album>/`. Setting some but not all of them aborts.

The scripts log directly into the Codmon parent API (`ps-api.codmon.com`,
email/password session) — no browser, no pasted JSON. Recurring, roughly
monthly task.

## Commands

```bash
cp .env.example .env   # fill CODMON_EMAIL / CODMON_PASSWORD
uv sync                # first run: creates .venv, installs deps
uv run scripts/codmon_sync.py                      # scan last 45 days, download every album found
uv run scripts/codmon_sync.py --date 2026-10-05    # one specific day
uv run scripts/codmon_sync.py --dry-run            # list albums/photos; no fetch or writes
make check             # lint + format-check + compile + tests (the CI gate)
```

Options: `--date`, `--lookback` (default 45 days), `--out` (default `downloads`),
`--workers` (default 4), `--tz` (EXIF offset, default `+09:00`), `--dry-run`.
Toolchain: uv, Python ≥ 3.14, ruff (line-length 120; `E`, `F`, `I`, `UP`, `B`),
pytest over `tests/`.

## Download pipeline

Three phases per run — never reimplement or re-test per photo:

1. **Classify** — one `width=10` probe fetch per photo (~300 B JPEG, aspect
   preserved) to learn landscape vs portrait.
2. **Calibrate** — pick 1 landscape + 1 portrait sample, fetch both combos, and
   keep the combo with the larger dims for that orientation. A
   single-orientation album calibrates what exists and assumes the default combo
   for the missing orientation.
3. **Download** — fetch every photo exactly once at the calibrated combo, then
   embed the album's `insert_datetime` as EXIF `DateTimeOriginal` +
   `OffsetTimeOriginal` (Tokyo `+09:00`). The CDN strips all EXIF, so without
   this importers date the photos at download time.

Combo rule: constrain the axis that is SMALLER in the photo's aspect so the auto
side exceeds the 500 cap (landscape → `&width=0&height=500` ~667x500/750x500/
1110x500; portrait → `&width=500&height=0` ~500x667/500x890). The per-axis
param cap is 500; omitting size params → 403.

Resolution is calibrated, never searched per photo. Albums whose target folder
already has files are skipped.

## After a run

Report count, dims seen, and the output folder (`./downloads/<date>_<title>`).

## Face filter

Finds photos containing a specific child by face — the API exposes no per-child
tags (each photo row is only `{id, url}`).

```bash
# 1. put 4-6 face photos in refs/<name>/face/ (not cropped to the face)
uv run scripts/face_filter.py enroll <name>
uv run scripts/face_filter.py scan                     # detect + embed every photo (cached)
uv run scripts/face_filter.py export <name> --threshold 0.55 --yes
uv run scripts/face_filter.py upload <name> --yes      # optional: mirror to the NAS
uv run scripts/face_filter.py compare <name> --threshold 0.55   # scrfd vs yunet
uv run scripts/review_server.py <name> --threshold 0.55         # keep/drop page
```

Rules:

- `refs/<name>/face/*.jpg` is the only human-provided input; `.cache/` holds all
  derived state (sqlite embeddings, ONNX models, aligned crops). Both are
  gitignored.
- **Album folders are never modified.** `export` copies matches to
  `matched/<name>/<album>/` (or `--matched <dir>`); `--link` hardlinks instead,
  `--prune` deletes earlier copies that no longer match. Re-running at another
  threshold is idempotent: existing files are kept, missing ones copied.
- Enrollment takes the largest face in each reference and warns on a multi-face
  photo, a face under 64px, or a reference <0.35 similar to the others.
- Pipeline: detector (5 landmarks) → ArcFace `w600k_r50` (align to the 112x112
  template, 512-d embedding) → cosine against the reference gallery. Detection
  and recognition are separate ONNX stages; the recognizer is the recognition
  half of InsightFace's buffalo_l.
- Detectors: `scrfd` (default, `det_10g` from buffalo_l) and `yunet`. Cache keys
  carry the full pipeline id (`v2:<detector>:w600k_r50.onnx:arc112:floor20`), so
  each detector keeps its own rows for the same photo. Bump `PIPELINE_VERSION`
  when stored vectors would change for identical inputs.
- SCRFD is the default: its landmarks hold on tilted/turned faces, where YuNet's
  drift ruins the 112x112 crop and therefore the embedding. YuNet detects
  slightly more ~30px faces but loses photos to bad crops; keep it for
  `compare` rather than as the default.
- Cache rows are keyed by file sha1, not path, so a photo is embedded once
  wherever it appears (album, staging dir, re-download). Every face down to 20px
  is stored, so `--threshold` and `--min-face` are pure re-scoring: never re-run
  `scan` to try a different cutoff. Default `--min-face` is 20px, so every
  stored face is judged; narrower ones are reported as "unusable", not as
  "not her".
- Score sweep (median face ~31px, sources up to 1110x500): expect the cutoff
  around 0.40-0.55. Close-ups match strongly (>0.6); wide group shots are the
  failure mode.
- `scan` and `collect` walk the `--photos` roots (default `downloads` only), so
  `--out` and `--photos` must agree.
- Body matching is deliberately absent: `reid_0265` is short-term ReID, fires
  only within one album's outfit, and returned no matches at 0.85 on these
  500px photos.
- `upload` mirrors `matched/<name>/` to `//<host>/<share>/<name>/<album>/`,
  skipping files already there with the same name and size; `--delete` removes
  remote files that no longer match. Nothing else is uploaded, and album folders
  already on the share are left alone.
- `APPRISE_URL` notifies after an **upload** that moved photos; downloads are
  silent.
- `evaluate <person> [--baseline X]` scores each candidate threshold against the
  decisions: `found` / `lost` over the photos counted as hers (pinned plus
  undecided matches at `--baseline`, default = the current default threshold),
  `exported` for what a run would copy, and `blocked out` / `blocked in` for the
  drops. The baseline is an assumption, so it is printed with the table.
- Hand decisions live in `refs/<name>/overrides.json` (`album/file.JPG` →
  `keep`/`drop`, written by `review_server.py` or by hand). They win over the
  score in both directions: `keep` exports a photo that scored low or had no
  detectable face, `drop` removes one that scored well. Entries that match no
  photo are reported as dead lines.

## Gotchas

- Signed URLs expire in ~40 minutes. A 403 during a run means re-run the script
  (it re-fetches fresh URLs from the API); never retry or tweak old URLs.
- The timeline mixes real photo albums with `kind=8` non-photo forms (attendance
  reports, absence notices) and the albums endpoint 404s on those ids. Discovery
  filters on `photos`/`photo_count` and treats 404 as "not an album"
  (`CodmonNotFound`); any other status still aborts.
- `TZ` (default `Asia/Tokyo`) resolves "today" for the scan window; `--tz` sets
  the EXIF offset and is a separate value.

## Layout

- `scripts/` — all implementation, grouped by role: album sync, Codmon API client, face pipeline, review page, SMB mirror, webhook. Each module's docstring states its role; tests put `scripts/` on `sys.path` and import the modules directly.
- `tests/` — pytest suite; run it via `make check`.
- `downloads/`, `refs/`, `matched/`, `.cache/` — local data and derived state, all gitignored (albums, reference photos, filtered copies, embeddings/models).
- Root config: `pyproject.toml` / `uv.lock`, `Makefile` (gates), `.env.example` (env template).
