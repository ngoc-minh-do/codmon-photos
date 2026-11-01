# codmon-photos — AGENTS.md

## Project

Download photos from Codmon at the best resolution the CDN allows into
`./downloads/<display_date>_<album_title>` (e.g. `downloads/2026-10-05_sports-day`;
a `_<album_id>` suffix is added only when two albums share the same date+title).
The date comes from each photo album's `insert_datetime` / `display_date`
(Tokyo time). Albums always land in `./downloads/` and are never uploaded.
If `SMB_HOST`/`SMB_SHARE`/`SMB_USER`/`SMB_PASSWORD` are set in `.env`, only the
face filter's output (`matched/<name>/`) is mirrored to the share as
`//<host>/<share>/<name>/<album>/`. Recurring, roughly monthly task.

The script logs directly into the Codmon parent API (`ps-api.codmon.com`,
email/password session) — no browser, no pasted JSON. The same internal API
conventions as the sibling `codmon-sync` project.

## How to run

```bash
cp .env.example .env   # fill CODMON_EMAIL / CODMON_PASSWORD (same creds as codmon-sync)
uv sync                # first run: creates .venv, installs deps
uv run scripts/codmon_sync.py               # scan last 45 days, download every album found
uv run scripts/codmon_sync.py --date 2026-10-05   # one specific day
```

Options: `--date`, `--lookback` (default 45 days), `--out`, `--workers`, `--tz`,
`--dry-run` (list albums/photos that would be downloaded; no fetch or writes).

The script is a 3-phase pipeline — never reimplement or re-test per photo:

1. **Classify** — one tiny `width=10` probe fetch per photo (~300 B JPEG, aspect
   preserved) to learn landscape vs portrait.
2. **Calibrate** — pick 1 landscape + 1 portrait sample; fetch both combos and
   keep the combo with the larger dims for each orientation. If an album is
   single-orientation (e.g. all landscape), calibrate what exists and assume the
   default combo for the missing orientation.
3. **Download** — fetch every photo exactly once at the calibrated combo into
   `./downloads/<display_date>_<album_title>`, then embed the album's
   `insert_datetime` as EXIF
   `DateTimeOriginal` + `OffsetTimeOriginal` (Tokyo +09:00) into each JPEG —
   the CDN strips all EXIF, so without this step importers (Immich, Windows)
   would date the photos at download time.

Combo rule: constrain the axis that is SMALLER in the photo's aspect so the auto
side exceeds the 500 cap (landscape -> `&width=0&height=500` ~667x500/750x500/
1110x500; portrait -> `&width=500&height=0` ~500x667/500x890). Per-axis param cap
is 500; omitting size params -> 403.

## After the run

- Report count, dims seen, and the output folder (`./downloads/<date>_<title>`).

## Face filter (optional)

Finds photos containing a specific child by face — the API exposes no per-child
tags (each photo row is only `{id, url}`).

```bash
# 1. put 4-6 face photos in refs/<name>/face/ (not cropped to the face)
uv run scripts/face_filter.py enroll <name>
uv run scripts/face_filter.py scan                    # detect + embed every photo (cached)
uv run scripts/face_filter.py report <name>            # sweep + .cache/report-<name>.html
uv run scripts/face_filter.py export <name> --threshold 0.55 --yes
uv run scripts/face_filter.py upload <name> --yes      # optional: mirror to the NAS
uv run scripts/face_filter.py compare <name> --threshold 0.55   # scrfd vs yunet
```

Rules:

- `refs/<name>/face/*.jpg` is the only human-provided input; `.cache/` holds all
  derived state (sqlite embeddings + ONNX models). Both are gitignored.
- **Album folders are never modified.** `export` copies matches to
  `matched/<name>/<album>/` (or `--matched <dir>`); `--link` hardlinks instead
  and `--prune` deletes earlier copies that no longer match. Rerunning with
  another threshold is idempotent: existing files are kept, missing ones copied.
- Enrollment takes the largest face in each reference and prints a warning for a
  multi-face photo, a face under 64px, or a reference that is <0.35 similar to
  the others (probably not her).
- Pipeline: detector (5 landmarks) -> ArcFace `w600k_r50` (align to the 112x112
  template, 512-d embedding) -> cosine against the reference gallery. Detection
  and recognition are separate ONNX stages; the embedding model is the
  recognition half of InsightFace's buffalo_l.
- Two detectors: `scrfd` (default, `det_10g` from buffalo_l) and `yunet`. Cache
  keys carry the full pipeline id (`v2:<detector>:w600k_r50.onnx:arc112:floor20`),
  so each detector keeps its own rows for the same photo and a changed component
  never reuses stale vectors. Bump `PIPELINE_VERSION` when stored vectors would
  change for identical inputs.
- SCRFD is the default: its landmarks hold up on tilted/turned faces, where
  YuNet's misplace them and the 112x112 crop (hence the embedding) is ruined.
  YuNet detects slightly more ~30px faces but loses photos to bad crops, so it
  stays available for comparison rather than as the default. `compare` prints
  the per-photo cross-checks that show this.
- Cache rows are keyed by file sha1, not path, so the same photo is embedded once
  wherever it appears (album, staging dir, re-download). Every face down to 20px
  is cached, so `--threshold` and `--min-face` are pure re-scoring: never re-run
  `scan` to try a different cutoff. Default `--min-face` is 40px — narrower faces
  are reported as "unusable", not as "not her".
- Score sweep (median face is ~31px, max ~1110x500 per album): expect the cutoff
  to sit around 0.40-0.55. Close-ups match strongly (>0.6); wide group shots are
  the failure mode.
- `scan` and `collect` walk the `--photos` roots (default `downloads` only).
- Body matching is deliberately absent: `reid_0265` is short-term ReID, so it
  only fires within one album's outfit, and on these 500px photos it returned no
  matches at its 0.85 threshold.
- `upload` mirrors `matched/<name>/` to `//<host>/<share>/<name>/<album>/`,
  skipping files already there with the same name and size; `--delete` removes
  remote files that no longer match. Nothing else is ever uploaded, and old
  album folders already on the share are left alone.
- `APPRISE_URL` notifies after an **upload** that moved photos; downloads are
  silent.
- `evaluate <person>` prints, per candidate threshold, how many of your `keep`
  decisions the score would have found and how many `drop` decisions it would
  have admitted — the check to run after a review session, before moving the
  default. Unreviewed photos are not part of that ground truth.
- Hand decisions live in `refs/<name>/overrides.json` (`album/file.JPG` ->
  `keep`/`drop`, written by `review_server.py` or by hand). They win over the
  score in both directions: `keep` exports a photo that scored low or had no
  detectable face, `drop` removes one that scored well. Entries that match no
  photo are reported as dead lines.

## Gotchas

- Signed URLs expire in ~40 minutes. A 403 during a run means re-run the script
  (it re-fetches fresh URLs from the API); do not retry or tweak old URLs.
- The timeline mixes real photo albums with `kind=8` non-photo forms (attendance
  reports, absence notices) and the albums endpoint 404s on those ids. Discovery
  filters on `photos`/`photo_count` and treats 404 as "not an album"
  (`CodmonNotFound`); any other status still aborts.
- SMB history: the user rejected always-on NAS upload, briefly used an optional
  staging-based SMB-only mode (albums to the share, no local copy), then dropped
  that: albums now always stay local and only `matched/<name>/` is uploaded.
- The user explicitly rejected per-photo best-resolution fetching. Calibration
  only.

## Layout

- `scripts/codmon_sync.py` — the pipeline (login -> download to dated folder; no uploads)
- `scripts/codmon_api.py` — Codmon parent-API client (login, children, timeline)
- `scripts/faces.py` — detect/align/embed, content-addressed sqlite cache, scoring, report
- `scripts/face_filter.py` — CLI over `faces.py` (enroll / scan / report / export / upload)
- `scripts/smb_sync.py` — upload planning + mirroring to the SMB share
- `scripts/notify.py` — Apprise webhook POST
- `refs/<name>/face/*.jpg` — reference photos of a child (gitignored)
- `matched/<name>/<album>/` — copies of the photos that matched (gitignored)
- `.cache/` — sqlite embedding cache, ONNX models, review reports (gitignored)
- `pyproject.toml` / `uv.lock` — uv-managed deps
- `.env.example` — credentials template; copy to `.env`
- `downloads/<date>_<title>/` — where albums land (always local)