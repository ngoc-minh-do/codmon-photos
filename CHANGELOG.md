# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Download Codmon photo albums directly from the internal parent API
  (`ps-api.codmon.com`) — no browser and no saved `response.json` needed.
- Automatic album discovery: scan the timeline for photo albums and pull each
  album's full photo list, downloading into `downloads/<display_date>_<title>`.
- 3-phase resolution pipeline (classify → calibrate → download) that finds the
  best CDN size combo once per orientation instead of per photo.
- EXIF stamping: `DateTimeOriginal` + `OffsetTimeOriginal` (`+09:00` Tokyo) are
  backfilled from the album's `insert_datetime` because the CDN strips all EXIF.
- Optional SMB mirror for the face filter's output: `face_filter.py upload`
  copies `matched/<name>/` to `//<host>/<share>/<name>/<album>/`, skipping files
  already present and deleting nothing unless `--delete` is passed.
- Skip albums whose folder already contains files so re-runs are cheap and
  resume cleanly.
- Optional `APPRISE_URL` success notification, sent only after an upload that
  actually put photos on the share.
- Optional face filter (`scripts/face_filter.py`): enroll a child from
  `refs/<name>/face/*.jpg`, score every downloaded photo by ArcFace similarity,
  review a threshold sweep in a self-contained HTML sheet, then copy matches to
  `matched/<name>/<album>/` (album folders are never modified). Detection uses
  YuNet and embeddings are the ArcFace `w600k_r50` head, both plain ONNX;
  embeddings are cached by file content in `.cache/faces.db`, so changing the
  threshold never re-runs a model and never needs a re-scan.
- SCRFD (`det_10g`) is the default detector, with `--detector yunet` as the
  alternative; a second face detector selectable via `--detector`, plus a
  `compare` command that reports agreement, per-detector-only finds and score
  deltas. Cache rows are keyed by the full pipeline id, so detectors can be
  scanned side by side without one overwriting the other.
- `review_server.py`: a localhost page with keep / drop buttons per photo, so
  photos the model misses or misjudges are corrected by hand instead of by
  changing the model or the threshold. Decisions live in
  `refs/<name>/overrides.json` (`album/file.JPG` -> `keep`/`drop`) and override
  the score in `report`, `export` and `prune`.
- Album discovery skips `kind=8` timeline entries that carry no photos (attendance
  forms, absence notices) and tolerates a 404 from the albums endpoint, so one
  unrelated item can no longer abort a whole run.
