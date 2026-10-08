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
- Optional SMB-only mode: when `SMB_*` vars are set in `.env`, albums are
  uploaded to the NAS share via a temp staging dir (no local copy); otherwise
  they stay under `downloads/`.
- Skip albums whose target folder already contains files (local or SMB) so
  re-runs are cheap and resume cleanly.
- Optional `APPRISE_URL` success notification, sent only when at least one album
  was actually downloaded.