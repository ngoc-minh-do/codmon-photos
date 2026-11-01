# codmon-photos

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Download your child's nursery photo albums from
[Codmon](https://parents.codmon.com/) at the best resolution its CDN allows.

Codmon has no public parent API. The parent web app is an SPA over an internal
JSON API (`ps-api.codmon.com`) with an email/password login, and these scripts
call that API directly — no browser, no pasted JSON. The integration is
unofficial, so a Codmon API change can break it.

## Features

- **Album discovery** over the last 45 days, one specific day, or a longer
  window of your choosing.
- **Calibrated resolution** — one probe per photo to learn its orientation, then
  one size combo per orientation chosen from a sample, instead of guessing or
  fetching every photo twice.
- **Restored EXIF dates** — the CDN strips EXIF, so each download gets the
  album's timestamp back as `DateTimeOriginal` / `OffsetTimeOriginal`; importers
  (Immich, Windows, …) then file photos by shooting date.
- **Face filter (optional)** — keeps the photos a chosen child appears in, with
  a local review page for the borderline calls.
- **SMB mirror (optional)** — pushes that filtered folder to a NAS share.

## How it works

1. **Log in and discover** — resolve the nursery, scan the timeline for photo
   albums, and pull each album's photo list.
2. **Classify and calibrate** — a tiny `width=10` probe per photo tells landscape
   from portrait; one sample per orientation decides which of the two
   `500`-capped size combos yields the larger image.
3. **Download and stamp** — every photo is fetched exactly once at the
   calibrated size into
   `./downloads/<display_date>_<album_title>/` (a `_<album_id>` suffix is added
   when two albums share a date and title), then stamped with the album's EXIF
   date in Tokyo time.

Albums whose folder already contains files are skipped, so re-runs resume
cheaply.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.14.

```bash
cp .env.example .env   # fill CODMON_EMAIL / CODMON_PASSWORD (and optionally SMB_* / APPRISE_URL)
uv sync
```

`TZ` (default `Asia/Tokyo`) sets the timezone used to resolve "today" for the
scan window.

## Usage

```bash
uv run scripts/codmon_sync.py                      # last 45 days, every album found
uv run scripts/codmon_sync.py --date 2026-10-05    # one specific day
uv run scripts/codmon_sync.py --lookback 90        # a longer window
uv run scripts/codmon_sync.py --dry-run            # list albums and photos, write nothing
```

Options: `--out` (default `downloads`), `--date`, `--lookback` (default 45
days), `--workers` (default 4), `--tz` (EXIF offset, default `+09:00`),
`--dry-run`. Run with `-h` for the full list.

## Face filter (optional)

The API carries no per-child tags, so this runs locally: face detection plus
ArcFace embeddings compared against reference photos you enroll once.

```bash
# put 4-6 face photos of your child in refs/<name>/face/
uv run scripts/face_filter.py enroll <name>          # build the reference gallery
uv run scripts/face_filter.py scan                   # detect + embed every photo (cached)
uv run scripts/face_filter.py export <name> --threshold 0.55 --yes
uv run scripts/face_filter.py upload <name> --yes    # optional: mirror to the NAS
```

`enroll` reports its aligned crops under `.cache/refs/<name>/<detector>/` so you
can confirm they are all the same child. `scan` caches embeddings by file
content, so changing `--threshold` or `--min-face` only re-scores.

### Picking a threshold

`export` is a dry run without `--yes` and prints each match with its score. The
cutoff sits around 0.40–0.55 on these ~500px photos: close-ups score above 0.6,
while wide group shots are where it fails.

For a visual pass, serve the review sheet and click keep / drop per photo:

```bash
uv run scripts/review_server.py <name> --threshold 0.55   # http://127.0.0.1:8765
```

Decisions are stored in `refs/<name>/overrides.json` (keyed by
`album/file.JPG`) and override the score in both directions: `keep` exports a
photo that scored low or had no detectable face, `drop` removes a false
positive. Check a cutoff against those decisions with:

```bash
uv run scripts/face_filter.py evaluate <name> [--baseline 0.55]
```

`found` / `lost` count photos treated as hers (your `keep` pins plus undecided
matches at the baseline); `blocked out` / `blocked in` cover your `drop`s.

### Detectors

`--detector scrfd` (default) is InsightFace's `det_10g`; `--detector yunet` is
the OpenCV alternative. Scans are cached per detector, so both can be run and
compared:

```bash
uv run scripts/face_filter.py scan --detector scrfd
uv run scripts/face_filter.py scan --detector yunet
uv run scripts/face_filter.py compare <name> --threshold 0.55
```

`compare` reports how many photos both detectors keep, how many only one does,
and the score delta on the photos both catch. SCRFD stays the default because
its landmarks hold on tilted faces; YuNet's drift there ruins the aligned crop.

### Export and upload

`export` **copies** matches to `matched/<name>/<album>/` and leaves `downloads/`
untouched; `--link` hardlinks instead, `--prune` drops earlier copies that no
longer match.

`upload` needs all four `SMB_*` vars and mirrors only `matched/<name>/` to
`//<host>/<share>/<name>/<album>/`, skipping files already present with the same
name and size. Nothing is deleted without `--delete`, and album folders are
never uploaded. `APPRISE_URL` notifies after an upload that moved photos;
downloads stay silent.

## Known limitations

- Unofficial API: reverse-engineered and subject to change.
- Signed photo URLs expire in ~40 minutes. A 403 mid-run means re-run the
  script, which fetches fresh URLs.
- Detection runs on CDN-sized photos where a face is typically ~30px wide.
  Nothing below 20px is stored, and wide group shots are where matching fails.
- Body re-identification is deliberately absent — the available ReID model only
  works short-term, within a single outfit, and found nothing on these photos.

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | all implementation: album sync, Codmon API client, face pipeline, review page, SMB mirror, webhook |
| `tests/` | pytest suite (imports modules from `scripts/`) |
| `downloads/` | downloaded albums, one folder per album (git-ignored) |
| `refs/` | reference photos per child — the only manual input (git-ignored) |
| `matched/` | filtered copies per child/album (git-ignored) |
| `.cache/` | embedding cache, ONNX models, derived crops and sheets (git-ignored) |
| root | `pyproject.toml` + `uv.lock` (deps), `Makefile` (gates), `.env.example` (config template) |

## Development

```bash
make check   # lint + format-check + compile + tests, the same gate CI runs
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately per
[SECURITY.md](SECURITY.md) — never as a public issue.

## License

[MIT](LICENSE) — © 2026 Ngoc Do
