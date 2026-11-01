# codmon-photos

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Download your child's nursery photo albums from
[Codmon](https://parents.codmon.com/) at the best resolution its CDN allows.

Codmon exposes no public parent API, but the parent web app is an SPA that
talks to an internal JSON API (`ps-api.codmon.com`) using session cookies from a
plain email/password `POST /login`. This project uses that API directly with the
Python standard library — **no browser needed**.

The integration is unofficial and can break if Codmon changes its API.

## What it does

1. **Discover** — log in, resolve the nursery `service_id`, scan the timeline
   for photo albums and pull each album's full photo list from the albums
   detail endpoint.
2. **Calibrate** (the resolution is never searched per-photo): one tiny
   `width=10` probe per photo learns its orientation; one landscape + one
   portrait sample decide which of the two 500-cap size combos yields the
   largest output for each orientation.
3. **Download** — fetch every photo exactly once at the calibrated combo into
   `./downloads/<display_date>_<album_title>`.
4. **Stamp** — the CDN strips all EXIF, so each JPEG gets `DateTimeOriginal`
   and `OffsetTimeOriginal` (default `+09:00` Tokyo) backfilled from the
   album's `insert_datetime`; without this, importers (Immich, Windows) date
   the photos at download time.
5. **Export + mirror (optional)** — albums always stay in `./downloads/`. The
   face filter can copy the photos a given child appears in to
   `matched/<name>/`, and `face_filter.py upload` mirrors that folder to a NAS
   SMB share (`//<host>/<share>/<name>/<album>/`). Albums are never uploaded.

## Setup

```bash
cp .env.example .env   # fill CODMON_EMAIL / CODMON_PASSWORD (and optionally SMB_*)
uv sync                # creates .venv, installs deps
```

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.14.

## Run

```bash
uv run scripts/codmon_sync.py                      # scan last 45 days, download every album found
uv run scripts/codmon_sync.py --date 2026-10-05    # one specific day
uv run scripts/codmon_sync.py --lookback 90        # a longer window
```

Options: `--date`, `--lookback` (default 45 days), `--out`, `--workers`,
`--tz`. Set `APPRISE_URL` in `.env` to get a success notification after an
upload that actually put photos on the share (all-skip runs stay silent). Run
`uv run scripts/codmon_sync.py -h`.

## Filter by face (optional)

Keep only the photos your child is in. The Codmon API carries no per-child tags,
so this runs locally: face detection (YuNet) + ArcFace embeddings compared with
photos you enroll once.

```bash
# put 4-6 face photos of your child in refs/<name>/face/
uv run scripts/face_filter.py enroll <name>          # build the reference gallery
uv run scripts/face_filter.py scan                   # embed every photo (cached)
uv run scripts/face_filter.py report <name>          # threshold sweep + review sheet
uv run scripts/face_filter.py export <name> --threshold 0.50 --yes
uv run scripts/face_filter.py upload <name> --yes   # optional: mirror to the NAS
```

Two detectors are available: `--detector scrfd` (default, buffalo_l's `det_10g`)
and `--detector yunet` (finds more of the tiny faces a 500px CDN photo leaves
behind, but its landmarks drift on tilted faces). Scans are cached per detector,
so both can be run and compared:

```bash
uv run scripts/face_filter.py scan --detector scrfd
uv run scripts/face_filter.py scan --detector yunet
uv run scripts/face_filter.py compare <name> --threshold 0.45
```

`compare` reports how many photos both detectors keep, how many only one does,
and the score delta on the photos both catch.

`report` writes a self-contained `.cache/report-<name>.html` with every photo,
its best similarity score and the matched face crop, so the cutoff can be picked
by looking rather than guessing. `export` **copies** matching photos to
`matched/<name>/<album>/` and leaves `downloads/` exactly as downloaded; `--link`
hardlinks instead of copying, `--prune` drops earlier copies that no longer
match. Scores are cached in `.cache/faces.db` keyed by file content, so trying
another threshold never re-runs a model and never needs a re-scan.

`upload` needs all four `SMB_*` vars in `.env`; it uploads only files missing on
the share (same name and size are skipped) and never deletes anything unless
`--delete` is passed. `APPRISE_URL` fires after an upload that actually moved
photos, not after a download.

`refs/`, `matched/` and `.cache/` are git-ignored: reference photos and
embeddings stay on your machine, and the ONNX models are downloaded once into
`.cache/models`.

## Layout

- `scripts/codmon_sync.py` — the pipeline (login → download to dated folder; SMB-only mode when configured)
- `scripts/codmon_api.py` — Codmon parent-API client (login, children, timeline, albums)
- `scripts/faces.py` — face detect/align/embed, embedding cache, scoring, review sheet
- `scripts/smb_sync.py` — upload planning/mirroring for the SMB share
- `scripts/notify.py` — Apprise webhook POST
- `scripts/face_filter.py` — CLI: `enroll` / `scan` / `report` / `export` / `upload`
- `refs/<name>/face/*.jpg` — reference photos of a child (git-ignored)
- `matched/<name>/<album>/` — copies of the photos that matched (git-ignored)
- `.cache/` — content-addressed embedding cache, ONNX models, review sheets (git-ignored)
- `pyproject.toml` / `uv.lock` — uv-managed deps
- `.env.example` — credentials template; copy to `.env`
- `downloads/<display_date>_<album_title>/` — where photos land

## Disclaimer / risk

- Uses your own account credentials to read Codmon; the API is reverse-engineered
  and unofficial. Use at your own risk.
- Signed photo URLs expire in ~40 minutes; a 403 mid-run just means re-run the
  script (it re-fetches fresh URLs).
- Credentials live only in `.env`, which is git-ignored.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately per
[SECURITY.md](SECURITY.md) — never as a public issue.

## License

[MIT](LICENSE) — © 2026 Ngoc Do