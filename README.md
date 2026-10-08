# codmon-photos

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
5. **Mirror (optional)** — if `SMB_*` vars are set in `.env`, each album is
   uploaded to a NAS SMB share via a temp staging dir and **no local copy is
   kept**; when they are unset, albums stay in `./downloads/`.

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
uv run scripts/codmon_sync.py --lookback 220       # scan ~since March
```

Options: `--date`, `--lookback` (default 45 days), `--out`, `--workers`,
`--tz`. Run `uv run scripts/codmon_sync.py -h`.

## Layout

- `scripts/codmon_sync.py` — the pipeline (login → download to dated folder; SMB-only mode when configured)
- `scripts/codmon_api.py` — Codmon parent-API client (login, children, timeline, albums)
- `pyproject.toml` / `uv.lock` — uv-managed deps
- `.env.example` — credentials template; copy to `.env`
- `downloads/<display_date>_<album_title>/` — where photos land

## Disclaimer / risk

- Uses your own account credentials to read Codmon; the API is reverse-engineered
  and unofficial. Use at your own risk.
- Signed photo URLs expire in ~40 minutes; a 403 mid-run just means re-run the
  script (it re-fetches fresh URLs).
- Credentials live only in `.env`, which is git-ignored.