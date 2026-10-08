# codmon-photos — AGENTS.md

## Project

Download photos from Codmon at the best resolution the CDN allows into
`./downloads/<date>` (e.g. `downloads/2026-09-14`). The date comes from each
photo album's `insert_datetime` / `display_date` (Tokyo time). No upload step —
photos stay local. Recurring, roughly monthly task.

The script logs directly into the Codmon parent API (`ps-api.codmon.com`,
email/password session) — no browser, no pasted JSON. The same internal API
conventions as the sibling `codmon-huckleberry-sync` project.

## How to run

```bash
cp .env.example .env   # fill CODMON_EMAIL / CODMON_PASSWORD (same creds as codmon-huckleberry-sync)
uv sync                # first run: creates .venv, installs deps
uv run scripts/codmon_sync.py               # scan last 45 days, download every album found
uv run scripts/codmon_sync.py --date 2026-10-05   # one specific day
```

Options: `--date`, `--lookback` (default 45 days), `--out`, `--workers`, `--tz`.

The script is a 3-phase pipeline — never reimplement or re-test per photo:

1. **Classify** — one tiny `width=10` probe fetch per photo (~300 B JPEG, aspect
   preserved) to learn landscape vs portrait.
2. **Calibrate** — pick 1 landscape + 1 portrait sample; fetch both combos and
   keep the combo with the larger dims for each orientation. If an album is
   single-orientation (e.g. all landscape), calibrate what exists and assume the
   default combo for the missing orientation.
3. **Download** — fetch every photo exactly once at the calibrated combo into
   `./downloads/<date>`, then embed the album's `insert_datetime` as EXIF
   `DateTimeOriginal` + `OffsetTimeOriginal` (Tokyo +09:00) into each JPEG —
   the CDN strips all EXIF, so without this step importers (Immich, Windows)
   would date the photos at download time.

Combo rule: constrain the axis that is SMALLER in the photo's aspect so the auto
side exceeds the 500 cap (landscape -> `&width=0&height=500` ~667x500/750x500/
1110x500; portrait -> `&width=500&height=0` ~500x667/500x890). Per-axis param cap
is 500; omitting size params -> 403.

## After the run

- Report count, dims seen, and the output folder (`./downloads/<date>`).

## Gotchas

- Signed URLs expire in ~40 minutes. A 403 during a run means re-run the script
  (it re-fetches fresh URLs from the API); do not retry or tweak old URLs.
- The user explicitly rejected per-photo best-resolution fetching. Calibration
  only.
- The user rejected uploading to the NAS SMB share (`\\192.168.0.100\share`);
  downloads stay local under a dated folder.

## Layout

- `scripts/codmon_sync.py` — the pipeline (login -> download to dated folder)
- `scripts/codmon_api.py` — Codmon parent-API client (login, children, timeline)
- `pyproject.toml` / `uv.lock` — uv-managed deps
- `.env.example` — credentials template; copy to `.env`
- `.opencode/skills/codmon-photos-to-nas/SKILL.md` — workflow skill
- `downloads/<date>/` — where photos land