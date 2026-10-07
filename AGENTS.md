# codmon-photos — AGENTS.md

## Project

Download photos from a Codmon API response JSON at the best resolution the CDN
allows into `./downloads/<date>` (e.g. `downloads/2026-09-14`). The date comes
from the response's `insert_datetime` (falls back to today). No upload step —
photos stay local. Recurring, roughly monthly task.

## How to run

```bash
uv run scripts/codmon_sync.py <response.json>   # default: ./response.json
```

First run: `uv sync` to create `.venv` and install deps.

Options: `--date`, `--out`, `--workers`.

The script is a 3-phase pipeline — never reimplement or re-test per photo:

1. **Classify** — one tiny `width=10` probe fetch per photo (~300 B JPEG, aspect
   preserved) to learn landscape vs portrait.
2. **Calibrate** — pick 1 landscape + 1 portrait sample; fetch both combos and
   keep the combo with the larger dims for each orientation. If an album is
   single-orientation (e.g. all landscape), calibrate what exists and assume the
   default combo for the missing orientation.
3. **Download** — fetch every photo exactly once at the calibrated combo into
   `./downloads/<date>`.

Combo rule: constrain the axis that is SMALLER in the photo's aspect so the auto
side exceeds the 500 cap (landscape -> `&width=0&height=500` ~667x500/750x500/
1110x500; portrait -> `&width=500&height=0` ~500x667/500x890). Per-axis param cap
is 500; omitting size params -> 403.

## After the run

- Report count, dims seen, and the output folder (`./downloads/<date>`).

## Gotchas

- Signed URLs expire in ~40 minutes. A 403 means the user must paste a NEW
  response; do not retry or tweak the old URLs.
- The user explicitly rejected per-photo best-resolution fetching. Calibration
  only.
- The user rejected uploading to the NAS SMB share (`\\192.168.0.100\share`);
  downloads stay local under a dated folder.
- `response.json` in the repo root holds the current API response; new ones get
  pasted/moved here.

## Layout

- `scripts/codmon_sync.py` — the pipeline (download to dated folder)
- `pyproject.toml` / `uv.lock` — uv-managed deps
- `.opencode/skills/codmon-photos-to-nas/SKILL.md` — workflow skill
- `response.json` — current API response
- `downloads/<date>/` — where photos land