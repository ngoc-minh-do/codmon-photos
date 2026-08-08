# codmon-photos

Download photos from a Codmon API response JSON at the best resolution the CDN allows,
then upload them to the home NAS (`\\192.168.0.100\share\<date>`, anonymous SMB).

## Usage

```bash
uv run scripts/codmon_sync.py <response.json>
```

- Downloads to `./downloads` then uploads to `//192.168.0.100/share/YYYY-MM-DD`.
- `--no-upload` to skip the SMB copy; `--date`, `--share`, `--host`, `--out`,
  `--workers` to override defaults (see `uv run scripts/codmon_sync.py -h`).

## Setup

```bash
uv sync          # installs pysmb + deps into .venv (tested; uv 0.12.x)
```

## Layout

- `scripts/codmon_sync.py` — the whole pipeline (download + SMB upload)
- `pyproject.toml` / `uv.lock` — uv-managed deps (pysmb, pyasn1, tqdm); no
  smbclient/mount.cifs required
- `AGENTS.md` — project instructions for the agent (pipeline, gotchas)
- `.opencode/skills/codmon-photos-to-nas/SKILL.md` — the opencode skill that drives
  this workflow when asked again
- `response.json` — the current photo API response (paste new ones here)

## Notes

- Signed photo URLs expire within ~40 minutes; a 403 means the response.json must be
  regenerated, not tweaked.
- Max photo size per axis is 500 for a requested param; the script auto-selects
  `width=0&height=500` (landscape) vs `width=500&height=0` (portrait) for the largest
  output.