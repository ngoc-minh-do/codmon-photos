"""Upload the photos that matched a person to an SMB share.

The album folders stay local; only ``matched/<person>/<album>/`` is mirrored to
``//<host>/<share>/<person>/<album>/<file>``. Files already on the share with the
same name and size are skipped, so re-running after a threshold change uploads
just the diff. Nothing is ever deleted unless ``prune`` is asked for.

Every function takes an optional ``client`` so the planning logic can be tested
without a NAS; the default is the ``smbclient`` module.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import smbclient

SMB_ENV = ("SMB_HOST", "SMB_SHARE", "SMB_USER", "SMB_PASSWORD")


def smb_config(env) -> dict[str, str] | None:
    """Resolve the SMB_* env vars. Returns {host, share, user, password} or
    None when none are set. Errors out if only some are set."""
    values = {key: str(env.get(key, "")).strip() for key in SMB_ENV}
    if not any(values.values()):
        return None
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise SystemExit(
            f"SMB_* partially configured; missing {', '.join(missing)} "
            "(set all four in .env or leave all unset to skip uploads)"
        )
    return values


def client_for(cfg: dict[str, str]):
    """Register the session and return the smbclient module."""
    smbclient.register_session(cfg["SMB_HOST"], username=cfg["SMB_USER"], password=cfg["SMB_PASSWORD"])
    return smbclient


def remote_base(cfg: dict[str, str], person: str) -> str:
    return f"//{cfg['SMB_HOST']}/{cfg['SMB_SHARE']}/{person}"


def remote_path(cfg: dict[str, str], person: str, album: str, name: str) -> str:
    return f"{remote_base(cfg, person)}/{album}/{name}"


def local_files(matched: Path, person: str) -> list[Path]:
    """Every file in ``matched/<person>/<album>/``."""
    root = matched / person
    return sorted(p for p in root.glob("*/*") if p.is_file()) if root.is_dir() else []


def _remote_size(client, path: str) -> int | None:
    try:
        return int(client.stat(path).st_size)
    except Exception:
        return None


def plan_uploads(matched: Path, person: str, cfg: dict[str, str], client=None) -> dict[str, list]:
    """Split the local files into to-upload vs already-there (same size)."""
    client = client or client_for(cfg)
    uploads, skipped = [], []
    for local in local_files(matched, person):
        remote = remote_path(cfg, person, local.parent.name, local.name)
        if _remote_size(client, remote) == local.stat().st_size:
            skipped.append(remote)
        else:
            uploads.append((local, remote))
    return {"uploads": uploads, "skipped": skipped}


def remote_files(client, base: str) -> list[str]:
    """Every file path under ``base`` on the share (empty when it is missing)."""
    out, stack = [], [base]
    while stack:
        current = stack.pop()
        try:
            entries = list(client.scandir(current))
        except Exception:
            continue
        for entry in entries:
            path = f"{current}/{entry.name}"
            if entry.is_dir():
                stack.append(path)
            else:
                out.append(path)
    return sorted(out)


def prune_targets(matched: Path, person: str, cfg: dict[str, str], client=None) -> list[str]:
    """Remote files under ``<person>/`` that no longer exist locally."""
    client = client or client_for(cfg)
    keep = {remote_path(cfg, person, p.parent.name, p.name) for p in local_files(matched, person)}
    return [remote for remote in remote_files(client, remote_base(cfg, person)) if remote not in keep]


def upload_files(uploads: list[tuple[Path, str]], client=None) -> int:
    """Copy (local, remote) pairs to the share; returns the number uploaded."""
    client = client or smbclient
    for local, remote in uploads:
        parent = remote.rsplit("/", 1)[0]
        client.makedirs(parent, exist_ok=True)
        with open(local, "rb") as src, client.open_file(remote, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return len(uploads)


def upload_report(stats: dict[str, list], prunable: list[str] | None = None) -> str:
    """One-line summary of a planned or completed upload."""
    line = f"{len(stats['uploads'])} to upload, {len(stats['skipped'])} already there"
    if prunable:
        line += f", {len(prunable)} remote-only (use --delete to remove)"
    return line


def upload_body(uploads: list[tuple[Path, str]], skipped: int, removed: int, base: str) -> str:
    """Multi-line notification body for a completed upload."""
    header = f"{len(uploads)} new photo{'s' if len(uploads) != 1 else ''} · {skipped} already there"
    albums = Counter(local.parent.name for local, _remote in uploads)
    lines = [header, *(f"- {album} ({count})" for album, count in sorted(albums.items()))]
    if removed:
        lines.append(f"{removed} removed")
    lines.append(f"→ {base}")
    return "\n".join(lines)
