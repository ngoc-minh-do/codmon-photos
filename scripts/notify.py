"""Apprise-style webhook helper.

Posts a small JSON payload to an Apprise API endpoint (``APPRISE_URL``). Never
raises: a failed notification must not fail the run that triggered it.
"""

from __future__ import annotations

import json
import urllib.request


def notify(url: str | None, title: str, body: str) -> bool:
    """POST a success notification. Returns False when disabled or on failure."""
    url = (url or "").strip()
    if not url:
        return False
    payload = json.dumps({"title": title, "body": body, "type": "success"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        return True
    except Exception as exc:
        print(f"   -> Apprise notification failed: {exc}")
        return False
