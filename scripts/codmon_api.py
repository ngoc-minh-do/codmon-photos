"""Direct Codmon API client — no response.json required.

Codmon exposes no public parent API, but parents.codmon.com is an SPA that
talks to an internal JSON API (ps-api.codmon.com) using session cookies from a
plain email/password login. This client mirrors the one used by the
codmon-huckleberry-sync project: login sets the cookies, /children resolves
the nursery service_id (with use_image_edge=true so photo URLs come back
signed for the CloudFront CDN), and a /timeline scan returns the photo albums
(kind=8) whose ``photos`` array carries ``{id, url}`` entries.
"""
from __future__ import annotations

import http.cookiejar
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

API_BASE = "https://ps-api.codmon.com"
ENV_PARAM = "__env__"
ENV_VALUE = "myapp"
PHOTO_KIND = "8"


class CodmonError(RuntimeError):
    pass


@dataclass
class Album:
    album_id: str
    title: str
    display_date: str  # YYYY-MM-DD
    insert_datetime: str  # 'YYYY-MM-DD HH:MM:SS' (Tokyo time)
    photos: list[tuple[str, str]]  # (photo_id, signed_url)


class CodmonClient:
    def __init__(self, email: str, password: str) -> None:
        jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self._email = email
        self._password = password
        self._service_ids: list[str] | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: list[tuple[str, str]] | dict[str, str] | None = None,
        json_body: dict | None = None,
    ) -> dict:
        url = f"{API_BASE}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        data = None
        headers = {}
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=60) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            raise CodmonError(f"Codmon API HTTP {exc.code} from {method} {path}") from exc
        except urllib.error.URLError as exc:
            raise CodmonError(f"Codmon API unreachable ({exc.reason}) for {method} {path}") from exc
        if not isinstance(payload, dict):
            raise CodmonError(f"Unexpected response shape from {path}")
        if payload.get("success") is False:
            raise CodmonError(f"Codmon API error from {path}: {payload.get('error')}")
        return payload

    def login(self) -> None:
        body = self._request(
            "POST",
            "/api/v2/parent/login",
            params={ENV_PARAM: ENV_VALUE},
            json_body={
                "login_id": self._email,
                "login_password": self._password,
                "use_db_replica": 1,
            },
        )
        if not body.get("success"):
            raise CodmonError(f"Codmon login failed: {body.get('error')}")

    def service_ids(self) -> list[str]:
        if self._service_ids is None:
            self.login()
            body = self._request(
                "GET",
                "/api/v2/parent/children/",
                params={ENV_PARAM: ENV_VALUE, "use_db_replica": 1, "use_image_edge": "true"},
            )
            ids: set[str] = set()
            for child in body.get("data", []):
                for relation in child.get("child_member_relations", []):
                    sid = relation.get("service_id")
                    if sid:
                        ids.add(str(sid))
            if not ids:
                raise CodmonError("No nursery service found for this Codmon account")
            self._service_ids = sorted(ids)
        return self._service_ids

    def albums(self, start: str, end: str) -> list[Album]:
        """Scan the timeline in [start, end] (YYYY-MM-DD), then fetch each photo
        album's full photo list from the albums detail endpoint. Newest first.

        The timeline list only embeds signed URLs for ~5 preview photos per
        album; the detail endpoint pages every photo (20/page) with a signed
        URL each.
        """
        albums: list[Album] = []
        seen: set[str] = set()
        for service_id in self.service_ids():
            page = 1
            while True:
                params = [
                    (ENV_PARAM, ENV_VALUE),
                    ("listpage", str(page)),
                    ("search_type[]", "new_all"),
                    ("start_date", start),
                    ("end_date", end),
                    ("service_id", service_id),
                    ("current_flag", "0"),
                    ("use_image_edge", "true"),
                    ("bookmark_only", "false"),
                ]
                body = self._request("GET", "/api/v2/parent/timeline/", params=params)
                for item in body.get("data", []):
                    if str(item.get("kind")) != PHOTO_KIND:
                        continue
                    album_id = str(item.get("id", ""))
                    if album_id in seen:
                        continue
                    detail, photos = self.album_detail(album_id)
                    if not photos:
                        continue
                    seen.add(album_id)
                    display = str(detail.get("display_date") or item.get("display_date") or "")
                    if len(display) < 10:
                        display = str(detail.get("insert_datetime") or "")[:10]
                    albums.append(
                        Album(
                            album_id=album_id,
                            title=str(detail.get("title") or item.get("title") or ""),
                            display_date=display[:10],
                            insert_datetime=str(
                                detail.get("insert_datetime") or item.get("insert_datetime") or ""
                            ),
                            photos=photos,
                        )
                    )
                next_page = body.get("next_page")
                if not isinstance(next_page, int) or next_page <= page:
                    break
                page = next_page
        albums.sort(key=lambda a: (a.display_date, a.insert_datetime), reverse=True)
        return albums

    @staticmethod
    def _photo_rows(item: dict) -> list[tuple[str, str]]:
        """Extract (photo_id, signed_url) rows from an album item. The timeline
        list nests the rows under ``photos.lists``; the albums detail endpoint
        exposes a plain ``photos`` list."""
        raw = item.get("photos") or []
        if isinstance(raw, dict):
            raw = raw.get("lists") or []
        rows: list[tuple[str, str]] = []
        for photo in raw:
            if not isinstance(photo, dict):
                continue
            url = photo.get("url")
            if isinstance(url, str) and "codmon.com" in url:
                rows.append((photo.get("id"), url))
        return rows

    def album_detail(self, album_id: str) -> tuple[dict, list[tuple[str, str]]]:
        """Fetch an album's full payload and every (photo_id, signed_url) row.

        The albums detail endpoint pages the photos (20/page); loop until all
        pages are read. ''_request'' needs an authenticated session, so ensure
        login happened first.
        """
        if self._service_ids is None:
            self.login()
        page = 1
        album: dict = {}
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        while True:
            body = self._request(
                "GET",
                f"/api/v2/parent/albums/{album_id}/",
                params={
                    ENV_PARAM: ENV_VALUE,
                    "use_image_edge": "true",
                    "listpage": str(page),
                    "records_in_page": "1000",
                },
            )
            if not album:
                album = body.get("data") or {}
            for pid, url in self._photo_rows(body.get("data") or {}):
                if pid is not None and pid in seen:
                    continue
                if pid is not None:
                    seen.add(pid)
                rows.append((pid, url))
            next_page = body.get("next_page")
            if not isinstance(next_page, int) or next_page <= page:
                break
            page = next_page
        return album, rows