import io
import urllib.error

import codmon_api
import pytest
from codmon_api import CodmonClient, CodmonError, CodmonNotFound


def test_is_photo_album_ignores_forms_and_empty_payloads():
    assert CodmonClient._is_photo_album({"photos": {"lists": [{"id": "1"}]}})
    assert CodmonClient._is_photo_album({"photo_count": 45})
    assert not CodmonClient._is_photo_album({"title": "遅刻・欠席連絡"})
    assert not CodmonClient._is_photo_album({"photos": {"lists": []}, "photo_count": 0})


def test_request_maps_404_to_not_found(monkeypatch):
    client = CodmonClient("e", "p")

    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(client._opener, "open", boom)
    with pytest.raises(CodmonNotFound):
        client._request("GET", "/api/v2/parent/albums/1/")

    def server_error(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {}, io.BytesIO(b""))

    monkeypatch.setattr(client._opener, "open", server_error)
    with pytest.raises(CodmonError) as exc:
        client._request("GET", "/api/v2/parent/albums/1/")
    assert not isinstance(exc.value, CodmonNotFound)


def test_albums_skips_forms_and_404s(monkeypatch):
    """A kind=8 timeline mixes real albums with non-photo forms; only albums survive."""
    client = CodmonClient("e", "p")
    client._service_ids = ["1"]
    asked = []

    def fake_request(method, path, *, params=None, json_body=None):
        if path == "/api/v2/parent/timeline/":
            return {
                "data": [
                    {
                        "id": "111",
                        "kind": 8,
                        "title": "birthday",
                        "display_date": "2026-10-23",
                        "photos": {"lists": [{"id": "p1", "url": "https://image.codmon.com/a.jpg"}]},
                    },
                    {"id": "222", "kind": 8, "title": "遅刻・欠席連絡"},
                    {
                        "id": "333",
                        "kind": 8,
                        "title": "deleted",
                        "photo_count": 40,
                        "photos": {"lists": [{"id": "p2", "url": "https://image.codmon.com/b.jpg"}]},
                    },
                    {"id": "444", "kind": 1, "title": "notice"},
                ],
                "next_page": 0,
            }
        asked.append(path)
        if path == "/api/v2/parent/albums/111/":
            return {
                "data": {
                    "title": "birthday",
                    "display_date": "2026-10-23",
                    "insert_datetime": "2026-10-23 12:11:03",
                    "photos": [{"id": "p1", "url": "https://image.codmon.com/a.jpg"}],
                }
            }
        if path == "/api/v2/parent/albums/333/":
            raise CodmonNotFound("Codmon API HTTP 404 from GET /api/v2/parent/albums/333/")
        raise AssertionError(f"unexpected request: {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    albums = client.albums("2026-10-01", "2026-10-31")

    assert [a.album_id for a in albums] == ["111"]
    assert albums[0].photos == [("p1", "https://image.codmon.com/a.jpg")]
    assert "/api/v2/parent/albums/222/" not in asked, "a form must not be fetched at all"
    assert "/api/v2/parent/albums/444/" not in asked, "non-photo kinds are ignored"


def test_albums_reports_other_errors(monkeypatch):
    client = CodmonClient("e", "p")
    client._service_ids = ["1"]

    def fake_request(method, path, *, params=None, json_body=None):
        if path == "/api/v2/parent/timeline/":
            return {
                "data": [{"id": "111", "kind": 8, "photo_count": 2, "photos": {"lists": [{"id": "p"}]}}],
                "next_page": 0,
            }
        raise CodmonError("Codmon API HTTP 500 from GET /api/v2/parent/albums/111/")

    monkeypatch.setattr(client, "_request", fake_request)
    with pytest.raises(CodmonError, match="500"):
        client.albums("2026-10-01", "2026-10-31")


def test_codmon_api_exports():
    assert issubclass(CodmonNotFound, CodmonError)
    assert codmon_api.PHOTO_KIND == "8"
