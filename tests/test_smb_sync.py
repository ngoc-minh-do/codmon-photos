import types

import pytest
import smb_sync


class FakeFile:
    def __init__(self, store, path):
        self.store, self.path = store, path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, data):
        self.store[self.path] = self.store.get(self.path, b"") + data
        return len(data)


class FakeEntry:
    def __init__(self, name, is_dir):
        self.name, self._dir = name, is_dir

    def is_dir(self):
        return self._dir


class FakeClient:
    """Minimal stand-in for the smbclient module (name+size skipping, listing)."""

    def __init__(self, files=None, dirs=None):
        self.files = dict(files or {})  # remote path -> bytes
        self.dirs = set(dirs or [])
        self.made = []

    def stat(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return types.SimpleNamespace(st_size=len(self.files[path]))

    def makedirs(self, path, exist_ok=False):
        self.made.append(path)

    def open_file(self, path, mode):
        return FakeFile(self.files, path)

    def remove(self, path):
        self.files.pop(path, None)

    def scandir(self, path):
        prefix = path.rstrip("/") + "/"
        seen = {}
        for known in list(self.files) + list(self.dirs):
            if not known.startswith(prefix):
                continue
            head, _, rest = known[len(prefix) :].partition("/")
            if head:
                seen[head] = seen.get(head, False) or bool(rest)
        if not seen:
            raise FileNotFoundError(path)
        return [FakeEntry(name, isdir) for name, isdir in sorted(seen.items())]


CFG = {"SMB_HOST": "nas", "SMB_SHARE": "photos", "SMB_USER": "u", "SMB_PASSWORD": "p"}


def make_matched(tmp_path, files):
    """Build matched/<person>/<album>/<name> and return the matched root."""
    root = tmp_path / "matched"
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def test_smb_config_none_when_unset_and_errors_when_partial():
    assert smb_sync.smb_config({}) is None
    assert smb_sync.smb_config({"SMB_HOST": " ", "SMB_SHARE": ""}) is None
    with pytest.raises(SystemExit, match="missing SMB_SHARE"):
        smb_sync.smb_config({"SMB_HOST": "nas"})


def test_remote_path_layout():
    assert smb_sync.remote_base(CFG, "kid") == "//nas/photos/kid"
    assert smb_sync.remote_path(CFG, "kid", "2026-09-25", "a.jpg") == "//nas/photos/kid/2026-09-25/a.jpg"


def test_plan_uploads_skips_same_size_and_queues_the_rest(tmp_path):
    matched = make_matched(tmp_path, {"kid/album/a.jpg": b"aaaa", "kid/album/b.jpg": b"bb"})
    client = FakeClient({"//nas/photos/kid/album/a.jpg": b"aaaa"})

    stats = smb_sync.plan_uploads(matched, "kid", CFG, client)

    assert [remote for _local, remote in stats["uploads"]] == ["//nas/photos/kid/album/b.jpg"]
    assert stats["skipped"] == ["//nas/photos/kid/album/a.jpg"]


def test_plan_uploads_reuploads_when_size_differs(tmp_path):
    matched = make_matched(tmp_path, {"kid/album/a.jpg": b"aaaa"})
    client = FakeClient({"//nas/photos/kid/album/a.jpg": b"aa"})

    stats = smb_sync.plan_uploads(matched, "kid", CFG, client)

    assert len(stats["uploads"]) == 1 and stats["skipped"] == []


def test_plan_uploads_empty_without_matches(tmp_path):
    stats = smb_sync.plan_uploads(tmp_path / "matched", "kid", CFG, FakeClient())
    assert stats == {"uploads": [], "skipped": []}


def test_upload_files_writes_bytes_and_creates_parents(tmp_path):
    matched = make_matched(tmp_path, {"kid/album/a.jpg": b"hello"})
    client = FakeClient()
    uploads = smb_sync.plan_uploads(matched, "kid", CFG, client)["uploads"]

    assert smb_sync.upload_files(uploads, client) == 1
    assert client.files["//nas/photos/kid/album/a.jpg"] == b"hello"
    assert client.made == ["//nas/photos/kid/album"]


def test_prune_targets_reports_remote_files_without_a_local_twin(tmp_path):
    matched = make_matched(tmp_path, {"kid/album/keep.jpg": b"k"})
    client = FakeClient(
        {
            "//nas/photos/kid/album/keep.jpg": b"k",
            "//nas/photos/kid/album/gone.jpg": b"g",
            "//nas/photos/other/album/notmine.jpg": b"n",
        }
    )

    assert smb_sync.prune_targets(matched, "kid", CFG, client) == ["//nas/photos/kid/album/gone.jpg"]


def test_remote_files_walks_nested_dirs(tmp_path):
    client = FakeClient({"//nas/photos/kid/a/x.jpg": b"x", "//nas/photos/kid/b/c/y.jpg": b"y"})
    assert smb_sync.remote_files(client, "//nas/photos/kid") == [
        "//nas/photos/kid/a/x.jpg",
        "//nas/photos/kid/b/c/y.jpg",
    ]


def test_upload_report_mentions_prunable_only_when_asked():
    stats = {"uploads": [1, 2], "skipped": [3]}
    assert smb_sync.upload_report(stats) == "2 to upload, 1 already there"
    assert (
        smb_sync.upload_report(stats, ["r"]) == "2 to upload, 1 already there, 1 remote-only (use --delete to remove)"
    )


def test_upload_body_lists_new_albums_and_optional_removals(tmp_path):
    matched = make_matched(tmp_path, {"kid/album a/one.jpg": b"1", "kid/album b/two.jpg": b"2"})
    uploads = smb_sync.plan_uploads(matched, "kid", CFG, FakeClient())["uploads"]

    assert smb_sync.upload_body(uploads, 3, 0, "//nas/photos/kid") == (
        "2 new photos · 3 already there\n- album a (1)\n- album b (1)\n→ //nas/photos/kid"
    )
    assert smb_sync.upload_body([], 0, 2, "//nas/photos/kid") == (
        "0 new photos · 0 already there\n2 removed\n→ //nas/photos/kid"
    )


def test_upload_report_is_planning_only_wording_free_of_confusion():
    """The CLI adds the dry-run line; the report itself must not claim success."""
    stats = {"uploads": [1, 2], "skipped": []}
    text = smb_sync.upload_report(stats)
    assert text == "2 to upload, 0 already there"
    assert "uploaded" not in text
