"""Face detection + recognition for Codmon albums.

Reference photos of your child live in ``refs/<person>/face/*.jpg`` (the only
human-provided input). Everything derived — face boxes, embeddings, scores —
lives in ``.cache/faces.db`` (sqlite), and the ONNX models in ``.cache/models``.

Two-stage pipeline, both plain ONNX/OpenCV:

1. **detect** — YuNet finds faces and 5 landmarks (it holds up better than SCRFD
   on the small faces a 500px CDN photo leaves behind).
2. **recognize** — the 5 landmarks are aligned to the ArcFace template and fed
   to ArcFace (``w600k_r50``, the recognition half of InsightFace's buffalo_l),
   yielding a 512-d embedding per face.

Rows are keyed by file *content* (sha1), not path, so a photo is embedded once no
matter where it shows up — local album, temp staging dir, or a re-download — and
changing the threshold never re-runs a model. Matches are copied out; the album
folders are never modified.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import sqlite3
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_DB = Path(".cache/faces.db")
DEFAULT_MODELS = Path(".cache/models")
DEFAULT_PHOTOS = Path("downloads")
DEFAULT_MATCHED = Path("matched")
DEFAULT_REFS = Path("refs")
DEFAULT_MIN_FACE = 40  # narrower faces are detected but not judged
DEFAULT_FLOOR = 20  # faces narrower than this are never stored
DEFAULT_THRESHOLD = 0.45
BANDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)

# ArcFace 5-point template (112x112); same constants as insightface's norm_crop.
ARC_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# (url, sha256). Buffalo_l ships as a zip; only the ArcFace head is extracted.
# An existing ~/.insightface copy is reused instead of downloading.
MODELS = {
    "yunet": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "arcface": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    ),
}
ARC_MEMBER = "w600k_r50.onnx"
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    sha1 TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    scanned_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS faces (
    sha1 TEXT NOT NULL,
    idx INTEGER NOT NULL,
    x REAL, y REAL, w REAL, h REAL,
    det_score REAL,
    emb BLOB NOT NULL,
    PRIMARY KEY (sha1, idx)
);
CREATE TABLE IF NOT EXISTS refs (
    person TEXT NOT NULL,
    source TEXT NOT NULL,
    crop TEXT NOT NULL,
    emb BLOB NOT NULL,
    PRIMARY KEY (person, source)
);
"""


# --- models -----------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"   downloading {url.rsplit('/', 1)[-1]} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def ensure_model(name: str, models_dir: Path = DEFAULT_MODELS) -> Path:
    """Path to a verified model file, downloading it once when missing."""
    url, want = MODELS[name]
    dest = models_dir / ("yunet.onnx" if name == "yunet" else "arcface_r50.onnx")
    if dest.exists() and sha256_file(dest) == want:
        return dest
    local = Path.home() / ".insightface/models/buffalo_l" / ARC_MEMBER
    if name == "arcface" and local.exists() and sha256_file(local) == want:
        return local  # reuse insightface's copy instead of duplicating 167 MB
    archive = models_dir / f"{name}.archive"
    _download(url, archive)
    if name == "arcface":
        with zipfile.ZipFile(archive) as zf, zf.open(ARC_MEMBER) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        archive.unlink()
    else:
        archive.replace(dest)
    got = sha256_file(dest)
    if got != want:
        raise SystemExit(f"{dest} failed its sha256 check (got {got})")
    return dest


def make_detector(model_path: Path):
    """YuNet at score 0.6 / NMS 0.3, keeping up to 5000 candidates."""
    import cv2

    return cv2.FaceDetectorYN.create(str(model_path), "", (640, 640), 0.6, 0.3, 5000)


def make_recognizer(model_path: Path):
    """ArcFace session; returns (session, input name)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def load_models(models_dir: Path = DEFAULT_MODELS):
    return make_detector(ensure_model("yunet", models_dir)), make_recognizer(ensure_model("arcface", models_dir))


# --- per-image work ---------------------------------------------------------


def detect(img, det, min_width: float = DEFAULT_FLOOR):
    """Faces wider than ``min_width`` as (box, 5 landmarks, det score)."""
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    out = []
    for f in faces if faces is not None else []:
        if f[2] < min_width:
            continue
        out.append(((float(f[0]), float(f[1]), float(f[2]), float(f[3])), f[4:14].reshape(5, 2), float(f[14])))
    return out


def align(img, landmarks, size: int = 112):
    """Warp one face onto the ArcFace template using its 5 landmarks."""
    import cv2

    m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARC_TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img, m, (size, size), borderValue=0.0)


def embed(img, landmarks, rec) -> np.ndarray:
    """L2-normalized 512-d ArcFace embedding for one aligned face."""
    import cv2

    sess, name = rec
    blob = cv2.dnn.blobFromImage(align(img, landmarks), 1.0 / 128, (112, 112), (127.5, 127.5), swapRB=True)
    vec = sess.run(None, {name: blob})[0].flatten().astype(np.float32)
    return vec / np.linalg.norm(vec)


def score_faces(faces, gallery: np.ndarray) -> tuple[float, int]:
    """Best (cosine, face index) of any face against the reference gallery."""
    if not faces or not gallery.size:
        return 0.0, -1
    per_face = (np.stack([f[2] for f in faces]) @ gallery.T).max(axis=1)
    idx = int(per_face.argmax())
    return float(per_face[idx]), idx


# --- storage ----------------------------------------------------------------


def open_db(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_photo(db: sqlite3.Connection, sha1: str, size: tuple[int, int], model: str, faces) -> None:
    """Cache one photo's faces under its content hash."""
    db.execute("DELETE FROM faces WHERE sha1 = ?", (sha1,))
    db.execute(
        "INSERT OR REPLACE INTO photos (sha1, model, width, height, scanned_at) VALUES (?,?,?,?,?)",
        (sha1, model, size[0], size[1], time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    db.executemany(
        "INSERT INTO faces (sha1, idx, x, y, w, h, det_score, emb) VALUES (?,?,?,?,?,?,?,?)",
        [(sha1, i, *box, score, emb.tobytes()) for i, (box, score, emb) in enumerate(faces)],
    )
    db.commit()


def cached_photo(db: sqlite3.Connection, sha1: str, model: str) -> bool:
    row = db.execute("SELECT model FROM photos WHERE sha1 = ?", (sha1,)).fetchone()
    return bool(row) and row[0] == model


def load_faces(db: sqlite3.Connection, sha1: str) -> list[tuple[tuple[float, float, float, float], float, np.ndarray]]:
    rows = db.execute("SELECT x, y, w, h, det_score, emb FROM faces WHERE sha1 = ? ORDER BY idx", (sha1,))
    return [((r[0], r[1], r[2], r[3]), r[4], np.frombuffer(r[5], dtype=np.float32)) for r in rows]


def gallery(db: sqlite3.Connection, person: str) -> np.ndarray:
    rows = db.execute("SELECT emb FROM refs WHERE person = ?", (person,)).fetchall()
    if not rows:
        raise SystemExit(f"no references enrolled for {person!r} — enroll first")
    return np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])


def photo_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PHOTO_EXTS)


def album_dirs(root: Path) -> list[Path]:
    """Directories under ``root`` that hold photos themselves."""
    return sorted({p.parent for p in photo_files(root) if p.parent != root})


# --- steps ------------------------------------------------------------------


def enroll(db: sqlite3.Connection, person: str, ref_dir: Path, det, rec, crop_dir: Path) -> list[str]:
    """Embed every photo in ``ref_dir`` (largest face each) as a reference.

    Returns human-readable warnings: multiple faces in a reference, a reference
    too small to be trusted, or a reference that disagrees with the others.
    """
    import cv2

    files = photo_files(ref_dir)
    if not files:
        raise SystemExit(f"no reference photos in {ref_dir}/ — add 4-6 face photos of {person!r}")
    crops = crop_dir / person
    crops.mkdir(parents=True, exist_ok=True)
    db.execute("DELETE FROM refs WHERE person = ?", (person,))
    warnings = []
    embs, sources = [], []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            warnings.append(f"{path.name}: unreadable")
            continue
        faces = detect(img, det, 16)
        if not faces:
            warnings.append(f"{path.name}: no face found — skipped")
            continue
        box, kps, _score = max(faces, key=lambda f: f[0][2])
        if box[2] < 64:
            warnings.append(f"{path.name}: face only {box[2]:.0f}px wide — a bigger photo would help")
        if len(faces) > 1:
            warnings.append(f"{path.name}: {len(faces)} faces, kept the largest ({box[2]:.0f}px)")
        emb = embed(img, kps, rec)
        crop_path = crops / f"{len(embs):02d}_{path.stem}.jpg"
        cv2.imwrite(str(crop_path), align(img, kps))
        embs.append(emb)
        sources.append((str(path), str(crop_path)))
    if not embs:
        raise SystemExit(f"could not enroll anything from {ref_dir}/")
    for (source, crop), emb in zip(sources, embs, strict=True):
        db.execute(
            "INSERT INTO refs (person, source, crop, emb) VALUES (?,?,?,?)", (person, source, crop, emb.tobytes())
        )
    db.commit()
    if len(embs) > 1:
        stack = np.stack(embs)
        sims = stack @ stack.T
        np.fill_diagonal(sims, 0.0)
        for (source, _crop), worst in zip(sources, sims.max(axis=1), strict=True):
            if worst < 0.35:
                warnings.append(f"{Path(source).name}: only {worst:.2f} similar to the others — sure it's her?")
    return warnings


def scan(
    db: sqlite3.Connection, roots: list[Path], det, rec, model: str, floor: float = DEFAULT_FLOOR
) -> tuple[int, int]:
    """Embed every uncached photo under ``roots``; returns (embedded, already cached)."""
    import cv2

    done = skipped = 0
    for root in roots:
        for path in photo_files(root):
            sha1 = file_sha1(path)
            if cached_photo(db, sha1, model):
                skipped += 1
                continue
            img = cv2.imread(str(path))
            if img is None:
                print(f"   unreadable: {path}")
                continue
            faces = [(box, score, embed(img, kps, rec)) for box, kps, score in detect(img, det, floor)]
            store_photo(db, sha1, (img.shape[1], img.shape[0]), model, faces)
            done += 1
            print(f"   {path.parent.name}/{path.name}: {len(faces)} face(s) >= {floor:.0f}px")
    return done, skipped


def collect(db: sqlite3.Connection, person: str, roots: list[Path], min_face: float = DEFAULT_MIN_FACE) -> list[dict]:
    """Per-photo rows: best score, best face box, face count above ``min_face``."""
    gal = gallery(db, person)
    rows = []
    for root in roots:
        for album in album_dirs(root):
            for path in photo_files(album):
                faces = [f for f in load_faces(db, file_sha1(path)) if f[0][2] >= min_face]
                score, idx = score_faces(faces, gal)
                best_box = faces[idx][0] if idx >= 0 else None
                rows.append(
                    {
                        "album": album.name,
                        "path": path,
                        "root": root,
                        "score": score,
                        "faces": len(faces),
                        "face_width": best_box[2] if best_box else 0.0,
                        "best_box": best_box,
                    }
                )
    return rows


# --- decisions --------------------------------------------------------------


def band_of(score: float, edges=BANDS) -> str:
    """Which candidate-threshold interval the score falls into."""
    if score < edges[0]:
        return f"<{edges[0]:.2f}"
    for low, high in zip(edges, edges[1:], strict=False):
        if score < high:
            return f"{low:.2f}-{high:.2f}"
    return f">={edges[-1]:.2f}"


def sweep(rows: list[dict], edges=BANDS) -> list[tuple[float, int, float]]:
    """(threshold, photos kept, share of all photos kept) per candidate threshold."""
    total = len(rows) or 1
    scores = [r["score"] for r in rows if r["faces"]]
    return [(t, sum(1 for s in scores if s >= t), sum(1 for s in scores if s >= t) / total * 100) for t in edges]


def matched_rows(rows: list[dict], threshold: float) -> list[dict]:
    """Photos whose best face scores at or above ``threshold``."""
    return [r for r in rows if r["faces"] and r["score"] >= threshold]


def export_matches(
    rows: list[dict],
    person: str,
    out: Path,
    threshold: float,
    *,
    link: bool = False,
    prune: bool = False,
    apply: bool = False,
) -> dict[str, int]:
    """Copy matches to ``<out>/<person>/<album>/``; the album folders stay untouched.

    Existing files are left alone unless ``prune`` is set, which removes copies
    under ``<out>/<person>/`` that no longer match (so lowering the threshold
    does not leave stale photos behind).
    """
    want: dict[Path, Path] = {}
    for r in matched_rows(rows, threshold):
        want[out / person / r["album"] / r["path"].name] = r["path"]
    stats = {"copied": 0, "kept": 0, "pruned": 0}
    for dest, src in want.items():
        if dest.exists():
            stats["kept"] += 1
            continue
        stats["copied"] += 1
        if not apply:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if link:
            os.link(src, dest)
        else:
            shutil.copy2(src, dest)
    stale = []
    if prune:
        for path in photo_files(out / person):
            if path not in want:
                stale.append(path)
                if apply:
                    path.unlink()
        stats["pruned"] = len(stale)
    return stats


# --- review report ----------------------------------------------------------


def _thumb_b64(img, max_w: int = 420) -> str:
    import cv2

    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * min(1.0, max_w / w)), int(h * min(1.0, max_w / w))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _crop_b64(img, box, margin: float = 0.35, height: int = 140) -> str:
    import cv2

    if box is None:
        return ""
    x, y, w, h = box
    dx, dy = w * margin, h * margin
    patch = img[
        max(0, int(y - dy)) : min(img.shape[0], int(y + h + dy)),
        max(0, int(x - dx)) : min(img.shape[1], int(x + w + dx)),
    ]
    if patch.size == 0:
        return ""
    patch = cv2.resize(patch, (max(1, int(height * patch.shape[1] / patch.shape[0])), height))
    ok, buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _file_b64(path: Path) -> str:
    data = path.read_bytes() if path.exists() else b""
    return "data:image/jpeg;base64," + base64.b64encode(data).decode() if data else ""


REPORT_CSS = """
body{font:14px/1.45 system-ui,sans-serif;background:#111;color:#ddd;margin:24px}
h1{font-size:18px} h2{font-size:15px;margin-top:26px;border-bottom:1px solid #333;padding-bottom:4px}
table{border-collapse:collapse;margin:6px 0} td,th{padding:2px 10px;border-bottom:1px solid #262626;text-align:left}
.card{display:inline-block;vertical-align:top;width:430px;margin:6px;background:#1a1a1a;border-radius:6px;padding:8px}
.card img.ph{width:100%;border-radius:4px} .row{display:flex;gap:8px;align-items:center;margin-top:4px}
.score{font-weight:700;font-variant-numeric:tabular-nums} .warn{color:#f0a020} .ok{color:#5ec26a} .muted{color:#777}
.path{font:11px ui-monospace,monospace;color:#888;word-break:break-all}
"""


def write_report(
    db: sqlite3.Connection, person: str, rows: list[dict], out: Path, threshold: float, min_face: float
) -> Path:
    """Self-contained HTML review sheet: every photo, its best score and face crop."""
    import cv2

    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"<h1>Scores for {person!r} — threshold {threshold:.2f}, min face {min_face:.0f}px</h1>"]
    parts.append("<h2>Enrolled references (verify these are all her)</h2><div>")
    for source, crop in db.execute("SELECT source, crop FROM refs WHERE person = ? ORDER BY source", (person,)):
        parts.append(
            f'<span class="card" style="width:140px"><img class="ph" src="{_file_b64(Path(crop))}">'
            f'<div class="path">{Path(source).name}</div></span>'
        )
    parts.append("</div>")

    unknown = sum(1 for r in rows if not r["faces"])
    parts.append(f"<h2>{len(rows)} photos, {unknown} with no usable face (&lt; {min_face:.0f}px)</h2>")
    parts.append("<table><tr><th>threshold</th><th>photos kept</th><th>share kept</th></tr>")
    parts.extend(f"<tr><td>{t:.2f}</td><td>{kept}</td><td>{share:.0f}%</td></tr>" for t, kept, share in sweep(rows))
    parts.append("</table>")

    for album in sorted({r["album"] for r in rows}):
        group = [r for r in rows if r["album"] == album]
        hits = sum(1 for r in group if r["faces"] and r["score"] >= threshold)
        misses = sum(1 for r in group if not r["faces"])
        parts.append(f"<h2>{album} — {len(group)} photos, {hits} at {threshold:.2f}, {misses} unusable</h2>")
        for r in sorted(group, key=lambda r: -r["score"]):
            img = cv2.imread(str(r["path"]))
            if img is None:
                continue
            keep = bool(r["faces"]) and r["score"] >= threshold
            cls = "ok" if keep else ("warn" if not r["faces"] else "")
            note = "no usable face" if not r["faces"] else f"{r['faces']} face(s) ≥ {min_face:.0f}px"
            parts.append(
                f'<div class="card"><img class="ph" src="{_thumb_b64(img)}">'
                f'<div class="row"><span class="score {cls}">{r["score"]:.3f}</span>'
                f'<span class="muted">{band_of(r["score"])}</span>'
                f'<img src="{_crop_b64(img, r["best_box"])}" style="height:70px">'
                f'<span class="muted">{note}</span></div>'
                f'<div class="path">{r["path"].relative_to(r["root"])}</div></div>'
            )

    html = f"<!doctype html><meta charset=utf-8><title>face review</title><style>{REPORT_CSS}</style>\n" + "\n".join(
        parts
    )
    out.write_text(html, encoding="utf-8")
    return out
