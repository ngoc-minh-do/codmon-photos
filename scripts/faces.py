"""Face detection + recognition for Codmon albums.

Reference photos of your child live in ``refs/<person>/face/*.jpg`` (the only
human-provided input). Everything derived — face boxes, embeddings, scores —
lives in ``.cache/faces.db`` (sqlite), and the ONNX models in ``.cache/models``.

Two-stage pipeline, both plain ONNX/OpenCV:

1. **detect** — a detector finds faces and 5 landmarks. Two are available and
   interchangeable: ``yunet`` (OpenCV, finds more of the tiny faces a 500px CDN
   photo leaves behind) and ``scrfd`` (``det_10g``, the detector inside
   InsightFace's buffalo_l).
2. **recognize** — the 5 landmarks are aligned to the ArcFace template and fed
   to ArcFace (``w600k_r50``, the recognition half of buffalo_l), yielding a
   512-d embedding per face.

Cache rows are keyed by file *content* (sha1) **plus a pipeline signature**
(detector, recognizer, template, size floor), so the same photo can hold
embeddings from several detectors at once — that is what makes side-by-side
comparison possible — while a changed component never reuses stale numbers.
Changing the threshold is a pure re-score. Matches are copied out; the album
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
YUNET = "yunet"
SCRFD = "scrfd"
DETECTORS = (YUNET, SCRFD)

MODELS = {
    # single-file models: (url, sha256)
    YUNET: (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    # buffalo_l members are extracted from one shared archive: (url, sha256)
    "arcface": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    ),
    SCRFD: (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    ),
}
BUFFALO_MEMBERS = {"arcface": "w600k_r50.onnx", SCRFD: "det_10g.onnx"}
ARC_MEMBER = BUFFALO_MEMBERS["arcface"]
MODEL_FILES = {YUNET: "yunet.onnx", "arcface": "arcface_r50.onnx", SCRFD: "scrfd_det_10g.onnx"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    sha1 TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    scanned_at TEXT NOT NULL,
    PRIMARY KEY (sha1, pipeline)
);
CREATE TABLE IF NOT EXISTS faces (
    sha1 TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    idx INTEGER NOT NULL,
    x REAL, y REAL, w REAL, h REAL,
    det_score REAL,
    emb BLOB NOT NULL,
    PRIMARY KEY (sha1, pipeline, idx)
);
CREATE TABLE IF NOT EXISTS refs (
    person TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    source TEXT NOT NULL,
    crop TEXT NOT NULL,
    emb BLOB NOT NULL,
    PRIMARY KEY (person, pipeline, source)
);
"""
# Bump when the stored vectors would change for the same inputs (detector set,
# recognition model, alignment template, or the detection floor).
PIPELINE_VERSION = "v2"
# Bump when the table shapes change; an older cache is rebuilt from scratch.
SCHEMA_VERSION = 3
CACHE_TABLES = ("faces", "photos", "refs")


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
    """Path to a verified model file, downloading it once when missing.

    ``yunet`` is a single file; ``arcface`` and ``scrfd`` are members of the
    buffalo_l archive and are extracted together, so asking for either one only
    downloads the zip once.
    """
    url, want = MODELS[name]
    dest = Path(models_dir) / MODEL_FILES[name]
    if dest.exists() and sha256_file(dest) == want:
        return dest
    if name in BUFFALO_MEMBERS:
        local = Path.home() / ".insightface/models/buffalo_l" / BUFFALO_MEMBERS[name]
        if local.exists() and sha256_file(local) == want:
            return local  # reuse insightface's copy instead of duplicating it
        archive = Path(models_dir) / "buffalo_l.zip"
        if not archive.exists():
            _download(url, archive)
        with zipfile.ZipFile(archive) as zf:
            for model, member in BUFFALO_MEMBERS.items():
                target = Path(models_dir) / MODEL_FILES[model]
                if target.exists() and sha256_file(target) == MODELS[model][1]:
                    continue
                with zf.open(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                got = sha256_file(target)
                if got != MODELS[model][1]:
                    raise SystemExit(f"{target} failed its sha256 check (got {got})")
        archive.unlink()
        return dest
    archive = Path(models_dir) / f"{name}.archive"
    _download(url, archive)
    archive.replace(dest)
    got = sha256_file(dest)
    if got != want:
        raise SystemExit(f"{dest} failed its sha256 check (got {got})")
    return dest


class YunetDetector:
    """OpenCV YuNet: strong on faces down to ~10px, includes 5 landmarks."""

    name = YUNET

    def __init__(self, model_path: Path, score: float = 0.6, nms: float = 0.3, top_k: int = 5000):
        import cv2

        self._cv = cv2.FaceDetectorYN.create(str(model_path), "", (640, 640), score, nms, top_k)

    def faces(self, img):
        h, w = img.shape[:2]
        self._cv.setInputSize((w, h))
        _, raw = self._cv.detect(img)
        if raw is None:
            return []
        return [
            ((float(r[0]), float(r[1]), float(r[2]), float(r[3])), r[4:14].reshape(5, 2), float(r[14])) for r in raw
        ]


def scrfd_anchors(height: int, width: int, stride: int, num_anchors: int = 2):
    """Anchor centres for one SCRFD feature level, in input-image pixels."""
    centres = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    centres = (centres * stride).reshape(-1, 2)
    if num_anchors > 1:
        centres = np.stack([centres] * num_anchors, axis=1).reshape(-1, 2)
    return centres


def distance2bbox(points, distance):
    """SCRFD box regression: distances from each anchor centre to the 4 edges."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    """SCRFD landmark regression: offsets from each anchor centre, 5 (x, y)."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def nms(dets, thresh: float):
    """Greedy NMS over (x1, y1, x2, y2, score) rows; returns kept indices."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class ScrfdDetector:
    """SCRFD det_10g (buffalo_l's detector) run directly on ONNX Runtime."""

    name = SCRFD
    STRIDES = (8, 16, 32)
    ANCHORS = 2

    def __init__(self, model_path: Path, input_size: int = 640, score: float = 0.5, nms_thresh: float = 0.4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self._sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self._outputs = [o.name for o in self._sess.get_outputs()]
        self._size = (input_size, input_size)
        self._score, self._nms = score, nms_thresh
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _forward(self, img):
        import cv2

        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, self._size, (127.5, 127.5), swapRB=True)
        outs = self._sess.run(self._outputs, {self._input: blob})
        batched = outs[0].ndim == 3
        height, width = blob.shape[2], blob.shape[3]
        scores, boxes, kpss = [], [], []
        for i, stride in enumerate(self.STRIDES):
            take = (lambda a: a[0]) if batched else (lambda a: a)
            score = take(outs[i]).ravel()
            box = take(outs[i + len(self.STRIDES)]) * stride
            kps = take(outs[i + 2 * len(self.STRIDES)]) * stride
            key = (height // stride, width // stride, stride)
            if key not in self._cache:
                self._cache[key] = scrfd_anchors(key[0], key[1], stride, self.ANCHORS)
            centres = self._cache[key]
            pos = np.where(score >= self._score)[0]
            scores.append(score[pos])
            boxes.append(distance2bbox(centres, box)[pos])
            kpss.append(distance2kps(centres, kps).reshape(-1, 5, 2)[pos])
        return np.concatenate(scores), np.concatenate(boxes), np.concatenate(kpss)

    def faces(self, img):
        import cv2

        h, w = img.shape[:2]
        model_ratio = self._size[1] / self._size[0]
        im_ratio = h / w
        if im_ratio > model_ratio:
            new_h, new_w = self._size[1], int(self._size[1] / im_ratio)
        else:
            new_w, new_h = self._size[0], int(self._size[0] * im_ratio)
        scale = new_h / h
        canvas = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
        scores, boxes, kpss = self._forward(canvas)
        if scores.size == 0:
            return []
        boxes, kpss = boxes / scale, kpss / scale
        dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
        order = np.argsort(-dets[:, 4], kind="stable")
        dets, kpss = dets[order], kpss[order]
        keep = nms(dets, self._nms)
        out = []
        for i in keep:
            x1, y1, x2, y2 = dets[i, :4]
            out.append(((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), kpss[i], float(dets[i, 4])))
        return out


def make_detector(name: str, model_path: Path):
    """Detector instance for ``name`` (``yunet`` or ``scrfd``)."""
    if name == SCRFD:
        return ScrfdDetector(model_path)
    if name == YUNET:
        return YunetDetector(model_path)
    raise SystemExit(f"unknown detector {name!r} (choose from {', '.join(DETECTORS)})")


def make_recognizer(model_path: Path):
    """ArcFace session; returns (session, input name)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def load_models(detector: str = YUNET, models_dir: Path = DEFAULT_MODELS):
    """(detector, recognizer) for ``detector``; downloads/verifies both models."""
    det = make_detector(detector, ensure_model(detector, models_dir))
    return det, make_recognizer(ensure_model("arcface", models_dir))


# --- per-image work ---------------------------------------------------------


def pipeline_id(det, floor: float = DEFAULT_FLOOR) -> str:
    """Cache key for one detection+recognition setup.

    Everything that changes the stored numbers is in here, so swapping a
    detector stores new rows instead of silently reusing the old vectors.
    """
    return f"{PIPELINE_VERSION}:{det.name}:{ARC_MEMBER}:arc112:floor{floor:g}"


def pipeline_for(db: sqlite3.Connection, detector: str, floor: float = DEFAULT_FLOOR) -> str:
    """Cached pipeline id for ``detector``, or the id a scan with it would use."""
    for known in pipelines(db):
        if detector_of(known) == detector:
            return known
    return f"{PIPELINE_VERSION}:{detector}:{ARC_MEMBER}:arc112:floor{floor:g}"


def detect(img, det, min_width: float = DEFAULT_FLOOR):
    """Faces wider than ``min_width`` as (box, 5 landmarks, det score)."""
    return [f for f in det.faces(img) if f[0][2] >= min_width]


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
    """Open the cache, rebuilding it when it was written by an older schema.

    The cache only holds values that can be recomputed from the photos, so a
    shape change drops the tables instead of migrating them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        stale = present & set(CACHE_TABLES)
        if stale:
            print(f"   cache schema changed (v{version} -> v{SCHEMA_VERSION}); rebuilding {path}")
        for table in CACHE_TABLES:
            db.execute(f"DROP TABLE IF EXISTS {table}")
        db.executescript(SCHEMA)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        db.commit()
        return db
    db.executescript(SCHEMA)
    return db


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_photo(db: sqlite3.Connection, sha1: str, size: tuple[int, int], pipeline: str, faces) -> None:
    """Cache one photo's faces under its content hash and pipeline id."""
    db.execute("DELETE FROM faces WHERE sha1 = ? AND pipeline = ?", (sha1, pipeline))
    db.execute(
        "INSERT OR REPLACE INTO photos (sha1, pipeline, width, height, scanned_at) VALUES (?,?,?,?,?)",
        (sha1, pipeline, size[0], size[1], time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    db.executemany(
        "INSERT INTO faces (sha1, pipeline, idx, x, y, w, h, det_score, emb) VALUES (?,?,?,?,?,?,?,?,?)",
        [(sha1, pipeline, i, *box, score, emb.tobytes()) for i, (box, score, emb) in enumerate(faces)],
    )
    db.commit()


def cached_photo(db: sqlite3.Connection, sha1: str, pipeline: str) -> bool:
    row = db.execute("SELECT 1 FROM photos WHERE sha1 = ? AND pipeline = ?", (sha1, pipeline)).fetchone()
    return row is not None


def pipelines(db: sqlite3.Connection) -> list[str]:
    """Every pipeline present in the cache, newest scan first."""
    rows = db.execute("SELECT pipeline, MAX(scanned_at) AS t FROM photos GROUP BY pipeline ORDER BY t DESC").fetchall()
    return [r[0] for r in rows]


def load_faces(
    db: sqlite3.Connection, sha1: str, pipeline: str
) -> list[tuple[tuple[float, float, float, float], float, np.ndarray]]:
    rows = db.execute(
        "SELECT x, y, w, h, det_score, emb FROM faces WHERE sha1 = ? AND pipeline = ? ORDER BY idx",
        (sha1, pipeline),
    )
    return [((r[0], r[1], r[2], r[3]), r[4], np.frombuffer(r[5], dtype=np.float32)) for r in rows]


def gallery(db: sqlite3.Connection, person: str, pipeline: str) -> np.ndarray:
    rows = db.execute("SELECT emb FROM refs WHERE person = ? AND pipeline = ?", (person, pipeline)).fetchall()
    if not rows:
        raise SystemExit(f"no references enrolled for {person!r} on pipeline {pipeline!r} — enroll first")
    return np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])


def photo_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PHOTO_EXTS)


def album_dirs(root: Path) -> list[Path]:
    """Directories under ``root`` that hold photos themselves."""
    return sorted({p.parent for p in photo_files(root) if p.parent != root})


# --- steps ------------------------------------------------------------------


def enroll(db: sqlite3.Connection, person: str, ref_dir: Path, det, rec, crop_dir: Path) -> list[str]:
    """Embed the photos in ``ref_dir`` as references for ``person``.

    Single-face photos are enrolled first and unambiguously. A photo with several
    faces then contributes the candidate that best matches what is already
    enrolled — picking "the largest face" instead grabs whichever classmate is
    closest to the camera, which is a wrong reference the scoring then has to
    live with. Returns human-readable warnings.
    """
    import cv2

    files = photo_files(ref_dir)
    if not files:
        raise SystemExit(f"no reference photos in {ref_dir}/ — add 4-6 face photos of {person!r}")
    pipeline = pipeline_id(det)
    crops = crop_dir / person / det.name
    if crops.is_dir():
        shutil.rmtree(crops)  # derived crops: rebuilt from scratch so no stale index files remain
    crops.mkdir(parents=True, exist_ok=True)
    db.execute("DELETE FROM refs WHERE person = ? AND pipeline = ?", (person, pipeline))
    warnings = []

    scanned = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            warnings.append(f"{path.name}: unreadable")
            continue
        faces = detect(img, det, 16)
        if not faces:
            warnings.append(f"{path.name}: no face found — skipped")
            continue
        scanned.append((path, img, faces))
    if not scanned:
        raise SystemExit(f"could not enroll anything from {ref_dir}/")
    # unambiguous photos first: they seed the gallery the ambiguous ones match against
    scanned.sort(key=lambda item: len(item[2]))

    embs, sources = [], []
    for path, img, faces in scanned:
        chosen, confidence = _pick_reference(faces, embs, img, rec)
        box, kps = chosen[0], chosen[1]
        if box[2] < 64:
            warnings.append(f"{path.name}: face only {box[2]:.0f}px wide — a bigger photo would help")
        if len(faces) > 1:
            if confidence is None:
                warnings.append(f"{path.name}: {len(faces)} faces, no gallery yet — kept the largest ({box[2]:.0f}px)")
            else:
                warnings.append(
                    f"{path.name}: {len(faces)} faces, kept the one matching the gallery ({confidence:.2f}, {box[2]:.0f}px)"
                )
                if confidence < 0.35:
                    warnings.append(f"{path.name}: no face there resembles the others ({confidence:.2f}) — skipped")
                    continue
        crop_path = crops / f"{len(embs):02d}_{path.stem}.jpg"
        cv2.imwrite(str(crop_path), align(img, kps))
        embs.append(embed(img, kps, rec))
        sources.append((str(path), str(crop_path)))
    if not embs:
        raise SystemExit(f"could not enroll anything from {ref_dir}/")
    for (source, crop), emb in zip(sources, embs, strict=True):
        db.execute(
            "INSERT INTO refs (person, pipeline, source, crop, emb) VALUES (?,?,?,?,?)",
            (person, pipeline, source, crop, emb.tobytes()),
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


def _pick_reference(faces, embs: list[np.ndarray], img, rec):
    """Which face of a reference photo to enrol, and how sure we are.

    One face is unambiguous; several are scored against the references gathered
    so far (``None`` confidence while the gallery is still empty, where the
    largest face is the only signal available).
    """
    if len(faces) == 1:
        return faces[0], None
    if not embs:
        return max(faces, key=lambda f: f[0][2]), None
    gallery = np.stack(embs)
    scores = [float((gallery @ embed(img, kps, rec)).max()) for _box, kps, _score in faces]
    best = int(np.argmax(scores))
    return faces[best], scores[best]


def scan(db: sqlite3.Connection, roots: list[Path], det, rec, floor: float = DEFAULT_FLOOR) -> tuple[int, int]:
    """Embed every uncached photo under ``roots``; returns (embedded, already cached)."""
    import cv2

    pipeline = pipeline_id(det, floor)
    done = skipped = vanished = 0
    for root in roots:
        for path in photo_files(root):
            try:
                sha1 = file_sha1(path)
            except OSError:
                vanished += 1  # renamed or deleted while scanning (e.g. a sync run)
                continue
            if cached_photo(db, sha1, pipeline):
                skipped += 1
                continue
            img = cv2.imread(str(path))
            if img is None:
                print(f"   unreadable: {path}")
                continue
            faces = [(box, score, embed(img, kps, rec)) for box, kps, score in detect(img, det, floor)]
            store_photo(db, sha1, (img.shape[1], img.shape[0]), pipeline, faces)
            done += 1
            print(f"   {path.parent.name}/{path.name}: {len(faces)} face(s) >= {floor:.0f}px")
    if vanished:
        print(f"   {vanished} photo(s) disappeared during the scan; re-run to pick them up")
    return done, skipped


def collect(
    db: sqlite3.Connection,
    person: str,
    roots: list[Path],
    min_face: float = DEFAULT_MIN_FACE,
    pipeline: str | None = None,
) -> list[dict]:
    """Per-photo rows: best score, best face box, face count above ``min_face``."""
    pipeline = pipeline or pipelines(db)[0]
    gal = gallery(db, person, pipeline)
    rows = []
    for root in roots:
        for album in album_dirs(root):
            for path in photo_files(album):
                faces = [f for f in load_faces(db, file_sha1(path), pipeline) if f[0][2] >= min_face]
                score, idx = score_faces(faces, gal)
                best_box = faces[idx][0] if idx >= 0 else None
                rows.append(
                    {
                        "album": album.name,
                        "pipeline": pipeline,
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


# --- detector comparison ----------------------------------------------------


def detector_of(pipeline: str) -> str:
    """``v2:scrfd:w600k_r50.onnx:arc112:floor20`` -> ``scrfd``."""
    parts = pipeline.split(":")
    return parts[1] if len(parts) > 1 else pipeline


def compare(rows_by_pipeline: dict[str, list[dict]], threshold: float) -> dict:
    """Agreement summary between pipelines over the same photos.

    ``rows_by_pipeline`` maps pipeline id -> ``collect()`` rows. Photos are
    matched by filename within their album, so both pipelines must have scanned
    the same roots.
    """
    keys = list(rows_by_pipeline)
    if len(keys) < 2:
        raise SystemExit(f"need two pipelines to compare, found {len(keys)}: {keys}")
    a, b = keys[0], keys[1]
    index = {k: {(r["album"], r["path"].name): r for r in rows_by_pipeline[k]} for k in keys}
    shared = sorted(set(index[a]) & set(index[b]))
    both = only_a = only_b = neither = 0
    deltas = []
    for key in shared:
        ra, rb = index[a][key], index[b][key]
        ha, hb = ra["faces"] and ra["score"] >= threshold, rb["faces"] and rb["score"] >= threshold
        if ha and hb:
            both += 1
            deltas.append(rb["score"] - ra["score"])
        elif ha:
            only_a += 1
        elif hb:
            only_b += 1
        else:
            neither += 1
    return {
        "pipelines": (a, b),
        "shared": len(shared),
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "hit_a": both + only_a,
        "hit_b": both + only_b,
        "union": both + only_a + only_b,
        "agree": both + neither,
        "deltas": np.array(deltas) if deltas else np.array([]),
    }


def disagreements(rows_by_pipeline: dict[str, list[dict]], threshold: float, limit: int = 20) -> list[dict]:
    """Photos the two pipelines classify differently, most face-width change first."""
    a, b = list(rows_by_pipeline)[:2]
    index = {k: {(r["album"], r["path"].name): r for r in rows_by_pipeline[k]} for k in (a, b)}
    out = []
    for key in sorted(set(index[a]) & set(index[b])):
        ra, rb = index[a][key], index[b][key]
        ha, hb = ra["faces"] and ra["score"] >= threshold, rb["faces"] and rb["score"] >= threshold
        if ha == hb:
            continue
        out.append(
            {
                "album": ra["album"],
                "name": ra["path"].name,
                "a": ra,
                "b": rb,
                "winner": a if ha else b,
                "width_delta": rb["face_width"] - ra["face_width"],
            }
        )
    out.sort(key=lambda d: -abs(d["width_delta"]))
    return out[:limit]


def compare_report(stats: dict, found: list[dict], threshold: float) -> str:
    """Terminal summary of a detector comparison: a 2x2 table plus score deltas.

    The four counts are disjoint and add up to the shared photos; a detector's
    own total is ``both + only that detector``, and ``union`` is photos kept by
    either one.
    """
    a, b = stats["pipelines"]
    da, db = detector_of(a), detector_of(b)
    shared = stats["shared"]
    lines = [
        f"{da} vs {db}  (threshold {threshold:.2f}, {shared} shared photos)",
        "",
        f"  {'':12s}{db + ' hit':>12s}{db + ' miss':>13s}{'total':>9s}",
        f"  {da + ' hit':12s}{stats['both']:12d}{stats['only_a']:13d}{stats['hit_a']:9d}",
        f"  {da + ' miss':12s}{stats['only_b']:12d}{stats['neither']:13d}{shared - stats['hit_a']:9d}",
        f"  {'total':12s}{stats['hit_b']:12d}{shared - stats['hit_b']:13d}{shared:9d}",
        "",
        f"  both {stats['both']}  ·  only {da} {stats['only_a']}  ·  only {db} {stats['only_b']}  "
        f"·  neither {stats['neither']}  ·  either {stats['union']}",
        f"  agree {stats['agree']}/{shared} ({stats['agree'] / (shared or 1) * 100:.0f}%)",
    ]
    if stats["deltas"].size:
        d = stats["deltas"]
        lines.append(
            f"  score delta ({db} - {da}) on shared hits: "
            f"median {np.median(d):+.3f}, mean {d.mean():+.3f}, p10 {np.percentile(d, 10):+.3f}, "
            f"p90 {np.percentile(d, 90):+.3f}"
        )
    for item in found:
        lines.append(
            f"  {detector_of(item['winner']):6s} only  {item['album']}/{item['name']}  "
            f"{da}={item['a']['score']:.2f}/{item['a']['faces']}f  "
            f"{db}={item['b']['score']:.2f}/{item['b']['faces']}f"
        )
    return "\n".join(lines)


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
    query = "SELECT source, crop FROM refs WHERE person = ? AND pipeline = ? ORDER BY source"
    for source, crop in db.execute(query, (person, rows[0]["pipeline"] if rows else "")):
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
