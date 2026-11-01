"""Face detection + recognition for Codmon albums.

Reference photos of your child live in ``refs/<person>/face/*.jpg`` (the only
human-provided input). Everything derived — face boxes, embeddings, scores —
lives in ``.cache/faces.db`` (sqlite), and the ONNX models in ``.cache/models``.

Two-stage pipeline, both plain ONNX/OpenCV:

1. **detect** — a detector finds faces and 5 landmarks. Two are available and
   interchangeable: ``scrfd`` (``det_10g``, the detector inside InsightFace's
   buffalo_l — the default, because its landmarks stay accurate on faces that
   are tilted or turned) and ``yunet`` (OpenCV, finds more of the tiny faces a
   500px CDN photo leaves behind, but misplaces landmarks on some of them).
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

import hashlib
import html
import json
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
DEFAULT_THRESHOLD = 0.55  # measured: below this, other children score as matches
BANDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
# evaluate walks a wider ladder than the report sweep: the sweep is about picking
# one of a few candidates, this is about seeing where the two populations meet.
EVALUATE_EDGES = tuple(round(0.20 + 0.05 * i, 2) for i in range(16))  # 0.20 .. 0.95

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
# Members come from one shared archive; nothing outside models_dir is read.
YUNET = "yunet"
SCRFD = "scrfd"
DETECTORS = (SCRFD, YUNET)
DEFAULT_DETECTOR = SCRFD

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
SCHEMA_VERSION = 4
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
    """Path to a verified model file inside ``models_dir``.

    Models live only here — nothing outside the project is read or trusted.
    ``yunet`` is a single file; ``arcface`` and ``scrfd`` are members of the
    buffalo_l archive and are extracted together, so asking for either one only
    downloads the zip once.
    """
    url, want = MODELS[name]
    dest = Path(models_dir) / MODEL_FILES[name]
    if dest.exists() and sha256_file(dest) == want:
        return dest
    if name in BUFFALO_MEMBERS:
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

    DEFAULT_SCORE = 0.6

    def __init__(self, model_path: Path, score: float = DEFAULT_SCORE, nms: float = 0.3, top_k: int = 5000):
        import cv2

        self.score = score
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
    DEFAULT_SCORE = 0.5

    def __init__(self, model_path: Path, input_size: int = 640, score: float = DEFAULT_SCORE, nms_thresh: float = 0.4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self._sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self._outputs = [o.name for o in self._sess.get_outputs()]
        self._size = (input_size, input_size)
        self.score, self._nms = score, nms_thresh
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
            pos = np.where(score >= self.score)[0]
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


def make_detector(name: str, model_path: Path, score: float | None = None):
    """Detector instance for ``name``, optionally with a different det score."""
    cls = {SCRFD: ScrfdDetector, YUNET: YunetDetector}.get(name)
    if cls is None:
        raise SystemExit(f"unknown detector {name!r} (choose from {', '.join(DETECTORS)})")
    return cls(model_path) if score is None else cls(model_path, score=score)


def make_recognizer(model_path: Path):
    """ArcFace session; returns (session, input name)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def load_models(detector: str = DEFAULT_DETECTOR, models_dir: Path = DEFAULT_MODELS, det_score: float | None = None):
    """(detector, recognizer) for ``detector``; downloads/verifies both models."""
    det = make_detector(detector, ensure_model(detector, models_dir), det_score)
    return det, make_recognizer(ensure_model("arcface", models_dir))


# --- per-image work ---------------------------------------------------------


def pipeline_id(det, floor: float = DEFAULT_FLOOR) -> str:
    """Cache key for one detection+recognition setup.

    Everything that changes the stored numbers is in here, so swapping a
    detector stores new rows instead of silently reusing the old vectors.
    """
    score = getattr(det, "score", None)
    score_tag = "?" if score is None else f"{score:g}"
    return f"{PIPELINE_VERSION}:{det.name}:s{score_tag}:{ARC_MEMBER}:arc112:floor{floor:g}"


def pipeline_for(db: sqlite3.Connection, detector: str, floor: float = DEFAULT_FLOOR) -> str:
    """Cached pipeline id for ``detector``, or the id a scan with it would use."""
    for known in pipelines(db):
        if detector_of(known) == detector:
            return known
    return f"{PIPELINE_VERSION}:{detector}:s?:{ARC_MEMBER}:arc112:floor{floor:g}"


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

    The gallery is seeded by photos that contain exactly one face; a photo with
    several faces then contributes the candidate that best matches what is
    already enrolled. Nothing is guessed: a multi-face photo is skipped until an
    unambiguous one has been enrolled, and it is skipped again when no candidate
    resembles the gallery. Returns human-readable warnings.
    """
    import cv2

    files = photo_files(ref_dir)
    if not files:
        raise SystemExit(f"no reference photos in {ref_dir}/ — add 4-6 face photos of {person!r}")
    pipeline = pipeline_id(det)
    crops = crop_dir / person / det.name
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

    pending = []  # (source, crop, embedding) — written only once enrollment succeeds
    embs = []
    for path, img, faces in scanned:
        if len(faces) > 1 and not embs:
            warnings.append(f"{path.name}: {len(faces)} faces and no single-face photo yet — skipped")
            continue
        chosen, confidence = _pick_reference(faces, embs, img, rec)
        box, kps = chosen[0], chosen[1]
        if box[2] < 64:
            warnings.append(f"{path.name}: face only {box[2]:.0f}px wide — a bigger photo would help")
        if confidence is not None:
            warnings.append(
                f"{path.name}: {len(faces)} faces, kept the one matching the gallery ({confidence:.2f}, {box[2]:.0f}px)"
            )
            if confidence < 0.35:
                warnings.append(f"{path.name}: no face there resembles the others ({confidence:.2f}) — skipped")
                continue
        crop = align(img, kps)
        emb = embed(img, kps, rec)
        pending.append((str(path), crop, emb))
        embs.append(emb)
    if not pending:
        raise SystemExit(f"could not enroll anything from {ref_dir}/ — add at least one photo with only her face in it")

    # nothing above touched the existing gallery: only now is it replaced
    if crops.is_dir():
        shutil.rmtree(crops)  # derived crops: rebuilt so no stale index files remain
    crops.mkdir(parents=True, exist_ok=True)
    db.execute("DELETE FROM refs WHERE person = ? AND pipeline = ?", (person, pipeline))
    sources = []
    for index, (source, crop, emb) in enumerate(pending):
        crop_path = crops / f"{index:02d}_{Path(source).stem}.jpg"
        cv2.imwrite(str(crop_path), crop)
        db.execute(
            "INSERT INTO refs (person, pipeline, source, crop, emb) VALUES (?,?,?,?,?)",
            (person, pipeline, source, str(crop_path), emb.tobytes()),
        )
        sources.append((source, str(crop_path)))
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
    """Which face of a reference photo to enrol, and how well it matches.

    A single face is unambiguous (``confidence`` is ``None``); several faces are
    scored against the references gathered so far, and the caller needs a
    non-empty gallery — it never guesses from face size.
    """
    if len(faces) == 1:
        return faces[0], None
    if not embs:
        raise ValueError("_pick_reference needs a seeded gallery to choose between faces")
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
    overrides: dict[str, str] | None = None,
) -> list[dict]:
    """Per-photo rows: best score, best face box, face count above ``min_face``.

    ``overrides`` maps a photo's path relative to its root (``album/file.JPG``) to
    ``keep``/``drop``, which is what a hand decision reads like in the json file.
    """
    pipeline = pipeline or pipelines(db)[0]
    overrides = overrides or {}
    gal = gallery(db, person, pipeline)
    rows = []
    for root in roots:
        for album in album_dirs(root):
            for path in photo_files(album):
                sha1 = file_sha1(path)
                faces = [f for f in load_faces(db, sha1, pipeline) if f[0][2] >= min_face]
                score, idx = score_faces(faces, gal)
                best_box = faces[idx][0] if idx >= 0 else None
                size = db.execute(
                    "SELECT width, height FROM photos WHERE sha1 = ? AND pipeline = ?", (sha1, pipeline)
                ).fetchone() or (0, 0)
                rel = str(path.relative_to(root))
                rows.append(
                    {
                        "album": album.name,
                        "pipeline": pipeline,
                        "sha1": sha1,
                        "rel": rel,
                        "override": overrides.get(rel),
                        "path": path,
                        "root": root,
                        "score": score,
                        "faces": len(faces),
                        "face_width": best_box[2] if best_box else 0.0,
                        "best_box": best_box,
                        "width": size[0],
                        "height": size[1],
                    }
                )
    return rows


# --- overrides --------------------------------------------------------------

OVERRIDE_ACTIONS = ("keep", "drop")


def overrides_path(refs_root: Path, person: str) -> Path:
    """Where hand decisions live: ``refs/<person>/overrides.json``."""
    return Path(refs_root) / person / "overrides.json"


def load_overrides(path: Path) -> dict[str, str]:
    """``album/file.JPG`` -> ``keep``/``drop``; a missing file means no decisions."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} should hold an object of album/file.JPG -> keep|drop")
    return {str(k): v for k, v in data.items() if v in OVERRIDE_ACTIONS}


def save_overrides(path: Path, decisions: dict[str, str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    body = json.dumps(dict(sorted(decisions.items())), indent=2, ensure_ascii=False)
    tmp.write_text(body + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def apply_decision(path: Path, key: str, action: str) -> dict[str, str]:
    """Record ``keep``/``drop``/``clear`` for ``album/file.JPG`` and persist it."""
    if action not in (*OVERRIDE_ACTIONS, "clear"):
        raise ValueError(f"unknown action {action!r} (keep, drop or clear)")
    decisions = load_overrides(path)
    if action == "clear":
        decisions.pop(key, None)
    else:
        decisions[key] = action
    save_overrides(path, decisions)
    return decisions


def evaluate(rows: list[dict], edges=EVALUATE_EDGES) -> list[dict]:
    """Per-threshold view of the hand decisions: would the score agree with you?

    ``keeps_matched`` counts photos you marked ``keep`` that the score would have
    found on its own; the rest need the pin. ``drops_matched`` counts false
    positives the score would have admitted — those need the pin to stay out.
    Reviews are the ground truth, so unreviewed photos are simply not counted.
    """
    keeps = [r for r in rows if r.get("override") == "keep"]
    drops = [r for r in rows if r.get("override") == "drop"]
    return [
        {
            "threshold": t,
            "keeps": len(keeps),
            "keeps_matched": sum(1 for r in keeps if r["score"] >= t),
            "drops": len(drops),
            "drops_matched": sum(1 for r in drops if r["score"] >= t),
        }
        for t in edges
    ]


EVALUATE_LEGEND = """\
  keeps found     keep photos the score would have accepted on its own
  keeps pinned    keep photos below the threshold - only your keep decision saves them
  drops rejected  drop photos below the threshold - the score already excludes them
  drops leaked    drop photos above the threshold - only your drop keeps them out"""


def evaluate_report(rows: list[dict], edges=EVALUATE_EDGES) -> str:
    """Text table over ``evaluate`` results, plus what is still unreviewed."""
    table = evaluate(rows, edges)
    keeps = sum(1 for r in rows if r.get("override") == "keep")
    drops = sum(1 for r in rows if r.get("override") == "drop")
    lines = [
        f"{keeps + drops} reviewed of {len(rows)} photos ({keeps} keep, {drops} drop)",
        "",
        EVALUATE_LEGEND,
        "",
        f"  {'threshold':>9} {'keeps found':>12} {'keeps pinned':>13} {'drops rejected':>15} {'drops leaked':>13}",
    ]
    for row in table:
        lines.append(
            f"  {row['threshold']:9.2f} {row['keeps_matched']:12d} {row['keeps'] - row['keeps_matched']:13d} "
            f"{row['drops'] - row['drops_matched']:15d} {row['drops_matched']:13d}"
        )
    lines.append("")
    lines.append("  a good threshold keeps drops leaked at 0 without pinning keeps that already score above it")
    return "\n".join(lines)


def unused_overrides(overrides: dict[str, str], rows: list[dict]) -> list[str]:
    """Decision keys that match no photo on disk (renamed album, typo)."""
    seen = {r["rel"] for r in rows}
    return sorted(key for key in overrides if key not in seen)


# --- decisions --------------------------------------------------------------


def sweep(rows: list[dict], edges=BANDS) -> list[tuple[float, int, float]]:
    """(threshold, photos kept, share of all photos kept) per candidate threshold."""
    total = len(rows) or 1
    scores = [r["score"] for r in rows if r["faces"]]
    kept = lambda t: sum(1 for s in scores if s >= t)  # noqa: E731 - sweep stays a one-liner
    return [(t, kept(t), kept(t) / total * 100) for t in edges]


def matched_rows(rows: list[dict], threshold: float) -> list[dict]:
    """Photos that end up in the export: scored matches plus hand-kept ones."""
    return [r for r in rows if verdict(r, threshold) in ("matched", "included")]


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


def dispute_kind(a: dict, b: dict, noise: float = 0.05) -> str:
    """Why two detectors disagree on a photo.

    ``detection`` — they found a different number of faces (one may have found
    none at all): real evidence about detection. ``boundary`` — same faces, and
    the scores differ by less than ``noise``: the photo just sits on the cutoff.
    ``score`` — same faces, but the similarity moved more than that.
    """
    if a["faces"] != b["faces"]:
        return "detection"
    return "boundary" if abs(b["score"] - a["score"]) <= noise else "score"


def disagreements(rows_by_pipeline: dict[str, list[dict]], threshold: float, limit: int = 0) -> list[dict]:
    """Photos the two pipelines classify differently, split by reason."""
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
                "kind": dispute_kind(ra, rb),
            }
        )
    out.sort(key=lambda d: -abs(d["width_delta"]))
    return out[:limit] if limit else out


KIND_ORDER = ("detection", "score", "boundary")
KIND_TITLES = {
    "detection": "Detection differences — the detectors saw a different number of faces",
    "score": "Score gaps — same faces, similarity moved",
    "boundary": "Boundary noise — same faces, scores within 0.05 of each other",
}


def comparison_groups(rows_by_pipeline: dict[str, list[dict]], threshold: float, sample: int = 20, seed: int = 0):
    """Disputes by kind (complete), plus random samples of the agreed buckets."""
    import random

    a, b = list(rows_by_pipeline)[:2]
    index = {k: {(r["album"], r["path"].name): r for r in rows_by_pipeline[k]} for k in (a, b)}
    groups = {kind: [] for kind in KIND_ORDER}
    both, neither = [], []
    for key in sorted(set(index[a]) & set(index[b])):
        ra, rb = index[a][key], index[b][key]
        ha, hb = ra["faces"] and ra["score"] >= threshold, rb["faces"] and rb["score"] >= threshold
        if ha and hb:
            both.append((key, ra, rb))
        elif not ha and not hb:
            neither.append((key, ra, rb))
        else:
            groups[dispute_kind(ra, rb)].append((key, ra, rb))
    rng = random.Random(seed)
    sampled = {
        "both": rng.sample(both, min(sample, len(both))),
        "neither": rng.sample(neither, min(sample, len(neither))),
    }
    return groups, sampled, {"both": len(both), "neither": len(neither)}


COMPARE_CSS = """
body{font:14px/1.45 system-ui,sans-serif;background:#111;color:#ddd;margin:24px}
h1{font-size:18px} h2{font-size:15px;margin-top:26px;border-bottom:1px solid #333;padding-bottom:4px}
h3{font-size:13px;color:#999;font-weight:600;margin:18px 0 6px}
table{border-collapse:collapse;margin:6px 0} td,th{padding:2px 10px;border-bottom:1px solid #262626;text-align:left}
.card{background:#1a1a1a;border-radius:6px;padding:8px;margin:10px 0}
.panes{display:flex;gap:8px}
.pane{flex:1;position:relative}
.pane img{width:100%;border-radius:4px;display:block}
.pane .tag{font:600 11px ui-monospace,monospace;color:#888;margin-bottom:3px}
.box{position:absolute;border:2px solid #666;border-radius:1px}
.pane-a .box{border-color:#3fb6d3} .pane-b .box{border-color:#e08a2b}
.box.best{border-width:3px;box-shadow:0 0 0 1px #0006}
.row{display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap}
.badge{font:600 11px ui-monospace,monospace;padding:2px 6px;border-radius:3px;background:#2a2a2a;color:#bbb}
.badge.hit{background:#1f3d24;color:#7ddc89} .badge.miss{background:#3d1f1f;color:#e07a7a}
.kind{font:600 11px ui-monospace,monospace;color:#999}
.muted{color:#777} .path{font:11px ui-monospace,monospace;color:#888;word-break:break-all}
details{margin-top:14px} summary{cursor:pointer;color:#999}
"""


def write_comparison(
    db: sqlite3.Connection,
    rows_by_pipeline: dict[str, list[dict]],
    stats: dict,
    out: Path,
    threshold: float,
    min_face: float = DEFAULT_MIN_FACE,
    sample: int = 20,
) -> Path:
    """Side-by-side sheet: the same photo rendered once per detector, boxes drawn.

    Detection differences come first — those are the only rows that say anything
    about which detector sees better. Boundary noise and the agreed buckets are
    folded into <details> so they are available without drowning the page.
    """
    a, b = stats["pipelines"]
    da, db_name = detector_of(a), detector_of(b)
    groups, sampled, totals = comparison_groups(rows_by_pipeline, threshold, sample)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        f"<h1>{da} vs {db_name} — threshold {threshold:.2f}, min face {min_face:.0f}px</h1>",
        f"<table><tr><th></th><th>{db_name} hit</th><th>{db_name} miss</th><th>total</th></tr>",
        f"<tr><th>{da} hit</th><td>{stats['both']}</td><td>{stats['only_a']}</td><td>{stats['hit_a']}</td></tr>",
        f"<tr><th>{da} miss</th><td>{stats['only_b']}</td><td>{stats['neither']}</td>"
        f"<td>{stats['shared'] - stats['hit_a']}</td></tr>",
        f"<tr><th>total</th><td>{stats['hit_b']}</td><td>{stats['shared'] - stats['hit_b']}</td>"
        f"<td>{stats['shared']}</td></tr></table>",
        f"<p>both {stats['both']} · only {da} {stats['only_a']} · only {db_name} {stats['only_b']} · "
        f"neither {stats['neither']} · either {stats['union']} · "
        f"agree {stats['agree']}/{stats['shared']} ({stats['agree'] / (stats['shared'] or 1) * 100:.0f}%)</p>",
        f"<p class='muted'>{len(groups['detection'])} detection differences · "
        f"{len(groups['score'])} score gaps · {len(groups['boundary'])} boundary noise</p>",
    ]

    def render(key, ra, rb, kind):  # noqa: ANN001
        album, name = key
        parts_html = [f'<div class="card"><div class="path">{album}/{name}</div><div class="panes">']
        for cls, pipeline, row in (("pane-a", a, ra), ("pane-b", b, rb)):
            hit = bool(row["faces"]) and row["score"] >= threshold
            detail = (
                f'<div class="row"><span class="badge {"hit" if hit else "miss"}">'
                f"{'hit' if hit else 'miss'}</span><span>{row['score']:.3f}</span>"
                f'<span class="muted">{row["faces"]} face(s)</span></div>'
            )
            faces = [f for f in load_faces(db, file_sha1(row["path"]), pipeline) if f[0][2] >= min_face]
            boxes = "".join(
                f'<span class="box{(" best" if row["best_box"] and box == row["best_box"] else "")}" '
                f'style="{face_box_style(box, row.get("width", 0), row.get("height", 0))}"></span>'
                for box, _s, _e in faces
            )
            pane = (
                f'<div class="{cls}"><div class="tag">{detector_of(pipeline)}</div>'
                f'<span class="frame"><img loading="lazy" src="{image_src(row["path"], out)}">{boxes}</span>'
                f"{detail}</div>"
            )
            parts_html.append(pane)
        parts_html.append("</div>")
        if kind:
            parts_html.append(f'<div class="row"><span class="kind">{kind}</span></div>')
        parts_html.append("</div>")
        return "".join(parts_html)

    for kind in KIND_ORDER:
        rows = groups[kind]
        if not rows:
            continue
        body = "\n".join(render(key, ra, rb, kind) for key, ra, rb in rows)
        if kind == "boundary":
            parts.append(f"<details><summary>{KIND_TITLES[kind]} ({len(rows)})</summary>{body}</details>")
        else:
            parts.append(f"<h2>{KIND_TITLES[kind]} ({len(rows)})</h2>")
            parts.append(body)

    for bucket, title in (("both", "Both matched"), ("neither", "Neither matched")):
        body = "\n".join(render(key, ra, rb, "") for key, ra, rb in sampled[bucket])
        parts.append(
            f"<details><summary>{title} — sample {len(sampled[bucket])} of {totals[bucket]}</summary>{body}</details>"
        )

    html_text = (
        "<!doctype html><meta charset=utf-8><title>detector comparison</title>"
        f"<style>{COMPARE_CSS}{REPORT_CSS}</style>\n" + "\n".join(parts)
    )
    out.write_text(html_text, encoding="utf-8")
    return out


def compare_report(stats: dict, found: list[dict], threshold: float, list_noise: bool = False) -> str:
    """Terminal summary: the 2x2 table, then disputes grouped by reason."""
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
    by_kind = {kind: [d for d in found if d["kind"] == kind] for kind in KIND_ORDER}
    lines.append(
        f"  {len(by_kind['detection'])} detection differences · {len(by_kind['score'])} score gaps · "
        f"{len(by_kind['boundary'])} boundary noise (|delta| <= 0.05)"
    )
    for kind in KIND_ORDER:
        rows = by_kind[kind]
        if not rows or (kind == "boundary" and not list_noise):
            continue
        lines.append(f"  -- {kind} --")
        for item in rows:
            who = detector_of(item["winner"])
            lines.append(
                f"  {who:6s} only  {item['album']}/{item['name']}  "
                f"{da}={item['a']['score']:.2f}/{item['a']['faces']}f  "
                f"{db}={item['b']['score']:.2f}/{item['b']['faces']}f"
            )
    return "\n".join(lines)


# --- review report ----------------------------------------------------------


REPORT_CSS = """
body{font:14px/1.45 system-ui,sans-serif;background:#111;color:#ddd;margin:24px}
h1{font-size:18px} h2{font-size:15px;margin-top:26px;border-bottom:1px solid #333;padding-bottom:4px}
table{border-collapse:collapse;margin:6px 0} td,th{padding:2px 10px;border-bottom:1px solid #262626;text-align:left}
.card{display:inline-block;vertical-align:top;width:440px;margin:6px;background:#1a1a1a;border-radius:6px;
      padding:8px;border-left:5px solid #333}
.card.matched{border-left-color:#5ec26a} .card.unknown{border-left-color:#f0a020}
.card.included{border-left-color:#6aa9e0} .card.excluded{border-left-color:#c05c5c}
.card.ref{width:140px;border-left-color:#444}
.frame{position:relative;display:block}
.frame img{width:100%;border-radius:4px;display:block;background:#000}
.box{position:absolute;border:2px solid #5ec26a;border-radius:1px;box-shadow:0 0 0 1px #0006}
.row{display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap}
.badge{font:700 13px ui-monospace,monospace;padding:2px 7px;border-radius:3px;background:#2a2a2a;color:#bbb}
.badge.ok{background:#1f3d24;color:#7ddc89} .badge.warn{background:#3d3018;color:#f0a020}
.badge.add{background:#1c3550;color:#8ec7f0} .badge.drop{background:#3d1f1f;color:#e07a7a}
.actions{display:flex;gap:6px;margin-top:6px}
.actions button{font:600 12px system-ui,sans-serif;background:#262626;color:#ccc;border:1px solid #333;
                border-radius:4px;padding:3px 9px;cursor:pointer}
.actions button:hover{background:#333}
.score{font-weight:600;font-size:14px;font-variant-numeric:tabular-nums} .muted{color:#777}
details.album{margin:10px 0}
details.album>summary{cursor:pointer;position:sticky;top:0;z-index:2;background:#111;color:#ddd;
                      font-size:15px;font-weight:600;padding:8px 0;border-bottom:1px solid #333}
details.album>summary .muted{font-weight:400;font-size:13px}
@media print{details{display:block} details>summary{position:static}}
"""

BADGES = {
    "matched": ("ok", "MATCH"),
    "below": ("", "below"),
    "unknown": ("warn", "no face"),
    "included": ("add", "ADDED"),
    "excluded": ("drop", "DROPPED"),
}
NEAR_MISS = 0.10  # "below but close": within this much under the threshold


def verdict(row: dict, threshold: float) -> str:
    """One of matched / below / unknown / included / excluded.

    A hand decision wins over the score in both directions: ``drop`` removes a
    photo even when it scored well, ``keep`` adds one that scored badly or had no
    detectable face at all.
    """
    override = row.get("override")
    if override == "drop":
        return "excluded"
    if override == "keep":
        return "included"
    if not row["faces"]:
        return "unknown"
    return "matched" if row["score"] >= threshold else "below"


def image_src(path: Path, out: Path) -> str:
    """Relative URL from the sheet to the album photo, so nothing is embedded.

    This is a URL, not a filesystem path, so it is percent-encoded: album names
    are UTF-8, and a few CDN filenames carry literal ``%28``-style escapes that
    must become ``%2528`` or the browser decodes them into ``(`` and misses.
    ``urllib.parse.unquote`` turns the result back into a real path.
    """
    import urllib.parse

    return html.escape(urllib.parse.quote(os.path.relpath(path, out.parent), safe="/"), quote=True)


def face_box_style(box, width: int, height: int) -> str:
    """CSS percentages placing a face box over the photo at any display size."""
    if not box or not width or not height:
        return ""
    x, y, w, h = box
    return (
        f"left:{x / width * 100:.2f}%;top:{y / height * 100:.2f}%;"
        f"width:{w / width * 100:.2f}%;height:{h / height * 100:.2f}%"
    )


def write_report(
    db: sqlite3.Connection,
    person: str,
    rows: list[dict],
    out: Path,
    threshold: float,
    min_face: float,
    interactive: bool = False,
    src_for=None,
) -> Path:
    """Write ``render_report`` output to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    body = render_report(db, person, rows, out, threshold, min_face, interactive, src_for=src_for)
    out.write_text(body, encoding="utf-8")
    return out


def render_report(
    db: sqlite3.Connection,
    person: str,
    rows: list[dict],
    out: Path,
    threshold: float,
    min_face: float,
    interactive: bool = False,
    subtitle: str = "",
    src_for=None,
) -> str:
    """HTML sheet with one collapsible section per album.

    The verdict is carried by the card's left border (green matched, orange no
    face, grey below) so a fast scroll reads as a colour pattern; the badge spells
    it out for screenshots and colour-blind viewers. Photos are linked relatively
    (nothing embedded), so re-running at another threshold is instant.
    """
    heading = f"Scores for {person!r} — threshold {threshold:.2f}, min face {min_face:.0f}px"
    if subtitle:
        heading += f" <span class='muted' style='font-size:13px;font-weight:400'>{html.escape(subtitle)}</span>"
    parts = [f"<h1>{heading}</h1>"]

    def src(path: Path) -> str:
        return src_for(path) if src_for else image_src(path, out)

    parts.append("<h2>Enrolled references (verify these are all her)</h2><div>")
    query = "SELECT source, crop FROM refs WHERE person = ? AND pipeline = ? ORDER BY source"
    pipeline = rows[0]["pipeline"] if rows else ""
    for source, crop in db.execute(query, (person, pipeline)):
        parts.append(
            f'<span class="card ref"><img loading="lazy" src="{src(Path(crop))}" '
            f'title="{html.escape(Path(source).name, quote=True)}"></span>'
        )
    parts.append("</div>")

    unknown = sum(1 for r in rows if not r["faces"])
    parts.append(f"<h2>{len(rows)} photos, {unknown} with no usable face (&lt; {min_face:.0f}px)</h2>")
    parts.append("<table><tr><th>threshold</th><th>photos kept</th><th>share kept</th></tr>")
    parts.extend(f"<tr><td>{t:.2f}</td><td>{kept}</td><td>{share:.0f}%</td></tr>" for t, kept, share in sweep(rows))
    parts.append("</table>")

    for album in sorted({r["album"] for r in rows}):
        group = [r for r in rows if r["album"] == album]
        states = [verdict(r, threshold) for r in group]
        matched = states.count("matched")
        unusable = states.count("unknown")
        below = states.count("below")
        near = sum(
            1 for r, s in zip(group, states, strict=True) if s == "below" and r["score"] >= threshold - NEAR_MISS
        )
        summary = (
            f"{album} <span class='muted'>— {len(group)} photos · {matched} matched · "
            f"{below} below ({near} within {NEAR_MISS:.2f}) · {unusable} unusable</span>"
        )
        parts.append(f"<details class='album'{' open' if matched else ''}><summary>{summary}</summary>")
        for r in sorted(group, key=lambda r: -r["score"]):
            state = verdict(r, threshold)
            cls, label = BADGES[state]
            box = face_box_style(r["best_box"], r.get("width", 0), r.get("height", 0))
            rel = r["rel"]
            actions = ""
            if interactive:
                key = html.escape(rel, quote=True)
                actions = (
                    f'<div class="actions">'
                    f'<button onclick="decide(this,&quot;{key}&quot;,&quot;keep&quot;)">keep</button>'
                    f'<button onclick="decide(this,&quot;{key}&quot;,&quot;drop&quot;)">drop</button>'
                    f'<button onclick="decide(this,&quot;{key}&quot;,&quot;clear&quot;)">clear</button></div>'
                )
            parts.append(
                f'<div class="card {state}" data-state="{state}">'
                f'<span class="frame" title="{html.escape(rel, quote=True)}">'
                f'<img loading="lazy" src="{src(r["path"])}">'
                f"{f'<span class="box" style="{box}"></span>' if box else ''}</span>"
                f'<div class="row"><span class="badge {cls}">{label}</span>'
                f'<span class="score">{r["score"]:.3f}</span>'
                f'<span class="muted">{r["faces"]} face(s) ≥ {min_face:.0f}px</span></div>{actions}</div>'
            )
        parts.append("</details>")

    head = f"<!doctype html><meta charset=utf-8><title>face review — {person}</title><style>{REPORT_CSS}</style>\n"
    return head + "\n".join(parts) + (INTERACTIVE_JS if interactive else "")


INTERACTIVE_JS = """
<script>
const BADGE = {matched:["ok","MATCH"], below:["","below"], unknown:["warn","no face"],
               included:["add","ADDED"], excluded:["drop","DROPPED"]};
async function decide(button, key, action) {
  const response = await fetch("/decide", {method: "POST", body: JSON.stringify({key, action})});
  if (!response.ok) { alert("could not save: " + response.status); return; }
  const card = button.closest(".card");
  const state = action === "keep" ? "included" : action === "drop" ? "excluded" : card.dataset.state;
  const [cls, label] = BADGE[state];
  card.className = "card " + state;
  const badge = card.querySelector(".badge");
  badge.className = "badge " + cls;
  badge.textContent = label;
}
</script>
"""
