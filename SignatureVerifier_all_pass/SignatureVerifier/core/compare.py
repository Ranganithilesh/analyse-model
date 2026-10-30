"""Signature comparison: scale/rotation/ink-colour invariant, deterministic (SV-22, 43, 45, 49, 51, 62).
Five independent metrics are fused; agreement between them is required for an automatic decision."""
import cv2
import numpy as np
from skimage.morphology import skeletonize

CW, CH = 256, 128
try:
    _HOG = cv2.HOGDescriptor((CW, CH), (32, 32), (16, 16), (16, 16), 9)
except AttributeError:          # some OpenCV builds lack HOGDescriptor -> equivalent scikit-image HOG
    _HOG = None


def _hog(img):
    if _HOG is not None:
        return _HOG.compute(img).ravel()
    from skimage.feature import hog
    return hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True).astype(np.float32)
ANGLES = (0, -8, 8, -16, 16)


def _tight(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def to_canvas(mask):
    t = _tight(mask)
    if t is None:
        return None
    h, w = t.shape
    sc = min((CW - 10) / w, (CH - 10) / h)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    r = (cv2.resize(t, (nw, nh), interpolation=cv2.INTER_AREA) > 40).astype(np.uint8)
    cv = np.zeros((CH, CW), np.uint8)
    y0, x0 = (CH - nh) // 2, (CW - nw) // 2
    cv[y0:y0 + nh, x0:x0 + nw] = r
    return cv


def rotate_mask(mask, ang):
    if ang == 0:
        return mask
    h, w = mask.shape
    d = int(np.hypot(h, w)) + 6
    big = np.zeros((d, d), np.uint8)
    y0, x0 = (d - h) // 2, (d - w) // 2
    big[y0:y0 + h, x0:x0 + w] = mask
    M = cv2.getRotationMatrix2D((d / 2, d / 2), ang, 1.0)
    return (cv2.warpAffine(big, M, (d, d), flags=cv2.INTER_LINEAR) > 40).astype(np.uint8) * 255


def _skel(c):
    return skeletonize(c > 0).astype(np.uint8)


def _counts(sk):
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    return int(((sk == 1) & (nb == 1)).sum()), int(((sk == 1) & (nb >= 3)).sum() // 3)


def features(mask):
    c = to_canvas(mask)
    if c is None:
        return None
    sk = _skel(c)
    thick = cv2.dilate(sk * 255, np.ones((3, 3), np.uint8))
    soft = cv2.GaussianBlur(thick, (9, 9), 0)
    t = _tight(mask)
    return {"canvas": c, "sk": sk, "dt": cv2.distanceTransform((sk == 0).astype(np.uint8), cv2.DIST_L2, 3),
            "hog": _hog(soft), "grid": cv2.resize(soft, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel(),
            "aspect": t.shape[1] / t.shape[0], "ec": _counts(sk), "ink_ratio": float((mask > 0).sum())}


def _metrics(a, b):
    def tm(dt, sk):                      # trimmed mean: ignores the worst 15% (missing strokes, flourishes - SV-47/48)
        v = np.sort(dt[sk > 0]) if sk.any() else np.array([50.0])
        return float(v[:max(1, int(len(v) * 0.85))].mean())
    d1, d2 = tm(b["dt"], a["sk"]), tm(a["dt"], b["sk"])
    chamfer = float(np.exp(-((d1 + d2) / 2) / 9.0))
    cos = float(np.dot(a["hog"], b["hog"]) / (np.linalg.norm(a["hog"]) * np.linalg.norm(b["hog"]) + 1e-9))
    hog = float(np.clip((cos - 0.45) / 0.5, 0, 1))
    ga, gb = a["grid"] - a["grid"].mean(), b["grid"] - b["grid"].mean()
    grid = float(np.clip(np.dot(ga, gb) / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-9), 0, 1))
    st = np.mean([1 - abs(x - y) / max(x, y, 1) for x, y in zip(a["ec"], b["ec"])])
    asp = min(a["aspect"], b["aspect"]) / max(a["aspect"], b["aspect"])
    return {"chamfer": chamfer, "hog": hog, "grid": grid, "structure": float(st), "aspect": float(asp)}


WEIGHTS = {"chamfer": 0.35, "hog": 0.25, "grid": 0.20, "structure": 0.10, "aspect": 0.10}


def identity_iou(m1, m2):
    """SV-45: pixel-level identity of two signatures (genuine repeats are never pixel-identical)."""
    a, b = to_canvas(m1), to_canvas(m2)
    if a is None or b is None:
        return 0.0
    k = np.ones((3, 3), np.uint8)
    ad, bd = cv2.dilate(a, k), cv2.dilate(b, k)
    return float(((a & bd).sum() + (ad & b).sum()) / (a.sum() + b.sum() + 1e-9))


def compare(q_mask, r_mask, try_flip=True):
    """Return dict(score 0-100, metrics, angle, votes_for, votes_against, iou)."""
    fr = features(r_mask)
    if fr is None or features(q_mask) is None:
        return {"score": 0.0, "metrics": {}, "angle": 0, "for": 0, "against": 0, "iou": 0.0}
    best = None
    angles = list(ANGLES)
    for ang in angles:
        fq = features(rotate_mask(q_mask, ang))
        if fq is None:
            continue
        m = _metrics(fq, fr)
        sc = 100 * sum(WEIGHTS[k] * m[k] for k in WEIGHTS)
        if best is None or sc > best[0]:
            best = (sc, m, ang)
    if try_flip and best[0] < 55:          # page may be upside-down (SV-03): also test 180 degrees
        for ang in (180, 172, 188, 90, 270):
            fq = features(rotate_mask(q_mask, ang))
            if fq is None:
                continue
            m = _metrics(fq, fr)
            sc = 100 * sum(WEIGHTS[k] * m[k] for k in WEIGHTS)
            if sc > best[0]:
                best = (sc, m, ang)
    sc, m, ang = best
    core = [m["chamfer"], m["hog"], m["grid"], m["structure"]]
    return {"score": round(float(sc), 1), "metrics": {k: round(v, 3) for k, v in m.items()}, "angle": ang,
            "for": sum(v >= 0.55 for v in core), "against": sum(v < 0.40 for v in core), "iou": identity_iou(q_mask, r_mask)}
