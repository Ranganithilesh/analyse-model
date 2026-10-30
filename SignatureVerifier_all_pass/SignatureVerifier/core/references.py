"""Bank specimen library (reference signatures) + audit log + review queue persistence.
Covers SV-38..42, 44, 63, 64. Everything is stored in ./data (plain files, no database server)."""
import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from .compare import compare
from .detector import detect_page, preprocess
from .loader import load_document

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LIB = os.path.join(DATA, "library")
os.makedirs(LIB, exist_ok=True)


class Library:
    def __init__(self, root=LIB):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "index.json")
        self.index = json.load(open(self.index_path)) if os.path.exists(self.index_path) else {}

    def save(self):
        json.dump(self.index, open(self.index_path, "w"), indent=1)

    def customers(self):
        return sorted(self.index)

    def name(self, cid):
        return self.index.get(cid, {}).get("name", cid)

    def specimens(self, cid):
        """Return list of dicts incl. loaded mask/crop (SV-41: a customer may have several)."""
        out = []
        for s in self.index.get(cid, {}).get("specimens", []):
            mp = os.path.join(self.root, cid, s["file"] + "_mask.png")
            cp = os.path.join(self.root, cid, s["file"] + "_crop.png")
            if os.path.exists(mp):
                out.append({**s, "mask": cv2.imread(mp, 0), "crop": cv2.imread(cp)})
        return out

    def add_customer(self, cid, name):
        self.index.setdefault(cid, {"name": name, "specimens": []})["name"] = name
        os.makedirs(os.path.join(self.root, cid), exist_ok=True)
        self.save()

    def delete_customer(self, cid):
        import shutil
        shutil.rmtree(os.path.join(self.root, cid), ignore_errors=True)
        self.index.pop(cid, None)
        self.save()

    def delete_specimen(self, cid, file):
        self.index[cid]["specimens"] = [s for s in self.index[cid]["specimens"] if s["file"] != file]
        for suf in ("_mask.png", "_crop.png"):
            try:
                os.remove(os.path.join(self.root, cid, file + suf))
            except OSError:
                pass
        self.save()

    def add_specimen(self, cid, mask, crop, quality, date, note="", low_quality=False):
        fn = f"s{int(time.time() * 1000) % 10**10}_{len(self.index[cid]['specimens'])}"
        cv2.imwrite(os.path.join(self.root, cid, fn + "_mask.png"), mask)
        cv2.imwrite(os.path.join(self.root, cid, fn + "_crop.png"), crop)
        self.index[cid]["specimens"].append({"file": fn, "date": str(date), "quality": round(float(quality), 2),
                                             "note": note, "low_quality": bool(low_quality)})
        self.save()

    def lookalikes(self, mask, exclude=None, thresh=70.0):
        """SV-44: other customers whose specimens look very similar."""
        out = []
        for cid in self.customers():
            if cid == exclude:
                continue
            best = max([compare(mask, s["mask"], try_flip=False)["score"] for s in self.specimens(cid)] or [0])
            if best >= thresh:
                out.append((cid, best))
        return out


def extract_specimen(name, data, params, password=None):
    """Find the signature in a specimen file (card scan / PDF / cropped image). Returns (mask, crop, quality, notes)."""
    notes = []
    doc = load_document(name, data, password=password, dpi=params.dpi, auto_rotate=params.auto_rotate)
    img = doc.render(0)

    # A wide, tightly-cropped specimen image is already a signature canvas.
    # The page detector groups nearby strokes, but flourishes/underlines and
    # the second part of a name can be disconnected and end up as separate
    # candidates (or be classified as handwriting/text).  For this input type,
    # preserve ALL visible pen-ink components and build one specimen mask.
    h, w = img.shape[:2]
    if w / max(h, 1) >= 1.8:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Keep dark ink and saturated blue/black pen strokes, including dots.
        blue_ink = (hsv[:, :, 1] > 45) & (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 2] < 250)
        dark_ink = gray < 175
        ink = (blue_ink | dark_ink).astype(np.uint8) * 255
        # Remove isolated scan noise, but retain small intentional marks/dots.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        cleaned = np.zeros_like(ink)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 3:
                cleaned[labels == i] = 255
        ys, xs = np.where(cleaned > 0)
        if len(xs) >= 20:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            # Only use this path when the marks substantially fill a wide
            # image; otherwise continue with normal page/signature detection.
            if (x1 - x0) >= 0.50 * w and (y1 - y0) >= 0.25 * h:
                pad = max(3, int(0.01 * max(h, w)))
                x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
                x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
                m = cleaned[y0:y1, x0:x1]
                crop = img[y0:y1, x0:x1].copy()
                lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                q = min(1.0, 0.4 + lap / 300.0) * min(1.0, min(m.shape) / 40.0)
                notes.append("Wide cropped specimen: all pen-ink components combined into one complete signature.")
                return m, crop, q, notes

    cands = [c for c in detect_page(img, doc.pages[0], params, 0) if c.kind == "signature"]
    if cands:
        c = max(cands, key=lambda c: (c.score + c.quality) * (1 + 0.0001 * c.bbox[2] * c.bbox[3]))
        if len(cands) > 1:
            notes.append(f"{len(cands)} signatures found on the specimen; the strongest was used (check it).")
        return c.mask, c.crop, c.quality, notes + c.notes
    # specimen is usually already a tight signature crop -> use the ink of the whole image
    _, _, ink = preprocess(img)
    wide = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.9 * img.shape[1])), 1)))
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(wide, np.ones((3, 3), np.uint8))))   # baseline rule across the crop
    ys, xs = np.where(ink > 0)
    if len(xs) < 80:
        raise ValueError("No ink found in the specimen image.")
    m = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    q = min(1.0, 0.4 + lap / 300.0) * min(1.0, min(m.shape) / 40.0)
    notes.append("Whole image used as signature (no separate region detected).")
    return m, crop, q, notes


# ---------------------------------------------------------------- audit + review queue
AUDIT = os.path.join(DATA, "audit_log.csv")
QUEUE = os.path.join(DATA, "review_queue.json")


def audit(user, action, doc, item, decision, note=""):
    new = not os.path.exists(AUDIT)
    with open(AUDIT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "user", "action", "document", "item", "decision", "note"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), user, action, doc, item, decision, note])


def load_audit():
    if not os.path.exists(AUDIT):
        return []
    return list(csv.DictReader(open(AUDIT, encoding="utf-8")))


def load_queue():
    return json.load(open(QUEUE)) if os.path.exists(QUEUE) else []


def save_queue(q):
    json.dump(q, open(QUEUE, "w"), indent=1)


def queue_add(doc_name, doc_sha, cand_id, page, reason, score, crop_png_path):
    q = load_queue()
    if any(x["doc_sha"] == doc_sha and x["cand"] == cand_id and x["status"] == "open" for x in q):
        return
    q.append({"doc": doc_name, "doc_sha": doc_sha, "cand": cand_id, "page": page, "reason": reason, "score": score,
              "crop": crop_png_path, "status": "open", "created": datetime.now().isoformat(timespec="seconds")})
    save_queue(q)
