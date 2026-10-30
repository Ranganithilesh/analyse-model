"""Quick signature check (simple, visual flow).

    document (image / PDF)  ->  find ONLY the pen signature  ->  crop  ->  clean (black ink on white)
    registered signature    ->  clean the same way
    compare both cleaned signatures  ->  APPROVED  /  NOT APPROVED

Everything is offline (OpenCV / scikit-image / PyMuPDF). No external API.
"""
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .compare import compare, identity_iou

# ----------------------------------------------------------------------------- decision
APPROVE_SCORE = 60.0      # cleaned-signature similarity (0-100) at/above which the signatures "look the same"
MIN_VOTES = 2             # how many of the 4 core metrics must individually agree


@dataclass
class Found:
    bbox: tuple                      # x, y, w, h in page pixels
    crop: np.ndarray                 # BGR crop of the signature region (document colours)
    mask: np.ndarray                 # 0/255 ink mask of the signature (crop size)
    score: float = 0.0               # how "signature-like" the region is
    notes: list = field(default_factory=list)


@dataclass
class QuickResult:
    approved: bool
    verdict: str                     # APPROVED | NOT APPROVED | NO SIGNATURE FOUND
    score: float
    reasons: list
    metrics: dict
    original: Optional[np.ndarray] = None       # page with the found box drawn
    doc_crop: Optional[np.ndarray] = None       # step 2: cropped signature
    doc_clean: Optional[np.ndarray] = None      # step 3: cleaned document signature
    ref_clean: Optional[np.ndarray] = None      # step 4: cleaned registered signature
    overlay: Optional[np.ndarray] = None        # both signatures overlaid (red = document, blue = registered)
    found: Optional[Found] = None


# ----------------------------------------------------------------------------- image loading
def load_page_images(name: str, data: bytes, dpi: int = 170, max_pages: int = 20) -> List[np.ndarray]:
    """Return the pages of a PDF / image file as BGR arrays."""
    if name.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz
        pdf = fitz.open(stream=data, filetype="pdf")
        if pdf.needs_pass:
            raise ValueError("PDF is password protected")
        out = []
        for i in range(min(len(pdf), max_pages)):
            pix = pdf[i].get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            out.append(cv2.cvtColor(arr[:, :, :3].copy(), cv2.COLOR_RGB2BGR))
        if not out:
            raise ValueError("PDF has no pages")
        return out
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Could not read the image")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:                                   # transparent PNG: composite on white
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr = (arr[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return [arr]


# ----------------------------------------------------------------------------- ink mask
def _ink_mask(img):
    """Ink = pixels clearly darker than their local paper (shadow / uneven light tolerant)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(15, (min(h, w) // 8) | 1)
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, max(3, (k // 4) | 1)), (w, h), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    # Otsu on the normalised image, but never so permissive that paper noise becomes ink
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(t, 150, 215))
    return ((norm < t).astype(np.uint8)) * 255, norm


def _remove_rules(ink, s):
    """Remove long straight printed lines (table borders, signature underline). Pen strokes crossing them are kept."""
    L = max(25, int(60 * s))
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    thick_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 4)))
    thick_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, L)))
    lines = (hl & ~cv2.dilate(thick_h, np.ones((5, 5), np.uint8))) | (vl & ~cv2.dilate(thick_v, np.ones((5, 5), np.uint8)))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    out = cv2.bitwise_and(ink, cv2.bitwise_not(lines))
    # restore pen strokes that cross a rule (ink present just above AND below the rule)
    for kern, a0, a1 in ((np.ones((8, 1), np.uint8), (0, 0), (0, 7)), (np.ones((1, 8), np.uint8), (0, 0), (7, 0))):
        out |= (lines & ink & cv2.dilate(out, kern, anchor=a0) & cv2.dilate(out, kern, anchor=a1))
    return out


def _remove_filled_areas(ink, s):
    """Photos, logos, header bands: big dense blobs (a pen signature is thin and sparse)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = ink.copy()
    for i in range(1, n):
        x, y, w, h, a = st[i]
        fill = a / float(w * h)
        if (w > 60 * s and h > 40 * s and fill > 0.45) or (w > 150 * s and fill > 0.35 and h > 12 * s):
            out[lab == i] = 0
    return out


# ----------------------------------------------------------------------------- optional anchor words
def _anchor_words(img, pdf_words=None):
    """Words like 'signature' / 'signed' with boxes (tesseract if installed). Pure bonus - never required."""
    words = []
    if shutil.which("tesseract"):
        try:
            ok, png = cv2.imencode(".png", img)
            r = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11", "tsv"], input=png.tobytes(),
                               capture_output=True, timeout=25)
            for line in r.stdout.decode("utf8", "ignore").splitlines()[1:]:
                p = line.split("\t")
                if len(p) >= 12 and p[11].strip():
                    x, y, w, h = map(int, p[6:10])
                    words.append((x, y, x + w, y + h, p[11].strip().lower()))
        except Exception:
            pass
    return [w for w in words if re.search(r"signat|signed|sign\b|sgd", w[4])]


# ----------------------------------------------------------------------------- finding the signature
def _text_like_filter(ink, s):
    """Drop printed text: small, similarly-sized components that sit in rows."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 6:
        return ink, 12.0
    hs = st[1:, 3][st[1:, 4] > 8]
    med_h = float(np.median(hs)) if len(hs) else 12.0
    med_h = float(np.clip(med_h, 6 * s, 30 * s))
    comps = [(i, st[i], cen[i]) for i in range(1, n) if st[i, 4] > 6]
    textish = []
    for i, b, c in comps:
        w, h = b[2], b[3]
        if 0.45 * med_h <= h <= 1.7 * med_h and w <= 2.2 * max(h, med_h * 0.6) + 4:
            textish.append((i, b, c))
    out = ink.copy()
    cy = np.array([c[1] for _, _, c in textish]) if textish else np.array([])
    cx = np.array([c[0] for _, _, c in textish]) if textish else np.array([])
    for k, (i, b, c) in enumerate(textish):
        near = (np.abs(cy - c[1]) < 0.6 * med_h) & (np.abs(cx - c[0]) < 14 * med_h)
        if near.sum() - 1 >= 3:                                   # at least 3 siblings on the same line -> printed text
            out[lab == i] = 0
    return out, med_h


def _merge_boxes(boxes, gap):
    """Join pieces of one signature (first/last name, flourish, underline) that sit close together."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                gx = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
                gy = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                if gx < gap * 1.5 and gy < gap * 0.8:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = (x0, y0, x1 - x0, y1 - y0, a[4] + b[4], a[5] | b[5])
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _cluster_boxes(sig, s):
    kx, ky = max(9, int(20 * s)), max(5, int(9 * s))
    grown = cv2.dilate(sig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        ink_px = int(((lab[y:y + h, x:x + w] == i) & (sig[y:y + h, x:x + w] > 0)).sum())
        if ink_px < 40:
            continue
        boxes.append((x, y, w, h, ink_px, (lab == i) & (sig > 0)))
    return boxes


def find_signatures(img, anchors=None, max_results=5) -> List[Found]:
    """Return the most signature-like regions of the page (best first)."""
    H, W = img.shape[:2]
    s = float(np.clip(max(H, W) / 1000.0, 0.4, 3.0))
    ink, norm = _ink_mask(img)
    ink = _remove_rules(ink, s)
    ink = _remove_filled_areas(ink, s)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nb, lab0, st0, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    ink[np.isin(lab0, np.where(st0[:, 4] < max(3, int(4 * s * s)))[0][1:])] = 0
    sig, med_h = _text_like_filter(ink, s)
    anchors = anchors if anchors is not None else _anchor_words(img)

    found = []
    for x, y, w, h, ink_px, m in _merge_boxes(_cluster_boxes(sig, s), 3 * med_h):
        if h < 1.7 * med_h or w < 2.2 * med_h or w < 25 * s:     # printed text lines are only ~1 text-height tall
            continue
        fill = ink_px / float(w * h)
        if fill > 0.40:
            continue
        # pen colour (blue / black): saturation of the ink pixels
        pix = hsv[m]
        sat = float(np.median(pix[:, 1])) if len(pix) else 0
        hue = float(np.median(pix[:, 0])) if len(pix) else 0
        blue = 90 <= hue <= 135 and sat > 60
        # sc: bigger & more cursive & sparser = more signature-like
        size = min(1.0, (h / (3.5 * med_h)))
        wide = min(1.0, w / (6 * med_h))
        sparse = 1.0 - min(1.0, fill / 0.4)
        sc = 0.35 * size + 0.25 * wide + 0.2 * sparse + (0.1 if blue else 0.0)
        for ax0, ay0, ax1, ay1, _t in (anchors or []):           # sits on / just right of / just above a 'signature' label
            near_x = x < ax1 + 12 * med_h and x + w > ax0 - 2 * med_h
            near_y = abs((y + h / 2) - (ay0 + ay1) / 2) < 3.5 * h
            if near_x and near_y:
                sc += 0.25
                break
        pad = int(max(4, 0.03 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        found.append(Found((x0, y0, x1 - x0, y1 - y0), img[y0:y1, x0:x1].copy(),
                           (m[y0:y1, x0:x1].astype(np.uint8)) * 255, sc,
                           ["pen colour: " + ("blue" if blue else "dark")]))
    found.sort(key=lambda f: -f.score)
    return found[:max_results]


def whole_image_signature(img) -> Found:
    """For tightly cropped signature images (registered specimen, or an uploaded signature-only file)."""
    ink, _ = _ink_mask(img)
    H, W = img.shape[:2]
    s = float(np.clip(max(H, W) / 1000.0, 0.4, 3.0))
    # drop a long baseline rule only when it spans nearly the full width
    L = int(0.9 * W)
    if L > 20:
        rule = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 4)))
        ink = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(rule & ~thick, np.ones((3, 3), np.uint8))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if st[i, 4] >= max(3, int(3 * s * s)):                   # keep dots; drop single-pixel scan noise
            keep[lab == i] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) < 30:
        raise ValueError("No ink found in this image")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = max(3, int(0.02 * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    return Found((x0, y0, x1 - x0, y1 - y0), img[y0:y1, x0:x1].copy(), keep[y0:y1, x0:x1], 1.0,
                 ["whole image used (tight signature crop)"])


def looks_like_signature_only(img) -> bool:
    """A signature-only upload: few ink pieces filling most of the image (a form has many text pieces)."""
    ink, _ = _ink_mask(img)
    ys, xs = np.where(ink > 0)
    if len(xs) < 50:
        return False
    H, W = img.shape[:2]
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    pieces = int((st[1:, 4] >= 12).sum())
    return pieces <= 25 and bw >= 0.5 * W and bh >= 0.25 * H and ink.mean() / 255.0 < 0.25


# ----------------------------------------------------------------------------- cleaning
def clean_signature(found: Found, scale: int = 3, thickness: float = 1.0) -> np.ndarray:
    """Black pen strokes on a pure white background, smooth edges, no paper / noise / lines.
    Returns a BGR image (white background)."""
    crop, mask = found.crop, found.mask
    h, w = mask.shape
    # work at higher resolution for smooth outlines
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    up_g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    up_m = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # soft ink strength: only inside the (slightly grown) detected mask so printed leftovers are gone
    grow = cv2.dilate(up_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * scale + 1, 2 * scale + 1)))
    paper = float(np.percentile(gray[mask == 0], 60)) if (mask == 0).any() else 255.0
    strength = np.clip((paper - up_g.astype(np.float32)) / max(40.0, paper - float(np.percentile(gray[mask > 0], 25))), 0, 1)
    soft = np.where(grow > 127, strength, 0).astype(np.float32)
    soft = np.maximum(soft, (up_m > 127).astype(np.float32) * 0.6)
    soft = cv2.GaussianBlur(soft, (0, 0), 0.9 * scale / 2)
    binary = (soft > 0.38).astype(np.uint8) * 255
    if thickness != 1.0:
        k = max(1, int(round(abs(thickness - 1.0) * scale)))
        binary = cv2.dilate(binary, np.ones((k, k), np.uint8)) if thickness > 1 else cv2.erode(binary, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        if st[i, 4] < 6 * scale * scale:
            binary[lab == i] = 0
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.full((60, 200, 3), 255, np.uint8)
    pad = 6 * scale
    b = binary[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
    b = cv2.copyMakeBorder(b, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    smooth = cv2.GaussianBlur(b, (0, 0), 0.8 * scale / 2)               # anti-aliased edges
    return cv2.cvtColor(255 - smooth, cv2.COLOR_GRAY2BGR)


def clean_to_mask(clean_bgr) -> np.ndarray:
    g = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    return ((g < 160).astype(np.uint8)) * 255


def _overlay(a_mask, b_mask):
    """Both signatures scaled to the same box: red = document, blue = registered, dark = overlap."""
    def canvas(m):
        ys, xs = np.where(m > 0)
        t = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sc = min(600.0 / t.shape[1], 240.0 / t.shape[0])
        t = cv2.resize(t, (max(1, int(t.shape[1] * sc)), max(1, int(t.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        c = np.zeros((260, 620), np.uint8)
        y0, x0 = (260 - t.shape[0]) // 2, (620 - t.shape[1]) // 2
        c[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        return c > 100
    A, B = canvas(a_mask), canvas(b_mask)
    out = np.full((260, 620, 3), 255, np.uint8)
    out[A & ~B] = (60, 60, 230)          # BGR red   : only in document
    out[B & ~A] = (230, 120, 40)         # BGR blue  : only in registered
    out[A & B] = (30, 30, 30)            # overlap
    return out


# ----------------------------------------------------------------------------- the whole flow
def _prep_reference(reference_img):
    if looks_like_signature_only(reference_img) or max(reference_img.shape[:2]) < 700:
        return whole_image_signature(reference_img)
    c = find_signatures(reference_img)
    return c[0] if c else whole_image_signature(reference_img)


def verify(doc_page: np.ndarray, reference_img: np.ndarray, approve_score: float = APPROVE_SCORE,
           doc_found: Optional[Found] = None) -> QuickResult:
    """doc_page: the uploaded document page (BGR). reference_img: registered signature image (BGR).
    If the page holds several pen signatures (borrower, guarantor ...) each one is compared with the registered
    signature and the closest one decides."""
    page = doc_page.copy()
    if doc_found is not None:
        cands = [doc_found]
    elif looks_like_signature_only(doc_page):
        cands = [whole_image_signature(doc_page)]
    else:
        allc = find_signatures(doc_page)
        top = allc[0].score if allc else 0
        cands = [c for c in allc if c.score >= max(0.62, top - 0.2)][:4]
    if not cands:
        return QuickResult(False, "NO SIGNATURE FOUND", 0.0, ["No pen signature could be found in the document. "
                           "Upload a clearer scan."], {}, original=page)
    ref_found = _prep_reference(reference_img)
    ref_clean = clean_signature(ref_found)
    rm = clean_to_mask(ref_clean)

    best = None
    for c in cands:
        dc = clean_signature(c)
        dm = clean_to_mask(dc)
        if (dm > 0).sum() < 30:
            continue
        r = compare(dm, rm)
        if best is None or r["score"] > best[0]["score"]:
            best = (r, c, dc, dm)
    if best is None:
        return QuickResult(False, "NO SIGNATURE FOUND", 0.0, ["The region contained no readable ink."], {}, original=page)
    cmp_, doc_found, doc_clean, dm = best
    for c in cands:                                           # grey boxes = other candidates, green = the one compared
        x, y, w, h = c.bbox
        col = (0, 160, 0) if c is doc_found else (150, 150, 150)
        cv2.rectangle(page, (int(x), int(y)), (int(x + w), int(y + h)), col, max(2, page.shape[1] // 400))
    score, met, votes = cmp_["score"], cmp_["metrics"], cmp_["for"]
    approved = score >= approve_score and votes >= MIN_VOTES
    reasons = []
    if approved:
        reasons.append(f"The cleaned document signature and the registered signature look the same (similarity {score:.0f}/100).")
    elif score < approve_score:
        reasons.append(f"Similarity {score:.0f}/100 is below the approval level ({approve_score:.0f}) - the signatures look different.")
    else:
        reasons.append("Overall score is high but the individual checks do not agree - treated as different.")
    if len(cands) > 1:
        reasons.append(f"{len(cands)} signatures were found in the document; the closest one to the registered signature was used.")
    if cmp_["iou"] >= 0.97:
        reasons.append("Note: the two signatures are pixel-identical, which can mean a pasted copy.")
    return QuickResult(approved, "APPROVED" if approved else "NOT APPROVED", score, reasons, met, original=page,
                       doc_crop=doc_found.crop, doc_clean=doc_clean, ref_clean=ref_clean,
                       overlay=_overlay(dm, rm), found=doc_found)
