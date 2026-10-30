"""Signature detection on a rendered page (no ML models, no APIs).
Handles stamps (SV-12,13,25), table lines (SV-14), printed/handwritten text (SV-21-25),
faint/small/blurred/shadowed/cut-off signatures (SV-16-20), thumb prints (SV-57),
duplicates (SV-28), roles (SV-08-11,58), multiple signatures (SV-07)."""
import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .config import FIELD_PATTERN, ROLE_PATTERNS


@dataclass
class Candidate:
    id: str
    page: int
    bbox: tuple                 # x, y, w, h  (page pixels, tight around ink)
    mask: np.ndarray            # 0/255 ink mask used for comparison
    crop: np.ndarray            # BGR crop for display
    kind: str = "signature"     # signature | handwriting | stamp | text | thumb
    score: float = 0.0          # detector confidence 0..1
    ink: str = "black"
    role: str = "unknown"
    role_conf: str = "none"     # label | position | none | manual
    role_label: str = ""
    quality: float = 1.0
    flags: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    source: str = "page"
    included: bool = True
    edge: tuple = (False, False, False, False)  # left, top, right, bottom


def image_quality(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(lap)


def preprocess(img, faint=False):
    """Illumination normalisation (shadows, folds, poor lighting - SV-20)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(3, (min(small.shape) // 10) | 1)
    bg = cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    thr = 238 if faint else 205
    ink = ((norm < thr).astype(np.uint8)) * 255
    if not faint:
        ink = cv2.medianBlur(ink, 3)          # de-speckle; skipped in faint mode (would erase thin strokes)
    else:
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        small_c = np.where(st[:, 4] < 12)[0]
        ink[np.isin(lab, small_c[small_c > 0])] = 0
    return gray, norm, ink


def _remove_lines(ink, s, long_side=1754):
    """Table lines / signature lines (SV-14). Strokes crossing a line are restored."""
    L = max(15, int(70 * s))      # long straight runs = printed lines (same rule for specimens and questioned signatures)
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    # thick runs (>=3px, i.e. pen strokes / flourishes) are NOT printed rules
    thick_h = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 4))), np.ones((5, 5), np.uint8))
    thick_v = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, L))), np.ones((5, 5), np.uint8))
    hl = hl & ~thick_h
    vl = vl & ~thick_v
    lines = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    cleaned = cv2.bitwise_and(ink, cv2.bitwise_not(lines))
    t = 9
    ones = lambda a, b: np.ones((a, b), np.uint8)
    for kern, a0, a1 in ((ones(t + 2, 1), (0, 0), (0, t + 1)), (ones(1, t + 2), (0, 0), (t + 1, 0))):
        below = cv2.dilate(cleaned, kern, anchor=a0)
        above = cv2.dilate(cleaned, kern, anchor=a1)
        cleaned |= (lines & ink & below & above)
    return cleaned, lines


def _ink_class(hsv, mask):
    sel = mask > 0
    if sel.sum() == 0:
        return "black"
    sat = np.median(hsv[..., 1][sel])
    hue = np.median(hsv[..., 0][sel])
    if sat < 55:
        return "black"
    if 90 <= hue <= 135:
        return "blue"
    if hue < 15 or hue >= 160:
        return "red"
    return "other"


def _find_rings(ink, s, H, W):
    """Circular / oval stamp borders (SV-12/13). Returns list of (x, y, w, h, band_mask)."""
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, hier = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    if hier is not None:
        for i, c in enumerate(cnts):
            if hier[0][i][3] != -1 or hier[0][i][2] == -1:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if min(w, h) < 80 * s or max(w, h) > 0.6 * max(H, W):
                continue
            area, per = cv2.contourArea(c), cv2.arcLength(c, True)
            if 4 * np.pi * area / (per * per + 1e-9) < 0.55:
                continue
            ch, best = hier[0][i][2], None
            while ch != -1:
                ca = cv2.contourArea(cnts[ch])
                if best is None or ca > best[0]:
                    best = (ca, ch)
                ch = hier[0][ch][0]
            if best and best[0] > 0.35 * area:
                band = np.zeros((H, W), np.uint8)
                cv2.drawContours(band, [c], -1, 255, -1)
                cv2.drawContours(band, [cnts[best[1]]], -1, 0, -1)
                rings.append((x, y, w, h, cv2.dilate(band, np.ones((5, 5), np.uint8))))
    if rings:
        return rings
    # fallback: Hough circle for rings that touch / are crossed by a signature
    f = 0.5
    sm = cv2.GaussianBlur(cv2.resize(ink, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (5, 5), 0)
    cs = cv2.HoughCircles(sm, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60 * s, param1=120, param2=28,
                          minRadius=int(40 * s * f * 1.4), maxRadius=int(min(H, W) * f * 0.3))
    if cs is None:
        return rings
    for (cx, cy, r) in cs[0][:3]:
        cx, cy, r = cx / f, cy / f, r / f
        th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        bestr, bestcov = r, 0
        for rr in np.arange(r - 6, r + 7, 2):
            xs = np.clip((cx + rr * np.cos(th)).astype(int), 0, W - 1)
            ys = np.clip((cy + rr * np.sin(th)).astype(int), 0, H - 1)
            cov = float((cv2.dilate(ink, np.ones((3, 3), np.uint8))[ys, xs] > 0).mean())
            if cov > bestcov:
                bestr, bestcov = rr, cov
        if bestcov >= 0.65:
            band = np.zeros((H, W), np.uint8)
            cv2.circle(band, (int(cx), int(cy)), int(bestr), 255, 7)
            rings.append((int(cx - bestr), int(cy - bestr), int(2 * bestr), int(2 * bestr), band))
    return rings


def _skel_stats(sub):
    h, w = sub.shape
    f = 1.0
    if max(h, w) > 500:
        f = 500.0 / max(h, w)
        sub = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    sk = skeletonize(sub > 0).astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    ends = int(((sk == 1) & (nb == 1)).sum())
    junc = int(((sk == 1) & (nb >= 3)).sum() > 0) * int(((sk == 1) & (nb >= 3)).sum() / 3)
    return int(sk.sum() / f), ends, min(junc, 40)


def _group_lines(words):
    lines = {}
    for w in words:
        lines.setdefault((w[5], w[6]), []).append(w)
    out = []
    for ws in lines.values():
        txt = " ".join(x[4] for x in sorted(ws, key=lambda z: z[0]))
        out.append((min(x[0] for x in ws), min(x[1] for x in ws), max(x[2] for x in ws), max(x[3] for x in ws), txt, len(ws)))
    return out


def find_label(bbox, words, s):
    """Nearest role label (PDF text layer) below / left / above the signature (SV-08..11)."""
    x, y, w, h = bbox
    best = (1e9, "unknown", "")
    fbest = (1e9, "")
    for (lx0, ly0, lx1, ly1, txt, n) in _group_lines(words):
        if n > 7:
            continue
        t = txt.lower().strip()
        xo = min(x + w, lx1) - max(x, lx0)
        yo = min(y + h, ly1) - max(y, ly0)
        d = None
        if ly0 >= y + h - 12 * s and ly0 - (y + h) < 150 * s and xo > -60 * s:      # below
            d = (ly0 - (y + h)) + 0.4 * abs((lx0 + lx1) / 2 - (x + w / 2))
        elif lx1 <= x + 12 * s and x - lx1 < 450 * s and yo > -25 * s:             # left, same row
            d = (x - lx1) * 0.6 + 20
        elif ly1 <= y + 8 * s and y - ly1 < 70 * s and xo > 0:                      # above
            d = (y - ly1) + 30
        if d is None:
            continue
        for role, pat in ROLE_PATTERNS:
            if re.search(pat, t):
                if d < best[0]:
                    best = (d, role, txt)
                break
        if re.match(FIELD_PATTERN, t) and "sign" not in t and d < fbest[0]:
            fbest = (d, txt)
    return best, fbest


def detect_page(img, info, params, page_index=0) -> List[Candidate]:
    H, W = img.shape[:2]
    s = float(np.clip(max(H, W) / 1754.0, 0.25, 3.0))     # resolution independent (A4 @150dpi = 1.0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    q = image_quality(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cands = _detect(img, hsv, info, params, page_index, H, W, s, faint=False, page_lap=q)
    if not any(c.kind == "signature" for c in cands):          # SV-18 / SV-27 second, more sensitive pass
        c2 = _detect(img, hsv, info, params, page_index, H, W, s, faint=True, page_lap=q)
        if any(c.kind == "signature" for c in c2):
            for c in c2:
                c.notes.append("Found only with high-sensitivity (faint) mode")
                c.flags.append("faint")
            cands = c2
    return cands


def _detect(img, hsv, info, params, pi, H, W, s, faint, page_lap):
    words = info.words if info is not None else []
    gray, norm, ink = preprocess(img, faint)
    cleaned, lines = _remove_lines(ink, s, max(H, W))

    # --- coloured stamps removed per pixel (red/green/purple...). Pen colours kept: black & blue (SV-12)
    sat, hue = hsv[..., 1], hsv[..., 0]
    val = hsv[..., 2]
    stampish = (hue <= 85) | (hue >= 160)      # red / orange / yellow / green; pen colours (black, blue) are kept
    colored = ((sat > 90) & (val > 70) & stampish & (cleaned > 0)).astype(np.uint8) * 255
    stamp_px = cv2.dilate(colored, np.ones((5, 5), np.uint8)) if colored.any() else colored
    sig = cv2.bitwise_and(cleaned, cv2.bitwise_not(stamp_px))
    if colored.any():
        sig = cv2.morphologyEx(sig, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # --- ring stamps (any colour, incl. black) (SV-12/13/25)
    rings = _find_rings(cleaned, s, H, W)
    ring_boxes = []
    ring_px = np.zeros((H, W), np.uint8)
    for (x, y, w, h, band) in rings:
        ring_cls = _ink_class(hsv, cleaned & band)
        ring_px |= band
        before = sig.copy()
        sig = cv2.bitwise_and(sig, cv2.bitwise_not(band))
        # strokes crossing the ring are restored (they have ink on both sides of the band)
        cxm, cym, rm = x + w / 2.0, y + h / 2.0, max(w, h) / 2.0
        yy, xx = np.ogrid[:H, :W]
        d2 = (xx - cxm) ** 2 + (yy - cym) ** 2
        inner = (d2 < (rm - 9 * s) ** 2).astype(np.uint8)
        outer = (d2 > (rm + 9 * s) ** 2).astype(np.uint8)
        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        crossing = band & before & cv2.dilate(sig & inner, k9) & cv2.dilate(sig & outer, k9)
        if ring_cls == "black":                      # only differently-coloured (pen) strokes can be separated
            crossing &= ((sat > 40).astype(np.uint8) * 255)
        sig = sig | crossing
        # components inside the ring: stamp text unless different ink colour or protruding
        n, lab, st, _ = cv2.connectedComponentsWithStats(sig, connectivity=8)
        for k in range(1, n):
            cx, cy, cw, ch_, ar = st[k]
            inside = (cx >= x - 4 and cy >= y - 4 and cx + cw <= x + w + 4 and cy + ch_ <= y + h + 4)
            if not inside:
                continue
            m = (lab == k).astype(np.uint8) * 255
            same = _ink_class(hsv, m) == ring_cls
            if same and cw <= 0.7 * w:
                sig[lab == k] = 0
        ring_boxes.append((x, y, w, h))

    # --- printed text from the PDF text layer (components mostly inside word boxes)
    if words:
        wm = np.zeros((H, W), np.uint8)
        for (x0, y0, x1, y1, *_r) in words:
            cv2.rectangle(wm, (int(x0) - 1, int(y0) - 1), (int(x1) + 1, int(y1) + 1), 255, -1)
        n, lab, st, _ = cv2.connectedComponentsWithStats(sig, connectivity=8)
        inw = np.bincount(lab[(wm > 0) & (sig > 0)], minlength=n)
        for k in range(1, n):
            if inw[k] >= 0.8 * st[k][4]:
                sig[lab == k] = 0

    # --- group strokes into regions
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(31 * s) | 1, int(27 * s) | 1))
    dil = cv2.dilate(sig, ke)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    raw = []
    for r in range(1, n):
        x, y, w, h, _a = st[r]
        sub = (sig[y:y + h, x:x + w] > 0) & (lab[y:y + h, x:x + w] == r)
        ys, xs = np.where(sub)
        if len(xs) < 60 or (ys.max() - ys.min()) < 11 * s:
            continue
        x0, x1, y0, y1 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
        mask = (sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]).astype(np.uint8) * 255
        bw, bh = x1 - x0, y1 - y0
        nn, _, cst, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comp = [c for c in cst[1:] if c[4] >= 6]
        ncomp = len(comp)
        if ncomp == 0:
            continue
        hs = np.array([c[3] for c in comp], float)
        cv_ = float(hs.std() / (hs.mean() + 1e-9))
        density = float((mask > 0).sum()) / (bw * bh)
        sk_len, ends, junc = _skel_stats(mask)
        text_like = (ncomp >= 4 and cv_ < 0.5 and float(np.median(hs)) < 45 * s and bh < 75 * s) or \
                    (ncomp >= 12 and float(np.median(hs)) < 45 * s and cv_ < 0.6)
        sc = 0.0
        if 22 * s <= bh <= 260 * s and 60 * s <= bw <= 700 * s:
            sc += 0.30
        elif bh >= 14 * s and bw >= 40 * s:
            sc += 0.12
        sc += 0.15 if ncomp <= 6 else 0
        sc += 0.05 if ncomp <= 3 else 0
        sc += 0.10 if (cv_ > 0.45 or ncomp <= 2) else 0
        sc += 0.10 if 0.02 <= density <= 0.38 else 0
        cx = ends + junc
        sc += 0.15 if cx >= 6 else 0.07 if cx >= 3 else 0
        sc += 0.10 if sk_len / (bw + bh) >= 1.0 else 0
        if text_like:
            sc -= 0.45
        if (bh < 22 * s or bw < 60 * s) and ncomp > 2:
            sc -= 0.45          # tiny multi-part marks are label words, not signatures
        # thumb impression: compact, dense, roundish blob (SV-57)
        hull = cv2.convexHull(cv2.findNonZero(mask))
        solidity = float((mask > 0).sum()) / (cv2.contourArea(hull) + 1e-9)
        thumb = (density > 0.30 and solidity > 0.7 and 0.55 < bw / bh < 1.7 and
                 50 * s < min(bw, bh) and max(bw, bh) < 320 * s and ncomp <= 15)
        kind = "signature"
        if thumb:
            kind, sc = "thumb", max(sc, 0.6)
        elif text_like or (bh < 26 * s and ncomp >= 3) or (bh < 22 * s and bw / bh > 4):      # short multi-part marks = words
            kind = "text"
        raw.append((x0, y0, bw, bh, mask, sc, kind, ncomp))

    # --- build candidates
    cands: List[Candidate] = []
    for k, (x0, y0, bw, bh, mask, sc, kind, ncomp) in enumerate(sorted(raw, key=lambda r: (r[1] // 60, r[0]))):
        pad = int(8 * s)
        cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x0 + bw + pad), min(H, y0 + bh + pad)
        crop = img[cy0:cy1, cx0:cx1].copy()
        c = Candidate(id=f"p{pi + 1}-c{k + 1}", page=pi, bbox=(int(x0), int(y0), int(bw), int(bh)), mask=mask,
                      crop=crop, kind=kind, score=float(max(0.0, min(1.0, sc))))
        c.ink = _ink_class(hsv[y0:y0 + bh, x0:x0 + bw], (mask > 0).astype(np.uint8))
        if c.ink in ("red", "other") and kind == "signature" and sc < 0.9:
            c.kind, c.score = "stamp", c.score * 0.5               # coloured logo/stamp (SV-25)
            c.notes.append(f"{c.ink}-coloured mark: likely stamp/logo")
        # SV-16 cut off at page edge
        edge = (x0 <= 2, y0 <= 2, x0 + bw >= W - 2, y0 + bh >= H - 2)
        c.edge = edge
        if any(edge):
            c.flags.append("cut_off")
            c.notes.append("Signature touches page edge: may be truncated (SV-16)")
        # SV-17 too small
        if bh < 22 * s or bw < 60 * s:
            c.flags.append("too_small")
            c.notes.append("Signature is very small - comparison unreliable (SV-17)")
        # SV-12/13 overlap with stamp
        for (rx, ry, rw, rh) in ring_boxes:
            if x0 < rx + rw and x0 + bw > rx and y0 < ry + rh and y0 + bh > ry:
                c.flags.append("stamp_overlap")
                c.notes.append("Overlaps a stamp - strokes may be hidden (SV-12/13)")
                break
        else:
            if stamp_px.any() and stamp_px[y0:y0 + bh, x0:x0 + bw].sum() > 0:
                c.flags.append("stamp_overlap")
                c.notes.append("Overlaps a coloured stamp; stamp colour removed (SV-12)")
        # SV-14 overlap with lines / printed text
        if lines[y0:y0 + bh, x0:x0 + bw].sum() > 0 and bh > 20 * s:
            c.notes.append("Crossed by table/signature lines; lines removed (SV-14)")
        if words:
            for wd in words:
                if wd[0] < x0 + bw and wd[2] > x0 and wd[1] < y0 + bh and wd[3] > y0 and kind == "signature" and ncomp <= 3 and bh > 40 * s:
                    c.flags.append("text_overlap")
                    c.notes.append("Overlaps printed text (SV-14)")
                    break
        # SV-48 stroke retention after cleaning
        region_ink = ink[y0:y0 + bh, x0:x0 + bw] > 0
        kept = (sig[y0:y0 + bh, x0:x0 + bw] > 0)
        lost = region_ink & ~kept & ~(cv2.dilate(lines, np.ones((5, 5), np.uint8))[y0:y0 + bh, x0:x0 + bw] > 0) \
            & ~(stamp_px[y0:y0 + bh, x0:x0 + bw] > 0) & ~(ring_px[y0:y0 + bh, x0:x0 + bw] > 0)
        retention = kept.sum() / (kept.sum() + lost.sum() + 1e-9)
        if retention < 0.85 and kept.sum() > 0:
            full = (ink[y0:y0 + bh, x0:x0 + bw]).copy()
            c.mask = full
            c.flags.append("cleaning_loss")
            c.notes.append("Cleaning removed too many strokes; un-cleaned crop used instead (SV-48)")
        # quality: sharpness, contrast, size, faintness (SV-18/19)
        g = gray[y0:y0 + bh, x0:x0 + bw]
        lap = cv2.Laplacian(g, cv2.CV_64F).var()
        contrast = 255 - float(np.median(norm[y0:y0 + bh, x0:x0 + bw][region_ink])) if region_ink.any() else 0
        sharp = min(1.0, lap / 400.0 + 0.0)
        qual = 0.30 * min(1.0, page_lap / 150.0) + 0.15 * sharp + 0.25 * min(1.0, contrast / 90.0) + \
            0.20 * min(1.0, min(bh / (30 * s), bw / (90 * s))) + 0.10
        if page_lap < 25:
            c.flags.append("blurred")
            c.notes.append("Image is blurred / low resolution (SV-19)")
        if contrast < 45 or faint:
            c.flags.append("faint")
            c.notes.append("Signature ink is very faint (SV-18)")
            qual -= 0.1
        if "cut_off" in c.flags:
            qual -= 0.25
        if "too_small" in c.flags:
            qual -= 0.2
        c.quality = float(max(0.0, min(1.0, qual)))
        # source (embedded raster image vs page content) SV-33
        for r in (info.img_rects if info else []):
            ix = min(r[2], x0 + bw) - max(r[0], x0)
            iy = min(r[3], y0 + bh) - max(r[1], y0)
            if ix > 0.6 * bw and iy > 0.6 * bh and (r[2] - r[0]) < 0.7 * W:
                c.source = "embedded image"
        # Reject obvious non-signature content inside embedded photos / graphics (SV-64).
        # A genuine raster signature is sparse ink on a mostly uniform background;
        # portraits and filled graphic blocks have broad colour coverage or dense fill.
        if c.source == "embedded image":
            patch = img[cy0:cy1, cx0:cx1]
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV) if patch.size else None
            sat_ratio = 0.0
            if hsv_patch is not None:
                # coloured stamp / ring pixels crossing the signature are not "photo colour" (SV-12/13)
                excl = (cv2.dilate(stamp_px | ring_px, np.ones((9, 9), np.uint8))[cy0:cy1, cx0:cx1] > 0)
                keep_px = ~excl
                sat_ratio = float((hsv_patch[..., 1][keep_px] > 70).mean()) if keep_px.any() else 0.0
            mask_density = float((c.mask > 0).mean()) if c.mask.size else 0.0
            # a fingerprint / thumb impression is a dense single-colour blob on white paper; a photo fills its rectangle
            white_frac = float((cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 235).mean()) if patch.size else 0.0
            thumb_blob = c.kind == "thumb" and white_frac > 0.12
            if (sat_ratio > 0.18 or mask_density > 0.42) and not thumb_blob:
                c.kind = "image"
                c.included = False
                c.notes.append("Embedded photo/graphic region excluded; not treated as a signature (SV-64)")

        # role / field label
        (d, role, label), (fd, flabel) = find_label(c.bbox, words, s) if words else ((1e9, "unknown", ""), (1e9, ""))
        if role != "unknown":
            c.role, c.role_conf, c.role_label = role, "label", label
        if c.kind in ("signature",) and flabel and fd < d and ncomp > 1 and c.score < 0.8:
            c.kind, c.score = "handwriting", c.score * 0.6
            c.notes.append(f"Handwritten entry next to field '{flabel}' - not a signature (SV-24)")
        if c.kind in ("signature", "thumb") and c.score >= params.min_sig_score or c.kind in ("thumb",):
            pass
        cands.append(c)

    cands = _nms(cands)
    sig_like = [c for c in cands if c.kind in ("signature", "thumb") and c.score >= params.min_sig_score]
    sig_like.sort(key=lambda c: -c.score)
    keep = {id(c) for c in sig_like[:params.max_candidates_per_page]}
    for c in cands:
        if id(c) not in keep and c.kind in ("signature",):
            c.kind = "text" if c.score < params.min_sig_score else c.kind
    cands = [c for c in cands if c.kind != "text" or c.score > 0.1 or True]
    # keep only meaningful 'other' marks to avoid clutter
    cands = [c for c in cands if c.kind in ("signature", "thumb") and c.score >= params.min_sig_score
             or c.kind in ("handwriting", "stamp") or (c.kind == "text" and c.score >= 0.2)]
    for i, c in enumerate(cands):
        c.id = f"p{pi + 1}-c{i + 1}"
        if c.kind in ("signature", "thumb") and c.score < params.min_sig_score:
            c.included = False
        if c.kind not in ("signature", "thumb"):
            c.included = False
    _positional_roles(cands)
    return cands


def _nms(cands):
    """Suppress duplicate detections of the same signature (SV-28)."""
    cands = sorted(cands, key=lambda c: -c.score)
    out = []
    for c in cands:
        x, y, w, h = c.bbox
        dup = False
        for o in out:
            ox, oy, ow, oh = o.bbox
            ix = max(0, min(x + w, ox + ow) - max(x, ox))
            iy = max(0, min(y + h, oy + oh) - max(y, oy))
            inter = ix * iy
            if inter / (w * h + ow * oh - inter + 1e-9) > 0.3 or inter / (min(w * h, ow * oh) + 1e-9) > 0.5:
                dup = True
                o.notes.append("Duplicate detection of the same signature suppressed (SV-28)") if "dup" not in o.flags else None
                if "dup" not in o.flags:
                    o.flags.append("dup")
                break
        if not dup:
            out.append(c)
    return out


def _positional_roles(cands):
    """SV-10: when labels are unreadable, suggest roles by position (low confidence)."""
    sigs = [c for c in cands if c.kind == "signature" and c.included is not False and c.role == "unknown"]
    if len(sigs) >= 2:
        for c in sigs:
            c.notes.append("Cannot tell whose signature this is - assign the signer manually (SV-10)")
            c.flags.append("role_unknown")
    elif len(sigs) == 1:
        sigs[0].flags.append("role_unknown")


def _canvas_of(items, x0, y0, x1, y1):
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for c in items:
        bx, by, bw, bh = c.bbox
        m[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw] |= c.mask[:bh, :bw]
    return m


def merge_split_signatures(by_page):
    """SV-15: one signature cut in two by a page/image boundary -> stitch halves together.
    The lower half is often fragmented into several pieces, so all pieces touching the top edge are merged."""
    pages = sorted(by_page)
    for a, b in zip(pages, pages[1:]):
        if b != a + 1:
            continue
        for A in [c for c in by_page[a] if c.edge[3] and c.kind == "signature"]:
            ax, ay, aw, ah = A.bbox
            Bs = [c for c in by_page[b] if c.edge[1] and min(ax + aw, c.bbox[0] + c.bbox[2]) - max(ax, c.bbox[0]) > 0.0
                  and c.kind in ("signature", "text", "handwriting")]
            if not Bs:
                continue
            bx0, by0 = min(c.bbox[0] for c in Bs), min(c.bbox[1] for c in Bs)
            bx1, by1 = max(c.bbox[0] + c.bbox[2] for c in Bs), max(c.bbox[1] + c.bbox[3] for c in Bs)
            if (bx1 - bx0) < 0.3 * aw:
                continue
            x0, x1 = min(ax, bx0), max(ax + aw, bx1)
            top = np.zeros((ah, x1 - x0), np.uint8)
            top[:, ax - x0:ax - x0 + aw] = A.mask
            bot = np.zeros((by1 - by0, x1 - x0), np.uint8)
            bot[:, bx0 - x0:bx1 - x0] = _canvas_of(Bs, bx0, by0, bx1, by1)
            A.mask = np.vstack([top, bot])
            A.bbox = (x0, ay, x1 - x0, ah + (by1 - by0))
            ch = max([A.crop.shape[1]] + [c.crop.shape[1] for c in Bs])
            pad = lambda im: cv2.copyMakeBorder(im, 0, 0, 0, ch - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            A.crop = np.vstack([pad(A.crop)] + [pad(c.crop) for c in Bs])
            A.flags = [f for f in A.flags if f != "cut_off"] + ["stitched"]
            A.notes.append(f"Signature was split between pages {a + 1} and {b + 1}; halves stitched (SV-15)")
            A.edge = (False, False, False, False)
            for c in Bs:
                by_page[b].remove(c)
    return by_page
