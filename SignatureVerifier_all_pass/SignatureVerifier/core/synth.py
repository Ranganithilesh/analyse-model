"""Synthetic signature / document generator used by the sample tool and the self-tests.
Everything is drawn locally with OpenCV + PyMuPDF."""
import io

import cv2
import numpy as np
from PIL import Image
try:
    import pymupdf as fitz
except ImportError:
    import fitz


def _spline(pts, n=14):
    pts = np.asarray(pts, float)
    P = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


def person_template(seed):
    r = np.random.RandomState(seed)
    n = r.randint(9, 14)
    x = np.sort(r.uniform(0, 1, n)) * 300
    y = r.uniform(0.1, 0.9, n) * 90 + np.sin(np.arange(n) * r.uniform(0.8, 2.2)) * 25
    loops = []
    for _ in range(r.randint(1, 3)):
        c = r.randint(1, n - 1)
        loops.append((x[c], y[c], r.uniform(14, 26)))
    underline = (r.uniform(0, 60), r.uniform(100, 118), r.uniform(200, 320))
    return {"pts": np.stack([x, y], 1), "loops": loops, "ul": underline, "seed": seed}


def render_signature(seed, variant=0, ink=(160, 40, 20), thickness=3, scale=1.0, angle=0.0, jitter=3.0, size=(360, 150)):
    """BGR ink colour; variant>0 produces natural intra-person variation."""
    t = person_template(seed)
    r = np.random.RandomState(seed * 1000 + variant)
    pts = t["pts"] + (r.normal(0, jitter * (1 if variant else 0), t["pts"].shape))
    pts[:, 0] *= 1 + r.normal(0, 0.03 if variant else 0)
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    curve = _spline(pts)
    poly = (curve + [30, 15]).astype(np.int32)
    cv2.polylines(img, [poly], False, ink, thickness, cv2.LINE_AA)
    for (cx, cy, rr) in t["loops"]:
        rr = rr * (1 + r.normal(0, 0.08 if variant else 0))
        cv2.ellipse(img, (int(cx + 30), int(cy + 15)), (int(rr), int(rr * 0.6)), 0, 0, 340, ink, thickness, cv2.LINE_AA)
    x0, y0, ln = t["ul"]
    cv2.line(img, (int(x0 + 30), int(y0 + 15)), (int(x0 + 30 + ln), int(y0 + 17 + (r.normal(0, 2) if variant else 0))), ink, thickness, cv2.LINE_AA)
    if scale != 1.0 or angle:
        M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, scale)
        img = cv2.warpAffine(img, M, size, borderValue=(255, 255, 255))
    return img


def sig_mask(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ((g < 200).astype(np.uint8)) * 255


def to_png_rgba(img):
    """Transparent PNG with the pen colour kept crisp (alpha carries the anti-aliasing)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = img.reshape(-1, 3)[g.ravel().argmin()]
    gmin = float(g.min())
    a = np.clip((255 - g) / max(1.0, 255 - gmin) * 255, 0, 255).astype(np.uint8)
    rgba = np.dstack([np.full(g.shape, dark[2], np.uint8), np.full(g.shape, dark[1], np.uint8), np.full(g.shape, dark[0], np.uint8), a])
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG")
    return b.getvalue()


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


# ------------------------------------------------------------------ documents
def make_pdf(signers, landscape=False, page_no=(1, 1), title="LOAN AGREEMENT", table=False, stamp_at=None,
             stamp_color=(0.8, 0.0, 0.0), as_images=False, lines_through=False, extra_text=True, vector=False):
    """signers: list of dict(label, seed, variant, pos=(x,y) in pt, size=(w,h), ink, scale, angle, paste=img)"""
    d = fitz.open()
    W, H = (842, 595) if landscape else (595, 842)
    p = d.new_page(width=W, height=H)
    p.insert_text((50, 60), title, fontsize=18)
    if extra_text:
        for i in range(6):
            p.insert_text((50, 100 + i * 16), f"Clause {i + 1}: The party agrees to the terms and conditions of this facility.", fontsize=9)
    p.insert_text((W - 140, H - 25), f"Page {page_no[0]} of {page_no[1]}", fontsize=8)
    if table:
        for r in range(4):
            p.draw_line((50, 230 + r * 28), (W - 50, 230 + r * 28), width=1)
        for c in range(4):
            x = 50 + c * (W - 100) / 3
            p.draw_line((x, 230), (x, 230 + 84), width=1)
    for s in signers:
        x, y = s.get("pos", (60, 400))
        w, h = s.get("size", (150, 62))
        img = s.get("paste") if s.get("paste") is not None else render_signature(
            s.get("seed", 1), s.get("variant", 1), ink=s.get("ink", (160, 40, 20)), scale=s.get("scale", 1.0), angle=s.get("angle", 0))
        if vector:
            t = person_template(s["seed"])
            pts = _spline(t["pts"]) * [w / 360, h / 150] + [x + 8, y + 8]
            p.draw_polyline([fitz.Point(*q) for q in pts], color=(0.1, 0.1, 0.7), width=1.2)
        else:
            p.insert_image(fitz.Rect(x, y, x + w, y + h), stream=to_png_rgba(img))
        p.draw_line((x, y + h + 3), (x + w + 20, y + h + 3), width=0.8)
        if s.get("label"):
            p.insert_text((x, y + h + 15), s["label"], fontsize=9)
    if lines_through:
        x, y = signers[0].get("pos", (60, 400))
        p.draw_line((x - 10, y + 30), (x + 200, y + 30), width=1.2)
    if stamp_at:
        cx, cy = stamp_at
        p.draw_circle((cx, cy), 42, color=stamp_color, width=2.2)
        p.draw_circle((cx, cy), 34, color=stamp_color, width=1)
        p.insert_text((cx - 26, cy + 3), "APPROVED", fontsize=9, color=stamp_color)
    data = d.tobytes()
    if as_images:
        data = to_scanned(data)
    return data


def to_scanned(pdf_bytes, dpi=150, degrade=None):
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()
    for pg in src:
        pix = pg.get_pixmap(dpi=dpi, alpha=False)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
        if degrade:
            arr = degrade(arr)
        npg = out.new_page(width=arr.shape[1] * 72 / dpi, height=arr.shape[0] * 72 / dpi)
        npg.insert_image(npg.rect, stream=png_bytes(arr))
    return out.tobytes()


def render_first_page(pdf_bytes, dpi=150):
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = src[0].get_pixmap(dpi=dpi, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()


# degradations ---------------------------------------------------------------
def blur(a):
    return cv2.GaussianBlur(cv2.resize(a, None, fx=0.35, fy=0.35), (0, 0), 1.6)


def shadow(a):
    h, w = a.shape[:2]
    grad = np.tile(np.linspace(0.55, 1.0, w), (h, 1))
    return (a * grad[..., None]).astype(np.uint8)


def noise(a):
    r = np.random.RandomState(1)
    return np.clip(a.astype(int) + r.normal(0, 9, a.shape), 0, 255).astype(np.uint8)


def faint(a):
    return (255 - (255 - a.astype(float)) * 0.20).astype(np.uint8)


def rotate180(a):
    return cv2.rotate(a, cv2.ROTATE_180)


def rotate90(a):
    return cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)
