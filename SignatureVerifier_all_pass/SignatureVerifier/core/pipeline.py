"""Document analysis pipeline: per-page isolation, timeouts, progress, stitching (SV-15, 28, 60, 61, 64)."""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from .compare import identity_iou
from .detector import Candidate, detect_page, merge_split_signatures, preprocess
from .loader import Document

_POOL = ThreadPoolExecutor(max_workers=2)


@dataclass
class PageResult:
    index: int
    scale: float = 1.0
    thumb: bytes = b""
    error: str = ""
    notes: list = field(default_factory=list)
    skipped: bool = False


@dataclass
class Analysis:
    doc_name: str
    sha: str
    pages: Dict[int, PageResult] = field(default_factory=dict)
    candidates: List[Candidate] = field(default_factory=list)
    warnings: list = field(default_factory=list)
    elapsed: float = 0.0
    order: list = field(default_factory=list)


def _process(doc: Document, i: int, params):
    img = doc.render(i)
    info = doc.pages[i]
    if not info.width:
        info.height, info.width = img.shape[:2]
    cands = detect_page(img, info, params, i)
    s = 900.0 / max(img.shape[:2])
    s = min(1.0, s)
    thumb = cv2.imencode(".jpg", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    return cands, thumb, s


def analyze(doc: Document, params, progress=None, order=None) -> Analysis:
    t0 = time.time()
    res = Analysis(doc.name, doc.sha, warnings=list(doc.warnings))
    idxs = list(order) if order else list(range(len(doc.pages)))
    if order is None and params.auto_reorder and doc.suggested_order:
        idxs = list(doc.suggested_order)
        res.warnings.append("Pages were re-ordered by their printed page numbers (SV-31).")
    res.order = idxs
    by_page: Dict[int, List[Candidate]] = {}
    for n, i in enumerate(idxs):
        if progress:
            progress(n / max(1, len(idxs)), f"Analysing page {i + 1} of {len(doc.pages)}")
        info = doc.pages[i]
        pr = PageResult(index=i, notes=list(info.notes))
        res.pages[i] = pr
        if info.dup_of is not None:
            pr.skipped = True
            by_page[i] = []
            continue
        fut = _POOL.submit(_process, doc, i, params)
        try:                                       # SV-60: one bad page never kills the run
            cands, pr.thumb, pr.scale = fut.result(timeout=params.page_timeout_s)
            by_page[i] = cands
        except FutTimeout:
            pr.error = f"Timed out after {params.page_timeout_s}s - send page to manual review"
            by_page[i] = []
        except Exception as e:
            pr.error = f"{type(e).__name__}: {e} - send page to manual review"
            by_page[i] = []
        if pr.error:
            res.warnings.append(f"Page {i + 1}: {pr.error}")
    by_page = merge_split_signatures(by_page)
    for i in idxs:
        res.candidates += by_page.get(i, [])
    for i, c in enumerate(res.candidates):
        c.id = f"p{c.page + 1}-c{i + 1}"
    _flag_repeats(res, params)
    if progress:
        progress(1.0, "Done")
    res.elapsed = time.time() - t0
    return res


def _flag_repeats(res: Analysis, params):
    """SV-28/45: identical signature repeated in different places of the file -> pasted copy suspected."""
    sigs = [c for c in res.candidates if c.kind == "signature" and c.included]
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            A, B = sigs[a], sigs[b]
            if min(A.bbox[3], B.bbox[3]) < 20 or min(A.bbox[2], B.bbox[2]) < 50:
                continue
            if identity_iou(A.mask, B.mask) >= params.copy_iou:
                for c, o in ((A, B), (B, A)):
                    if "pixel_identical" not in c.flags:
                        c.flags.append("pixel_identical")
                    c.notes.append(f"Pixel-identical to {o.id}: genuine signatures are never exactly the same - pasted copy suspected (SV-45)")


def manual_candidate(doc: Document, page: int, bbox, params) -> Candidate:
    """SV-27: user draws/enters a region when automatic detection missed the signature."""
    img = doc.render(page)
    x, y, w, h = [int(v) for v in bbox]
    sub = preprocess(img)[2][y:y + h, x:x + w]
    if (sub > 0).sum() < 30:
        sub = preprocess(img, faint=True)[2][y:y + h, x:x + w]       # faint ink: more sensitive pass
    ys, xs = np.where(sub > 0)
    if len(xs) < 30:
        raise ValueError("No ink in the selected region.")
    m = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = img[y:y + h, x:x + w].copy()
    c = Candidate(id=f"p{page + 1}-m", page=page, bbox=(x + int(xs.min()), y + int(ys.min()), m.shape[1], m.shape[0]), mask=m,
                  crop=crop, kind="signature", score=1.0, role_conf="manual", quality=0.8)
    c.notes.append("Region added manually (SV-27)")
    return c
