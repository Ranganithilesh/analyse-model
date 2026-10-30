"""Document loading: PDF / images, orientation, page kind, digital-signature inspection.
Covers SV-01..06, 29-32, 33-37, 61, 64."""
import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np
try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz
from PIL import Image, ImageSequence

ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class DocumentError(Exception):
    """File is corrupted / unreadable / unsupported (SV-32)."""


class PasswordRequired(DocumentError):
    """File is password protected (SV-32)."""


@dataclass
class PageInfo:
    index: int
    width: int = 0
    height: int = 0
    orientation: str = "portrait"
    kind: str = "image"               # text | image | mixed | blank
    words: list = field(default_factory=list)   # (x0,y0,x1,y1,text,block,line) px
    img_rects: list = field(default_factory=list)
    img_xrefs: list = field(default_factory=list)
    label: Optional[int] = None
    total: Optional[int] = None
    rotation_applied: int = 0
    exact_hash: str = ""
    dup_of: Optional[int] = None
    text: str = ""
    notes: list = field(default_factory=list)


class Document:
    def __init__(self, name, data, is_pdf, dpi=150):
        self.name, self.data, self.is_pdf, self.dpi = name, data, is_pdf, dpi
        self.sha = hashlib.sha256(data).hexdigest()
        self.pages: List[PageInfo] = []
        self.warnings: List[str] = []
        self.digital_sigs: List[dict] = []
        self._pdf = None
        self._imgs: List[np.ndarray] = []
        self.suggested_order: Optional[List[int]] = None

    def render(self, i, dpi=None) -> np.ndarray:
        """Render page i as BGR array (lazy, so large PDFs do not fill memory - SV-61)."""
        dpi = dpi or self.dpi
        if self.is_pdf:
            pix = self._pdf[i].get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            return cv2.cvtColor(arr[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
        return self._imgs[i].copy()


# ---------------------------------------------------------------- orientation
def _upright_score(ink):
    """Ascenders outnumber descenders in upright Latin text -> upper part of a text band has more ink."""
    rows = ink.sum(1).astype(float)
    if rows.max() <= 0:
        return 0.0
    on = rows > 0.2 * rows.max()
    score, i, n = 0.0, 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            band = ink[i:j]
            if 6 <= band.shape[0] <= 80:
                t = band.shape[0] // 3
                score += (float(band[:t].sum()) - float(band[-t:].sum())) / (float(band.sum()) + 1)
            i = j
        else:
            i += 1
    return score


def estimate_image_rotation(gray) -> int:
    """Clockwise correction (0/90/180/270) for image-only pages (SV-03). Best effort; UI can override."""
    s = 800.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    ink = (cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12) > 0).astype(np.uint8)
    if ink.mean() < 0.004:
        return 0
    ratio = ink.sum(1).astype(float).var() / (ink.sum(0).astype(float).var() + 1e-9)
    if 0.8 < ratio < 1.25:
        return 0
    cands = (0, 180) if ratio >= 1 else (90, 270)
    sc = {a: _upright_score(ink if a == 0 else cv2.rotate(ink, ROT[a])) for a in cands}
    best = max(sc, key=sc.get)
    if cands == (0, 180):
        return 0      # upside-down scans are not auto-flipped (heuristic unreliable): comparator tests 180 deg, UI has a rotate control
    return best


def _text_rotation(page) -> int:
    """Clockwise correction from text direction vectors of a text PDF page."""
    votes = {0: 0, 90: 0, 180: 0, 270: 0}
    m = page.rotation_matrix
    mm = fitz.Matrix(m.a, m.b, m.c, m.d, 0, 0)
    for b in page.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            n = sum(len(sp["text"]) for sp in ln["spans"])
            d = fitz.Point(*ln["dir"]) * mm
            if abs(d.x) >= abs(d.y):
                votes[0 if d.x > 0 else 180] += n
            else:
                votes[90 if d.y < 0 else 270] += n
    return max(votes, key=votes.get) if sum(votes.values()) else 0


# ---------------------------------------------------------------- digital signatures
def inspect_digital_signatures(data: bytes):
    """SV-33..37: find cryptographic signatures, integrity, certificate validity."""
    out, trail = [], len(data.rstrip())
    matches = list(re.finditer(rb"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]", data))
    last_end = max([int(m.group(3)) + int(m.group(4)) for m in matches], default=0)
    for m in matches:
        a, b, c, d = (int(x) for x in m.groups())
        info = {"byte_range": (a, b, c, d), "problems": [], "signer": None, "valid_from": None, "valid_to": None}
        try:
            signed = data[a:a + b] + data[c:c + d]
            hexs = data[b:c].strip().strip(b"<>").strip()
            der = bytes.fromhex(hexs.decode("ascii"))
            if (c + d) == last_end and (c + d) < trail - 2:
                info["problems"].append("File was modified AFTER signing (extra bytes after signed range)")
            found = any(hashlib.new(h, signed).digest() in der for h in ("sha256", "sha1", "sha384", "sha512"))
            info["digest_found"] = found
            if not found:
                info["problems"].append("Signed digest could not be confirmed - content may have been altered")
            try:
                from cryptography.hazmat.primitives.serialization import pkcs7
                certs = pkcs7.load_der_pkcs7_certificates(der.rstrip(b"\x00") or der)
                now = datetime.now(timezone.utc)
                for ct in certs[:1]:
                    nb = getattr(ct, "not_valid_before_utc", None) or ct.not_valid_before.replace(tzinfo=timezone.utc)
                    na = getattr(ct, "not_valid_after_utc", None) or ct.not_valid_after.replace(tzinfo=timezone.utc)
                    info["signer"] = ct.subject.rfc4514_string()
                    info["valid_from"], info["valid_to"] = nb.date().isoformat(), na.date().isoformat()
                    if now > na:
                        info["problems"].append(f"Certificate EXPIRED on {na.date()}")
                    if now < nb:
                        info["problems"].append("Certificate not yet valid")
                    if ct.subject == ct.issuer:
                        info["problems"].append("Self-signed certificate (no trusted chain)")
            except Exception:
                info["problems"].append("Certificate could not be parsed")
        except Exception as e:
            info["problems"].append(f"Signature container unreadable ({type(e).__name__})")
        info["ok"] = not info["problems"]
        out.append(info)
    return out


# ---------------------------------------------------------------- loading
def _page_numbers(text):
    m = re.search(r"(?i)page\s*(\d+)\s*(?:of|/)\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(?i)\bpage\s*(?:no\.?)?\s*(\d+)\b", text)
    return (int(m.group(1)), None) if m else (None, None)


def _finish(doc: Document):
    """Cross-page checks: duplicates (SV-29), order (SV-31), merged documents (SV-30)."""
    seen = {}
    for p in doc.pages:
        if p.exact_hash in seen:
            p.dup_of = seen[p.exact_hash]
            p.notes.append(f"Exact duplicate of page {p.dup_of + 1} - skipped (SV-29)")
            doc.warnings.append(f"Page {p.index + 1} is an exact duplicate of page {p.dup_of + 1}; skipped.")
        else:
            seen[p.exact_hash] = p.index
    labels = [p.label for p in doc.pages]
    if all(l is not None for l in labels) and len(labels) > 1:
        if labels != sorted(labels):
            doc.suggested_order = sorted(range(len(labels)), key=lambda i: labels[i])
            doc.warnings.append(f"Pages appear to be in the wrong order (page numbers {labels}); "
                                "automatic re-ordering is available (SV-31).")
        if any(b == 1 and a != 1 for a, b in zip(labels, labels[1:])):
            doc.warnings.append("Page numbering restarts inside the file: several documents may be merged (SV-30).")
    sizes = {(p.width // 40, p.height // 40, p.orientation) for p in doc.pages if p.width}
    if len(sizes) > 1:
        doc.warnings.append("Pages differ in size/orientation: file may contain merged documents (SV-30). "
                            "Each page is analysed independently.")
    kinds = {p.kind for p in doc.pages}
    if "text" in kinds and "image" in kinds:
        doc.warnings.append("File mixes text pages and scanned-image pages (SV-30/05).")


def load_document(name: str, data: bytes, password: Optional[str] = None, dpi: int = 150,
                  max_pages: int = 500, auto_rotate: bool = True) -> Document:
    if not data:
        raise DocumentError("File is empty (0 bytes).")
    is_pdf = data[:1024].find(b"%PDF") != -1
    doc = Document(name, data, is_pdf, dpi)
    if is_pdf:
        try:
            pdf = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            raise DocumentError(f"PDF is damaged or unreadable: {e}")
        if pdf.needs_pass:
            if not password or not pdf.authenticate(password):
                raise PasswordRequired("PDF is password protected - enter the password.")
        if pdf.page_count == 0:
            raise DocumentError("PDF has no pages.")
        doc._pdf = pdf
        if pdf.page_count > max_pages:
            doc.warnings.append(f"Large PDF: only first {max_pages} of {pdf.page_count} pages processed (SV-61).")
        doc.digital_sigs = inspect_digital_signatures(data)
        for i in range(min(pdf.page_count, max_pages)):
            try:
                p = pdf[i]
                text = p.get_text()
                imgs = p.get_images(full=True)
                info = PageInfo(index=i, text=text)
                if auto_rotate:
                    rot = _text_rotation(p) if len(text.strip()) > 20 else 0
                    if not rot and len(text.strip()) <= 20 and imgs:
                        low = p.get_pixmap(dpi=100, alpha=False)
                        g = cv2.cvtColor(np.frombuffer(low.samples, np.uint8).reshape(low.height, low.width, low.n)[:, :, :3], cv2.COLOR_RGB2GRAY)
                        rot = estimate_image_rotation(g)
                    if rot:
                        p.set_rotation((p.rotation + rot) % 360)
                        info.rotation_applied = rot
                        info.notes.append(f"Page was rotated/upside-down; auto-corrected by {rot} deg (SV-03)")
                sc = dpi / 72.0
                rm = p.rotation_matrix
                info.width, info.height = int(p.rect.width * sc), int(p.rect.height * sc)
                info.orientation = "landscape" if info.width > info.height else "portrait"
                for w in p.get_text("words"):
                    r = fitz.Rect(w[:4]) * rm
                    info.words.append((r.x0 * sc, r.y0 * sc, r.x1 * sc, r.y1 * sc, w[4], w[5], w[6]))
                for im in imgs:
                    try:
                        for r in p.get_image_rects(im[0]):
                            r = r * rm
                            info.img_rects.append((r.x0 * sc, r.y0 * sc, r.x1 * sc, r.y1 * sc))
                            info.img_xrefs.append(im[0])
                    except Exception:
                        pass
                n = len(text.strip())
                info.kind = "mixed" if n > 20 and imgs else "text" if n > 20 else "image" if imgs else "blank"
                if info.kind == "image":
                    info.notes.append("Raster-only page, no text layer: role labels need manual input (SV-04)")
                elif info.kind == "mixed":
                    info.notes.append("Page has both text and raster images (SV-05)")
                info.label, info.total = _page_numbers(text)
                low = p.get_pixmap(dpi=72, alpha=False)
                info.exact_hash = hashlib.md5(low.samples).hexdigest()
                if re.search(r"(?i)digitally signed|electronically signed|e-?signed|docusign|esign", text):
                    info.notes.append("Electronic-signature text mark found (SV-34)")
                doc.pages.append(info)
            except Exception as e:  # SV-32 partially damaged page
                doc.warnings.append(f"Page {i + 1} could not be read ({type(e).__name__}); skipped.")
                doc.pages.append(PageInfo(index=i, kind="blank", notes=["unreadable"]))
        for s in doc.digital_sigs:
            for pr in s["problems"]:
                doc.warnings.append("Digital signature: " + pr + " (SV-36/37)")
    else:
        try:
            im = Image.open(io.BytesIO(data))
            frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)][:max_pages]
        except Exception as e:
            raise DocumentError(f"Not a readable PDF or image: {e}")
        for i, fr in enumerate(frames):
            arr = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
            info = PageInfo(index=i, kind="image")
            if auto_rotate:
                rot = estimate_image_rotation(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY))
                if rot:
                    arr = cv2.rotate(arr, ROT[rot])
                    info.rotation_applied = rot
                    info.notes.append(f"Image rotated by {rot} deg automatically (SV-03)")
            info.height, info.width = arr.shape[:2]
            info.orientation = "landscape" if info.width > info.height else "portrait"
            info.exact_hash = hashlib.md5(cv2.resize(arr, (64, 64)).tobytes()).hexdigest()
            info.notes.append("Scanned image: no text layer (SV-04)")
            doc._imgs.append(arr)
            doc.pages.append(info)
    _finish(doc)
    return doc


def images_to_document(items, dpi=150, auto_rotate=True) -> Document:
    """Several uploaded image files treated as the pages of one document (SV-31)."""
    arrs = []
    for name, data in items:
        d = load_document(name, data, dpi=dpi, auto_rotate=auto_rotate)
        arrs += [d.render(i) for i in range(len(d.pages))]
    big = Document("combined_images", b"".join(x[1] for x in items), False, dpi)
    big._imgs = arrs
    for i, a in enumerate(arrs):
        h, w = a.shape[:2]
        big.pages.append(PageInfo(index=i, width=w, height=h, orientation="landscape" if w > h else "portrait",
                                  exact_hash=hashlib.md5(cv2.resize(a, (64, 64)).tobytes()).hexdigest(),
                                  notes=["Scanned image: no text layer (SV-04)"]))
    _finish(big)
    return big
