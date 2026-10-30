"""Creates demo customers (specimens) and sample documents.
Run:  python tools/make_samples.py       (or press 'Load demo data' in the app)"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from core import references
from core.config import Params
from core.references import DATA, Library
from core.synth import (blur, faint, make_pdf, noise, png_bytes, render_signature, shadow, to_scanned)

OUT = os.path.join(DATA, "samples")
P = Params()
CUSTOMERS = {"ALICE": ("Alice Rao", 1), "BOB": ("Bob Menon", 2), "CAROL": ("Carol Iyer", 3), "DAVE": ("Dave Nair", 4)}


def _join(*pdfs):
    out = fitz.open()
    for b in pdfs:
        out.insert_pdf(fitz.open(stream=b, filetype="pdf"))
    return out.tobytes()


def build_library(lib: Library):
    for cid, (name, seed) in CUSTOMERS.items():
        if lib.specimens(cid):
            continue
        lib.add_customer(cid, name)
        for v, date in ((1, "2021-03-01"), (2, "2024-06-15")) if cid == "ALICE" else ((1, "2022-01-10"),):
            img = render_signature(seed, v)
            card = make_pdf([dict(seed=seed, paste=img, pos=(60, 420))], extra_text=False, title=f"SPECIMEN CARD - {name}")
            mask, crop, q, _ = references.extract_specimen("card.pdf", card, P)
            lib.add_specimen(cid, mask, crop, q, date)


def build_documents():
    os.makedirs(OUT, exist_ok=True)
    A = lambda lbl, seed, v, x=60, y=420, **k: dict(label=lbl, seed=seed, variant=v, pos=(x, y), **k)
    docs = {
        "01_valid_loan_alice_bob.pdf": make_pdf([A("Borrower", 1, 5), A("Guarantor", 2, 5, 330)]),
        "02_forged_guarantor.pdf": make_pdf([A("Borrower", 1, 6), A("Guarantor", 3, 6, 330)]),
        "03_pasted_copy_of_specimen.pdf": make_pdf([dict(label="Borrower", seed=1, pos=(60, 420), paste=render_signature(1, 1)), A("Guarantor", 2, 7, 330)]),
        "04_scan_with_red_stamp_and_table.pdf": to_scanned(make_pdf([A("Borrower", 1, 8), A("Guarantor", 2, 8, 330)], table=True, stamp_at=(120, 440))),
        "05_missing_guarantor.pdf": make_pdf([A("Borrower", 1, 9)]),
        "06_landscape.pdf": make_pdf([A("Borrower", 1, 10), A("Guarantor", 2, 10, 450)], landscape=True),
        "07_faint_scan.pdf": to_scanned(make_pdf([A("Borrower", 1, 11), A("Guarantor", 2, 11, 330)]), degrade=faint),
        "08_blurred_shadow_scan.pdf": to_scanned(make_pdf([A("Borrower", 1, 12), A("Guarantor", 2, 12, 330)]),
                                                 degrade=lambda a: noise(shadow(cv2.GaussianBlur(a, (0, 0), 1.2)))),
        "09_two_pages_wrong_order.pdf": _join(make_pdf([A("Borrower", 1, 13), A("Guarantor", 2, 13, 330)], page_no=(2, 2), title="SCHEDULE B"),
                                              make_pdf([], page_no=(1, 2), title="LOAN AGREEMENT")),
        "10_witness_and_officer.pdf": make_pdf([A("Borrower", 1, 14), A("Witness", 4, 3, 330), A("Bank Officer", 3, 3, 60, 560)]),
    }
    for n, d in docs.items():
        open(os.path.join(OUT, n), "wb").write(d)
    return list(docs)


def main():
    build_library(Library())
    names = build_documents()
    print("Specimens registered for:", ", ".join(CUSTOMERS))
    print("Sample documents written to", OUT)
    for n in names:
        print("  -", n)


if __name__ == "__main__":
    main()
