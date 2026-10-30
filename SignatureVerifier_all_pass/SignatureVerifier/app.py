"""Signature Verification Console - offline Streamlit app (no external APIs).
Start:  streamlit run app.py"""
import hashlib
import io
import json
import os
import sys
import time
import warnings
from datetime import date

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.deps import ensure as _ensure
_ensure("cryptography")          # needed for PDF digital-signature checks (SV-36/37)
from core import loader, quick, references
from core.config import ALL_ROLES, CUSTOMER_ROLES, Params
from core.pipeline import analyze, manual_candidate
from core.references import DATA, Library, audit, extract_specimen, load_audit, load_queue, queue_add, save_queue
from core.scenarios import load_scenarios
from core.verify import verify_document

st.set_page_config(page_title="Signature Verification Console", page_icon="✍️", layout="wide")

COLORS = {"MATCH": "#1b8a5a", "PASS": "#1b8a5a", "MISMATCH": "#c0392b", "REJECT / ESCALATE": "#c0392b", "SUSPECT": "#8e1b4d",
          "REVIEW": "#d68910", "MANUAL REVIEW": "#d68910", "NO_REFERENCE": "#d68910", "EXCLUDED": "#5d6d7e",
          "INCOMPLETE": "#b9770e", "NO SIGNATURE": "#b9770e"}
st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1400px}
.banner{border-radius:12px;padding:18px 24px;color:#fff;margin-bottom:14px}
.banner h2{margin:0;color:#fff;font-size:1.6rem}.banner p{margin:4px 0 0 0;opacity:.92}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;color:#fff;font-weight:600;font-size:.82rem}
.flag{display:inline-block;padding:1px 9px;border-radius:6px;background:#eef1f5;color:#34495e;font-size:.75rem;margin:0 4px 4px 0;border:1px solid #d5dbe3}
.card{border:1px solid #e1e5ea;border-radius:12px;padding:14px;margin-bottom:12px;background:rgba(127,127,127,.04)}
[data-testid="stMetricValue"]{font-size:1.5rem}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("params", Params())
ss.setdefault("jobs", {})
ss.setdefault("seen_sha", {})
ss.setdefault("vcache", {})
ss.setdefault("selftest", None)
lib = Library()
P: Params = ss.params


def badge(v, text=None):
    labels = {"MATCH": "APPROVED", "PASS": "APPROVED",
              "MISMATCH": "NOT APPROVED", "SUSPECT": "NOT APPROVED",
              "REJECT / ESCALATE": "NOT APPROVED", "REVIEW": "MANUAL REVIEW",
              "MANUAL REVIEW": "MANUAL REVIEW", "NO_REFERENCE": "MANUAL REVIEW",
              "INCOMPLETE": "INCOMPLETE", "NO SIGNATURE": "NO SIGNATURE"}
    return f"<span class='badge' style='background:{COLORS.get(v, '#5d6d7e')}'>{text or labels.get(v, v)}</span>"


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_img(mask):
    return 255 - mask


# ======================================================================== sidebar
with st.sidebar:
    st.markdown("## ✍️ Signature Verification")
    st.caption("Offline · deterministic · audit-ready")
    page = st.radio("Navigate", ["Quick check", "Verify document", "Specimen library", "Review queue", "Scenario coverage",
                                 "Self-test", "Settings", "Audit log"], label_visibility="collapsed")
    st.divider()
    ss.user = st.text_input("Reviewer name", ss.get("user", "reviewer"))
    st.caption(f"{len(lib.customers())} customers · {len([x for x in load_queue() if x['status'] == 'open'])} open reviews")


# ======================================================================== library
def page_library():
    st.title("Specimen library")
    st.caption("Bank reference signatures. A customer may have several specimens (SV-41); dates are used to spot changed signatures (SV-42).")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Add specimen")
        cid = st.text_input("Customer ID", placeholder="e.g. CUST1001").strip().upper()
        name = st.text_input("Customer name")
        dt = st.date_input("Specimen date", date.today())
        files = st.file_uploader("Specimen images / PDFs", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"], accept_multiple_files=True)
        if st.button("Register specimen(s)", type="primary", disabled=not (cid and files)):
            lib.add_customer(cid, name or cid)
            for f in files:
                try:
                    mask, crop, q, notes = extract_specimen(f.name, f.getvalue(), P)
                    low = q < P.ref_min_quality
                    for other, sc in lib.lookalikes(mask, exclude=cid, thresh=P.accept_threshold):
                        st.warning(f"{f.name}: very similar to specimen of {other} (score {sc:.0f}) - look-alike risk (SV-44).")
                    if low:
                        st.warning(f"{f.name}: specimen quality {q:.2f} is low - decisions against it will need review (SV-39).")
                    lib.add_specimen(cid, mask, crop, q, dt, "; ".join(notes), low)
                    st.success(f"{f.name}: registered (quality {q:.2f}).")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
        st.divider()
        if st.button("Load demo data (4 customers + 10 sample documents)"):
            from tools import make_samples
            with st.spinner("Generating..."):
                make_samples.build_library(lib)
                make_samples.build_documents()
            st.success("Demo data created. Sample documents are in data/samples/ - upload them on the Verify page.")
            st.rerun()
    with c2:
        st.subheader("Registered customers")
        if not lib.customers():
            st.info("No customers yet. Add a specimen or load the demo data.")
        for cid in lib.customers():
            with st.expander(f"{cid} - {lib.name(cid)}  ({len(lib.specimens(cid))} specimens)"):
                cols = st.columns(4)
                for i, s in enumerate(lib.specimens(cid)):
                    with cols[i % 4]:
                        st.image(rgb(s["crop"]), caption=f"{s['date']} · q={s['quality']}" + (" ⚠ low" if s["low_quality"] else ""))
                        if st.button("Delete", key=f"ds_{cid}_{s['file']}"):
                            lib.delete_specimen(cid, s["file"]); st.rerun()
                if st.button("Delete customer", key=f"dc_{cid}"):
                    lib.delete_customer(cid); st.rerun()


# ======================================================================== quick check
def _show(img_bgr, caption):
    st.markdown(f"**{caption}**")
    try:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width="stretch")
    except TypeError:                                   # older Streamlit versions
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)


def _best_quick(pages, refs, strict):
    """Run the quick-check mechanism on every page against every specimen; return (result, page_index, specimen_name)."""
    best = None
    for pi, pg in enumerate(pages):
        for rname, rimg in refs:
            try:
                r = quick.verify(pg, rimg, approve_score=strict)
            except Exception as e:
                st.warning(f"Page {pi + 1}: {type(e).__name__}: {e}")
                continue
            if r.doc_clean is not None and (best is None or r.score > best[0].score):
                best = (r, pi, rname)
    return best


def _render_quick(r, pi, npages, rname, strict):
    if r is None:
        st.markdown("<div class='banner' style='background:#d68910'><h2>NO SIGNATURE FOUND</h2>"
                    "<p>No pen signature could be found in the document. Try a clearer scan.</p></div>", unsafe_allow_html=True)
        return
    col = "#1b8a5a" if r.approved else "#c0392b"
    st.markdown(f"<div class='banner' style='background:{col}'><h2>{'✅ ' if r.approved else '❌ '}{r.verdict}</h2>"
                f"<p>{' '.join(r.reasons)}</p></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Similarity", f"{r.score:.0f} / 100")
    m2.metric("Approval level", f"{strict}")
    m3.metric("Page used", f"{pi + 1} of {npages}")
    left, right = st.columns(2)
    with left:
        _show(r.original, "1. Original document (green box = signature found)")
        _show(r.doc_crop, "2. Cropped document signature")
    with right:
        _show(r.doc_clean, "3. Cleaned document signature")
        _show(r.ref_clean, "4. Cleaned registered signature")
    with st.expander("Overlay of both signatures and detail scores"):
        st.image(cv2.cvtColor(r.overlay, cv2.COLOR_BGR2RGB), caption="Red = only in document · Blue = only in registered · Dark = overlap")
        st.json({"registered specimen used": rname, **r.metrics})


def page_quick():
    st.title("Quick signature check")
    st.caption("Upload the document → only the pen signature is cut out and cleaned → the registered signature is cleaned the same way → "
               "both are compared. Same = **APPROVED**, different = **NOT APPROVED**.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Registered signature")
        src = st.radio("Where is the registered signature?", ["Registered customer (library)", "Upload a registered signature"],
                       horizontal=True, label_visibility="collapsed")
        refs = []
        if src.startswith("Registered"):
            cust = lib.customers()
            if not cust:
                st.info("No customers registered yet. Add one in **Specimen library**, or choose *Upload a registered signature*.")
            else:
                cid = st.selectbox("Customer", cust, format_func=lambda x: f"{x} - {lib.name(x)}")
                refs = [(f"{cid} · {sp['date']}", sp["crop"]) for sp in lib.specimens(cid)]
                if refs:
                    st.image(cv2.cvtColor(refs[-1][1], cv2.COLOR_BGR2RGB), caption="Registered specimen (latest)", width=260)
        else:
            rf = st.file_uploader("Registered signature (image or PDF)", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"], key="q_ref")
            if rf:
                try:
                    refs = [(rf.name, quick.load_page_images(rf.name, rf.getvalue())[0])]
                    st.image(cv2.cvtColor(refs[0][1], cv2.COLOR_BGR2RGB), width=260)
                except Exception as e:
                    st.error(f"{rf.name}: {e}")
    with c2:
        st.subheader("Document to verify")
        df = st.file_uploader("Document (PDF or image) - e.g. the loan application form", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"], key="q_doc")
        strict = st.slider("Approval level (similarity needed to APPROVE)", 40, 85, int(quick.APPROVE_SCORE),
                           help="Higher = stricter. Genuine signatures usually score 60-90, different people 20-45.")

    if not (refs and df):
        st.info("Choose the registered signature and upload the document.")
        return
    try:
        pages = quick.load_page_images(df.name, df.getvalue(), dpi=170)
    except Exception as e:
        st.error(f"{df.name}: could not be read - {e}")
        return
    best = None
    with st.spinner("Finding the pen signature, cleaning and comparing..."):
        for pi, pg in enumerate(pages):
            for rname, rimg in refs:
                try:
                    r = quick.verify(pg, rimg, approve_score=float(strict))
                except Exception as e:
                    r = None
                    st.warning(f"Page {pi + 1}: {type(e).__name__}: {e}")
                if r is not None and r.doc_clean is not None and (best is None or r.score > best[0].score):
                    best = (r, pi, rname)
    if best is None:
        st.markdown("<div class='banner' style='background:#d68910'><h2>NO SIGNATURE FOUND</h2>"
                    "<p>No pen signature could be found in the document. Try a clearer scan.</p></div>", unsafe_allow_html=True)
        return
    r, pi, rname = best
    col = "#1b8a5a" if r.approved else "#c0392b"
    st.markdown(f"<div class='banner' style='background:{col}'><h2>{'✅ ' if r.approved else '❌ '}{r.verdict}</h2>"
                f"<p>{' '.join(r.reasons)}</p></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Similarity", f"{r.score:.0f} / 100")
    m2.metric("Approval level", f"{strict}")
    m3.metric("Page used", f"{pi + 1} of {len(pages)}")
    left, right = st.columns(2)
    with left:
        _show(r.original, "1. Original document (green box = signature found)")
        _show(r.doc_crop, "2. Cropped document signature")
    with right:
        _show(r.doc_clean, "3. Cleaned document signature")
        _show(r.ref_clean, "4. Cleaned registered signature")
    with st.expander("Overlay of both signatures and detail scores"):
        st.image(cv2.cvtColor(r.overlay, cv2.COLOR_BGR2RGB), caption="Red = only in document · Blue = only in registered · Dark = overlap")
        st.json({"registered specimen used": rname, **r.metrics})
    audit(ss.user, "quick_check", df.name, rname, r.verdict, f"score {r.score:.0f}")


# ======================================================================== verify
def page_verify():
    st.title("Verify document")
    cust = [""] + lib.customers()
    with st.expander("1 · Signers and required signatures", expanded=True):
        roles = st.multiselect("Signer roles on this document", list(CUSTOMER_ROLES), default=["borrower", "guarantor"])
        assign, policy = {}, {}
        cols = st.columns(max(1, min(len(roles), 4)))
        for i, r in enumerate(roles):
            with cols[i % len(cols)]:
                label = r.replace("_", " ").title()
                cid = st.selectbox(f"{label} - customer", cust, key=f"as_{r}", format_func=lambda x: f"{x} - {lib.name(x)}" if x else "(select)")
                n = st.number_input(f"{label} - required signatures", 0, 5, 1, key=f"rq_{r}")
                if cid:
                    assign[r] = cid
                if n:
                    policy[r] = int(n)
        st.caption("Witness / bank-officer signatures are recognised automatically and excluded from customer comparison (SV-58).")
    with st.expander("2 · Upload", expanded=True):
        files = st.file_uploader("PDF or image files", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
        c1, c2 = st.columns(2)
        combine = c1.checkbox("Treat several image files as pages of ONE document", value=False)
        pwd = c2.text_input("PDF password (if protected)", type="password")
    strict = st.slider("Approval level (similarity needed to APPROVE)", 40, 85, int(quick.APPROVE_SCORE),
                       help="Higher = stricter. Genuine signatures usually score 60-90, different people 20-45.")
    if st.button("Analyse & verify", type="primary", disabled=not files):
        if not assign:
            st.error("Select the registered customer for at least one signer role in step 1.")
        else:
            out = []
            for f in files:
                try:
                    pages = quick.load_page_images(f.name, f.getvalue(), dpi=170)
                except Exception as e:
                    st.error(f"{f.name}: could not be processed - {e}  (SV-32)")
                    continue
                with st.spinner(f"{f.name}: finding the pen signature, cleaning and comparing..."):
                    for role, cid in assign.items():
                        refs = [(f"{cid} · {sp['date']}", sp["crop"]) for sp in lib.specimens(cid)]
                        if not refs:
                            out.append({"file": f.name, "role": role, "cid": cid, "best": None, "npages": len(pages), "noref": True})
                            continue
                        best = _best_quick(pages, refs, float(strict))
                        out.append({"file": f.name, "role": role, "cid": cid, "best": best, "npages": len(pages), "noref": False})
                        audit(ss.user, "quick_verify", f.name, f"{role}:{cid}", best[0].verdict if best else "NO SIGNATURE FOUND",
                              f"score {best[0].score:.0f}" if best else "")
            ss.vresults = out
    if not ss.get("vresults"):
        st.info("Choose the registered customer for each signer, upload the document and click **Analyse & verify**.")
        return
    res = ss.vresults
    verdicts = [("NO REFERENCE" if x["noref"] else (x["best"][0].verdict if x["best"] else "NO SIGNATURE FOUND")) for x in res]
    if all(v == "APPROVED" for v in verdicts):
        st.success("DOCUMENT APPROVED - every selected signer's signature matches the registered signature.")
    elif any(v == "NOT APPROVED" for v in verdicts):
        st.error("DOCUMENT NOT APPROVED - at least one signature does not match the registered signature.")
    for x in res:
        st.divider()
        st.subheader(f"{x['file']} · {x['role'].replace('_', ' ').title()} ({x['cid']} - {lib.name(x['cid'])})")
        if x["noref"]:
            st.warning("No specimen signature registered for this customer. Add one in Specimen library.")
        elif not x["best"]:
            _render_quick(None, 0, x["npages"], "", strict)
        else:
            r, pi, rname = x["best"]
            _render_quick(r, pi, x["npages"], rname, strict)


def get_verdict(key, job):
    an, doc = job["an"], job["doc"]
    sig = (key, tuple((c.id, c.role, c.included) for c in an.candidates), json.dumps(job["assign"], sort_keys=True),
           json.dumps(job["policy"], sort_keys=True), json.dumps(P.as_dict(), sort_keys=True), len(lib.customers()),
           sum(len(lib.index[c]["specimens"]) for c in lib.customers()))
    if sig not in ss.vcache:
        ss.vcache = {sig: verify_document(an, lib, job["assign"], job["policy"], P, doc.digital_sigs)}
    return ss.vcache[sig]


def render_job(key, job):
    doc, an = job["doc"], job["an"]
    sigs_all = [c for c in an.candidates if c.kind in ("signature", "thumb")]
    for c in sigs_all:                                     # apply user overrides before verification
        c.role = st.session_state.get(f"role_{key}_{c.id}", c.role)
        c.included = st.session_state.get(f"inc_{key}_{c.id}", c.included)
    for c in an.candidates:
        if c.kind not in ("signature", "thumb"):
            c.included = st.session_state.get(f"inc_{key}_{c.id}", False)
            if c.included:
                c.kind = "signature"
    dv = get_verdict(key, job)
    doc_label = {"PASS": "APPROVED", "REJECT / ESCALATE": "NOT APPROVED",
                 "MANUAL REVIEW": "MANUAL REVIEW", "INCOMPLETE": "INCOMPLETE",
                 "NO SIGNATURE": "NO SIGNATURE"}.get(dv.verdict, dv.verdict)
    st.markdown(f"<div class='banner' style='background:{COLORS.get(dv.verdict, '#5d6d7e')}'><h2>{doc_label}</h2><p>{dv.headline}</p></div>", unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("Pages", len(doc.pages)); m[1].metric("Signatures", len([r for r in dv.results]))
    m[2].metric("Match", dv.summary.get("MATCH", 0)); m[3].metric("Review", dv.summary.get("REVIEW", 0) + dv.summary.get("NO_REFERENCE", 0))
    m[4].metric("Mismatch / suspect", dv.summary.get("MISMATCH", 0) + dv.summary.get("SUSPECT", 0)); m[5].metric("Time", f"{an.elapsed:.1f}s")
    msgs = list(an.warnings) + dv.warnings
    if msgs:
        with st.expander(f"⚠ {len(msgs)} document warning(s)", expanded=True):
            for w in msgs:
                st.write("• " + w)
    if doc.digital_sigs:
        with st.expander("Digital (cryptographic) signatures"):
            st.dataframe(pd.DataFrame([{"Signer": s["signer"], "Valid from": s["valid_from"], "Valid to": s["valid_to"], "OK": s["ok"],
                                        "Problems": "; ".join(s["problems"])} for s in doc.digital_sigs]), use_container_width=True)
    elif doc.is_pdf:
        st.caption("No cryptographic signature in this PDF: visible signature images cannot be verified cryptographically (SV-35).")
    t1, t2, t3, t4 = st.tabs(["Signatures", "Pages", "Add region manually", "Export"])
    res = {r.cand_id: r for r in dv.results}
    with t1:
        if not sigs_all:
            st.info("No signature detected. See 'Other marks' below or add a region manually (SV-26/27).")
        for c in sigs_all:
            r = res.get(c.id)
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                a, b, d = st.columns([1.2, 1.5, 1.2])
                with a:
                    st.image(rgb(c.crop), caption=f"{c.id} · page {c.page + 1} · {c.source} · {c.ink} ink", use_container_width=True)
                with b:
                    if r:
                        st.markdown(badge(r.verdict), unsafe_allow_html=True)
                        for why in r.reasons:
                            st.write("• " + why)
                    for n in c.notes:
                        st.caption("ℹ " + n)
                    st.markdown("".join(f"<span class='flag'>{f}</span>" for f in dict.fromkeys(c.flags)), unsafe_allow_html=True)
                    if c.role_label:
                        st.caption(f"Detected field: “{c.role_label}”")
                with d:
                    st.selectbox("Signer role", list(ALL_ROLES), index=list(ALL_ROLES).index(c.role), key=f"role_{key}_{c.id}")
                    st.checkbox("Count as signature", value=c.included, key=f"inc_{key}_{c.id}")
                    if r and r.best_specimen is not None:
                        st.image(rgb(r.best_specimen["crop"]), caption=f"Best specimen of {r.customer} ({r.best_specimen['date']})", use_container_width=True)
                    if r and st.button("Send to manual review", key=f"q_{key}_{c.id}"):
                        os.makedirs(os.path.join(DATA, "review"), exist_ok=True)
                        pth = os.path.join(DATA, "review", f"{doc.sha[:10]}_{c.id}.png")
                        cv2.imwrite(pth, c.crop)
                        queue_add(doc.name, doc.sha, c.id, c.page, "; ".join(r.reasons) or r.verdict, r.score, pth)
                        audit(ss.user, "queued", doc.name, c.id, r.verdict)
                        st.toast("Added to review queue")
                st.markdown("</div>", unsafe_allow_html=True)
        others = [c for c in an.candidates if c.kind not in ("signature", "thumb")]
        if others:
            with st.expander(f"Other marks ({len(others)}): stamps, logos, printed or handwritten text - excluded by default"):
                cols = st.columns(5)
                for i, c in enumerate(others):
                    with cols[i % 5]:
                        st.image(rgb(c.crop), caption=f"{c.id} · {c.kind}")
                        st.checkbox("It IS a signature", key=f"inc_{key}_{c.id}")
    with t2:
        pidx = st.select_slider("Page", options=[i + 1 for i in an.order], value=an.order[0] + 1) - 1 if len(an.order) > 1 else an.order[0]
        pr, info = an.pages[pidx], doc.pages[pidx]
        st.caption(f"{info.orientation} · {info.kind} · {info.width}×{info.height}px" + ("  ·  " + " | ".join(info.notes) if info.notes else ""))
        if pr.error:
            st.error(pr.error)
        if pr.thumb:
            im = cv2.imdecode(np.frombuffer(pr.thumb, np.uint8), 1)
            for c in an.candidates:
                if c.page != pidx:
                    continue
                x, y, w, h = [int(v * pr.scale) for v in c.bbox]
                r = res.get(c.id)
                col = {"MATCH": (90, 138, 27), "MISMATCH": (43, 57, 192), "SUSPECT": (77, 27, 142), "REVIEW": (9, 134, 214)}.get(r.verdict if r else "", (160, 160, 160))
                cv2.rectangle(im, (x, y), (x + w, y + h), col, 2)
                cv2.putText(im, c.id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            st.image(rgb(im), use_container_width=True)
        b1, b2, b3 = st.columns(3)
        if b1.button("Open full-resolution page"):
            try:
                st.image(rgb(doc.render(pidx)), use_container_width=True)
            except Exception as e:
                st.warning(f"Original page cannot be retrieved ({e}); showing the stored copy instead (SV-64).")
        deg = b2.selectbox("Rotate page by", [90, 180, 270], key="rotdeg")
        if b3.button("Rotate & re-analyse this document"):
            rotate_doc_page(doc, pidx, deg)
            job["an"] = analyze(doc, P, order=an.order)
            st.rerun()
    with t3:
        pg = st.number_input("Page", 1, len(doc.pages), 1) - 1
        W, H = max(1, doc.pages[pg].width), max(1, doc.pages[pg].height)
        c = st.columns(4)
        x = c[0].number_input("x", 0, W, 0); y = c[1].number_input("y", 0, H, 0)
        w = c[2].number_input("width", 1, W, min(400, W)); h = c[3].number_input("height", 1, H, min(200, H))
        st.caption(f"Pixel coordinates on a {W}×{H} render of the page (use the Pages tab to locate the signature).")
        if st.button("Add region as signature"):
            try:
                nc = manual_candidate(doc, pg, (x, y, w, h), P)
                nc.id = f"p{pg + 1}-m{len(an.candidates) + 1}"
                an.candidates.append(nc)
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with t4:
        rows = [{"document": doc.name, "candidate": r.cand_id, "page": r.page + 1, "role": r.role, "customer": r.customer, "verdict": r.verdict,
                 "score": r.score, "reasons": " | ".join(r.reasons)} for r in dv.results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button("Download CSV", pd.DataFrame(rows).to_csv(index=False), f"{doc.name}_result.csv")
        st.download_button("Download JSON", json.dumps({"document": doc.name, "sha256": doc.sha, "verdict": dv.verdict, "headline": dv.headline,
                                                         "results": rows, "warnings": msgs, "params": P.as_dict()}, indent=1), f"{doc.name}_result.json")


def rotate_doc_page(doc, i, deg):
    if doc.is_pdf:
        p = doc._pdf[i]
        p.set_rotation((p.rotation + deg) % 360)
        doc.pages[i].words = []
        doc.pages[i].width, doc.pages[i].height = int(p.rect.width * doc.dpi / 72), int(p.rect.height * doc.dpi / 72)
    else:
        doc._imgs[i] = cv2.rotate(doc._imgs[i], {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg])
        doc.pages[i].height, doc.pages[i].width = doc._imgs[i].shape[:2]
    doc.pages[i].notes.append(f"Manually rotated by {deg} deg (SV-03)")


# ======================================================================== review queue
def page_review():
    st.title("Manual review queue")
    st.caption("Signatures that could not be decided automatically (SV-63). Decisions are written to the audit log.")
    q = load_queue()
    open_items = [x for x in q if x["status"] == "open"]
    if not open_items:
        st.success("Queue is empty.")
    for i, it in enumerate(q):
        if it["status"] != "open":
            continue
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            a, b = st.columns([1, 2])
            with a:
                if os.path.exists(it["crop"]):
                    st.image(it["crop"], caption=f"{it['cand']} · page {it['page'] + 1}")
                else:
                    st.warning("Stored crop missing - ask for the original document (SV-64).")
            with b:
                st.markdown(f"**{it['doc']}** · score {it['score']:.0f} · queued {it['created']}")
                st.write(it["reason"])
                dec = st.radio("Decision", ["Genuine (approve)", "Forged / reject", "Need original document"], key=f"dec{i}", horizontal=True)
                note = st.text_input("Note", key=f"note{i}")
                if st.button("Save decision", key=f"sv{i}"):
                    it["status"], it["decision"], it["note"] = "closed", dec, note
                    save_queue(q)
                    audit(ss.user, "review-decision", it["doc"], it["cand"], dec, note)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ======================================================================== coverage / selftest / settings / audit
def page_coverage():
    st.title("Scenario coverage (SV-01 … SV-64)")
    rows = load_scenarios()
    res = {r[0]: r for r in (ss.selftest or [])}
    for r in rows:
        r["Self-test"] = ("✅ pass" if res[r["ID"]][1] else "❌ fail") if r["ID"] in res else "—"
    q = st.text_input("Filter")
    df = pd.DataFrame(rows)
    if q:
        df = df[df.apply(lambda x: q.lower() in " ".join(map(str, x)).lower(), axis=1)]
    st.dataframe(df, use_container_width=True, height=620, hide_index=True)
    st.caption("Run the **Self-test** page to execute automatic checks for each scenario with generated documents.")


def page_selftest():
    st.title("Self-test")
    st.caption("Generates synthetic documents (stamps, blur, rotation, forged and pasted signatures, corrupt and encrypted files …) and checks the behaviour for every scenario. Takes about a minute.")
    if st.button("Run self-tests", type="primary"):
        from tests import run_selftests
        with st.spinner("Running 64 scenario checks…"):
            ss.selftest = run_selftests.run_all()
    if ss.selftest:
        ok = sum(r[1] for r in ss.selftest)
        st.metric("Passed", f"{ok} / {len(ss.selftest)}")
        st.dataframe(pd.DataFrame([{"Scenario": r[0], "Result": "PASS" if r[1] else "FAIL", "Seconds": r[3], "Detail": r[2][:300]} for r in ss.selftest]),
                     use_container_width=True, height=600, hide_index=True)


def page_settings():
    st.title("Settings")
    p = ss.params
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Decision thresholds")
        p.accept_threshold = st.slider("Accept (MATCH) at or above", 50.0, 95.0, float(p.accept_threshold), 1.0)
        p.reject_threshold = st.slider("Reject (MISMATCH) at or below", 10.0, 70.0, float(p.reject_threshold), 1.0)
        st.caption("Scores between the two limits go to manual review (SV-51). Raise 'accept' to reduce false accepts (SV-52); lower 'reject' to reduce false rejects (SV-53).")
        p.copy_iou = st.slider("Pasted-copy sensitivity (pixel identity)", 0.85, 1.0, float(p.copy_iou), 0.01)
        p.min_quality = st.slider("Minimum image quality for automatic decisions", 0.0, 0.8, float(p.min_quality), 0.05)
    with c2:
        st.subheader("Processing")
        p.dpi = st.select_slider("PDF render DPI", [100, 150, 200, 300], int(p.dpi))
        p.max_pages = st.number_input("Max pages per file", 1, 2000, int(p.max_pages))
        p.page_timeout_s = st.number_input("Per-page timeout (s)", 5, 600, int(p.page_timeout_s))
        p.auto_rotate = st.checkbox("Auto-rotate sideways pages", p.auto_rotate)
        p.auto_reorder = st.checkbox("Auto-reorder pages by printed page number", p.auto_reorder)
        p.min_sig_score = st.slider("Detector sensitivity (lower = more candidates)", 0.2, 0.8, float(p.min_sig_score), 0.05)
    if st.button("Reset to defaults"):
        ss.params = Params(); st.rerun()


def page_audit():
    st.title("Audit log")
    rows = load_audit()
    if rows:
        df = pd.DataFrame(rows).iloc[::-1]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "audit_log.csv")
    else:
        st.info("No entries yet.")


{"Quick check": page_quick, "Verify document": page_verify, "Specimen library": page_library, "Review queue": page_review,
 "Scenario coverage": page_coverage, "Self-test": page_selftest, "Settings": page_settings, "Audit log": page_audit}[page]()
