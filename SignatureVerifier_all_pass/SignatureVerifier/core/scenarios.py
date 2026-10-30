"""Maps every scenario SV-01..SV-64 to the module/function that handles it and how."""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

H = {
 "SV-01": ("loader.load_document", "Page orientation detected from page size; landscape pages are rendered and analysed exactly like portrait."),
 "SV-02": ("loader.load_document", "Portrait is the default path; orientation is shown per page."),
 "SV-03": ("loader._text_rotation / estimate_image_rotation / compare", "Text PDFs: rotation from text direction. Sideways scans: auto-rotated. Upside-down scans: comparator also tests 180/90/270 deg; UI offers manual rotate."),
 "SV-04": ("loader / detector", "Image-only pages are analysed visually (no text layer needed). Role labels fall back to manual assignment."),
 "SV-05": ("loader / detector", "Mixed pages: text layer used for roles/printed-text removal, pixels used for signatures."),
 "SV-06": ("pipeline.analyze", "Every page is analysed; signatures keep their page number; page viewer jumps to them."),
 "SV-07": ("detector.detect_page", "All regions on a page are returned (up to a configurable limit), each verified separately."),
 "SV-08": ("detector.find_label / verify_document", "Role labels (account holder, guarantor...) read from the text layer; each role uses its own specimen."),
 "SV-09": ("detector.find_label / verify_document", "Borrower, co-borrower and guarantor are separate roles with separate customers."),
 "SV-10": ("detector._positional_roles / verify", "Unknown signer -> flagged, compared with all assigned customers, never auto-accepted; user assigns the role."),
 "SV-11": ("config.ROLE_PATTERNS", "Director and authorised signatory are distinct roles, policy can require both."),
 "SV-12": ("detector._detect", "Coloured stamps removed per pixel; ring stamps removed with crossing strokes restored; result flagged stamp_overlap -> review."),
 "SV-13": ("detector._find_rings / verify", "Hidden strokes cannot be recovered: signature flagged and automatic decision withheld."),
 "SV-14": ("detector._remove_lines", "Table lines removed, strokes crossing them restored; printed words masked via text layer; flagged."),
 "SV-15": ("detector.merge_split_signatures", "Halves touching adjacent page edges are stitched into one signature."),
 "SV-16": ("detector._detect", "Edge contact flagged cut_off; automatic decision withheld."),
 "SV-17": ("detector._detect", "Tiny signatures still detected, flagged too_small, quality reduced, manual review."),
 "SV-18": ("detector.detect_page", "Second high-sensitivity pass when nothing found; flagged faint; low quality -> review."),
 "SV-19": ("detector._detect", "Resolution independent scaling; blur measured (Laplacian); blurred -> review."),
 "SV-20": ("detector.preprocess", "Background/illumination normalisation removes shadows, folds, gradients."),
 "SV-21": ("detector._detect", "Candidates are ranked by signature-likeness; text-like handwriting and print are separated."),
 "SV-22": ("compare.compare", "Comparison works on ink masks (skeletons) - ink colour is ignored."),
 "SV-23": ("detector._detect", "Neighbouring marks are separated by spatial grouping; label-based 'handwriting' classification."),
 "SV-24": ("detector._detect", "Handwriting next to Name/Date fields (text layer) is reclassified as handwriting."),
 "SV-25": ("detector._detect", "Stamps, logos and printed text are classified separately and excluded by default."),
 "SV-26": ("verify.verify_document", "No signature on the document -> NO SIGNATURE verdict."),
 "SV-27": ("pipeline.manual_candidate", "Second pass + manual region selector in the UI."),
 "SV-28": ("detector._nms", "Overlapping detections of the same signature are suppressed."),
 "SV-29": ("loader._finish / UI", "Identical pages and identical uploaded files are detected by hash and skipped."),
 "SV-30": ("loader._finish", "Page-size changes, page-number restarts and mixed kinds raise a merged-document warning."),
 "SV-31": ("loader._finish / pipeline", "Printed page numbers detected; pages automatically re-ordered."),
 "SV-32": ("loader.load_document", "Corrupt/empty/password-protected files give a clear message and password prompt; partial pages skipped."),
 "SV-33": ("detector._detect / loader", "Embedded raster signatures are detected and marked as 'embedded image'."),
 "SV-34": ("loader.load_document", "Electronic-signature text marks and cryptographic signatures are reported."),
 "SV-35": ("loader.inspect_digital_signatures", "A visible signature image without a /ByteRange signature is reported as not cryptographically verifiable."),
 "SV-36": ("loader.inspect_digital_signatures", "Bytes after the signed range / digest mismatch -> 'modified after signing'."),
 "SV-37": ("loader.inspect_digital_signatures", "Certificate validity period, self-signed status checked."),
 "SV-38": ("verify.verify_candidate", "No specimen -> NO_REFERENCE, manual review."),
 "SV-39": ("references.extract_specimen / verify", "Specimen quality scored at registration and at verification; weak specimen -> review."),
 "SV-40": ("verify.verify_candidate", "Library scan: if another customer matches clearly better, flagged as wrong person."),
 "SV-41": ("verify._against / _combine", "All specimens of a customer are compared; best + top-2 average."),
 "SV-42": ("verify.verify_candidate", "Specimen dates compared; match to old specimens only -> review."),
 "SV-43": ("verify.verify_candidate", "Mid-range scores or metric disagreement -> REVIEW rather than reject."),
 "SV-44": ("references.Library.lookalikes / verify", "Look-alike specimens warned at registration; ambiguity checked at verification."),
 "SV-45": ("compare.identity_iou / pipeline._flag_repeats", "Pixel-identical signatures (vs specimen or elsewhere in the file) -> SUSPECT."),
 "SV-46": ("compare / verify", "Multi-metric structural comparison; skilled forgeries are escalated to manual review (see README limits)."),
 "SV-47": ("detector._detect", "Crop is the union of all stroke components of the region - strokes cannot be cut."),
 "SV-48": ("detector._detect", "Stroke retention measured after cleaning; fallback to un-cleaned crop and flag."),
 "SV-49": ("compare.to_canvas / compare", "Scale-normalised skeleton comparison with rotation search (+-16 deg, 180/90/270)."),
 "SV-50": ("verify.verify_document", "Every signature is compared only with the specimen of its own role; duplicate assignments warned."),
 "SV-51": ("verify.verify_candidate", "Grey zone between thresholds -> REVIEW."),
 "SV-52": ("verify.verify_candidate", "Accept needs high score AND >=3 agreeing metrics; copies/quality problems block auto-accept; thresholds tunable."),
 "SV-53": ("verify.verify_candidate", "Reject needs low score AND agreeing metrics; image-quality problems downgrade to review."),
 "SV-54": ("verify.verify_document", "Policy of required roles; missing -> INCOMPLETE."),
 "SV-55": ("verify.verify_document", "Required count per role (e.g. 2 signatures)."),
 "SV-56": ("verify.verify_candidate", "Signature closer to a customer outside the assigned parties -> unauthorised signer."),
 "SV-57": ("detector._detect / verify", "Thumb impressions detected (dense oval blob) and sent to manual review."),
 "SV-58": ("detector.find_label / verify", "Witness / bank-officer roles are excluded from customer comparison."),
 "SV-59": ("verify.verify_document", "Per-signature results plus partial-match summary."),
 "SV-60": ("pipeline.analyze / verify", "Per-page timeout and try/except; per-signature try/except -> manual review, run continues."),
 "SV-61": ("loader / pipeline", "Lazy page rendering, page cap, progress bar, candidate limit, thumbnails only."),
 "SV-62": ("compare / pipeline", "Fully deterministic (no randomness); cache per signature/specimen; self-test checks it."),
 "SV-63": ("references.queue_add / UI", "Review queue with side-by-side evidence, approve/reject, audit log."),
 "SV-64": ("pipeline / UI", "Stored page thumbnail + crop used when the original page cannot be re-rendered."),
}


def load_scenarios():
    rows = []
    path = os.path.join(HERE, "data", "Scenarios.txt")
    for line in open(path, encoding="utf-8"):
        if "\t" in line:
            sid, q = line.rstrip("\n").split("\t", 1)
            mod, how = H.get(sid, ("-", "-"))
            rows.append({"ID": sid, "Scenario": q, "Handled in": mod, "How it is handled": how})
    return rows
