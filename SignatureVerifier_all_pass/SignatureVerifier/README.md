# Signature Verification Console

Offline signature verification for bank documents. **No external API, no cloud, no ML downloads** –
everything runs locally with OpenCV, PyMuPDF and scikit-image. It handles all 64 scenarios (SV-01 … SV-64)
from `data/Scenarios.txt`; the *Scenario coverage* page shows, for each one, where and how it is handled.

## Run in VS Code

1. Extract the folder and open it in VS Code (Python 3.9+).
2. Open a terminal in the folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   streamlit run app.py
   ```
   (or double-click `run.bat` / run `./run.sh`). The browser opens at http://localhost:8501.
3. First use: **Specimen library → Load demo data**. This registers 4 customers and writes 10 sample
   documents to `data/samples/`. Then go to **Verify document**, choose Borrower = ALICE and
   Guarantor = BOB, upload the samples and click *Analyse & verify*.

## What each sample shows

| Sample | Expected result |
|---|---|
| 01_valid_loan_alice_bob | PASS |
| 02_forged_guarantor | REJECT – guarantor signature belongs to someone else |
| 03_pasted_copy_of_specimen | REJECT – borrower is a pixel-identical copy of the specimen |
| 04_scan_with_red_stamp_and_table | MANUAL REVIEW – scanned (no text layer → signer roles must be assigned), stamp + table lines handled |
| 05_missing_guarantor | INCOMPLETE |
| 06_landscape | PASS |
| 07_faint_scan | MANUAL REVIEW – faint ink, never auto-accepted |
| 08_blurred_shadow_scan | MANUAL REVIEW – blur/shadow flagged |
| 09_two_pages_wrong_order | PASS – pages re-ordered automatically |
| 10_witness_and_officer | witness / officer excluded; guarantor missing → INCOMPLETE |

## Quick check (new, simplest flow)

Sidebar → **Quick check**. Pick the registered customer (or upload a registered signature), upload the document (PDF/image).
The app finds only the pen signature, crops it, cleans it to black ink on white, cleans the registered signature the same way,
shows the 4 steps (original → crop → cleaned document signature → cleaned registered signature) and says **APPROVED** when
they look the same, otherwise **NOT APPROVED**. The *Approval level* slider sets how strict the match is.
If the document has several signatures (borrower, guarantor ...) the closest one to the registered signature is used.

## Pages

* **Verify document** – choose signers/roles and required signatures, upload PDFs/images (password box for protected
  files), see per-signature verdicts, evidence side-by-side with the specimen, page viewer with boxes, manual region
  tool, rotate tool, CSV/JSON export, "send to manual review".
* **Specimen library** – register several dated specimens per customer; quality and look-alike warnings.
* **Review queue** – approve / reject / ask for the original; every decision goes to the audit log.
* **Scenario coverage** – all 64 scenarios with handling and self-test status.
* **Self-test** – generates test documents and checks each scenario (`python tests/run_selftests.py` does the same in a terminal).
* **Settings** – thresholds, DPI, timeouts, sensitivity.  **Audit log** – every action.

## How it decides

1. **Load** – PDF/image, password/corruption handling, orientation, duplicates, page order, merged-file hints,
   cryptographic signature checks (tamper, expiry).
2. **Detect** – illumination normalisation, table-line removal, stamp removal (colour + ring detection), printed text
   removal via the text layer, stroke grouping, scoring, thumb-print detection, role labels from nearby text.
3. **Compare** – ink-colour/size/angle-invariant skeleton comparison with five metrics (chamfer, HOG, grid, structure, aspect).
4. **Decide** – MATCH needs a high score *and* agreeing metrics; MISMATCH needs a low score *and* agreement; everything
   else (grey zone, poor quality, truncated, overlapped by a stamp, unknown signer, pasted copy, low-quality specimen …)
   is sent to a human. Pasted copies, wrong-person signatures and missing/incomplete signatures are reported explicitly.

## Honest limits

* This is a classical computer-vision pipeline, not a forensic examiner. **Skilled forgeries (SV-46) and false
  accepts/rejects (SV-52/53) cannot be eliminated**; the design reduces them (metric agreement, quality gates,
  copy detection, wrong-person scan, tunable thresholds) and routes uncertain cases to manual review. Calibrate the
  thresholds on your own genuine/forged samples before production use.
* Scans have no text layer, so signer roles cannot be read automatically (no OCR without external tools) – the UI
  lets the reviewer assign them, and such signatures are never auto-accepted.
* Upside-down scans are not auto-flipped (unreliable); comparison tests 180°/90°/270° and the page can be rotated manually.
* Certificate checks verify the signed digest and validity dates, not a trusted-root chain.
* Self-test data are synthetic; real scans will behave differently, so tune on real documents.

## Layout

```
app.py                 Streamlit UI
core/loader.py         PDF/image loading, orientation, duplicates, digital signatures
core/detector.py       signature detection, stamps, lines, roles, stitching
core/compare.py        signature comparison metrics
core/verify.py         verdict logic (per signature and per document)
core/references.py     specimen library, audit log, review queue
core/pipeline.py       per-page isolation, timeouts, manual regions
core/scenarios.py      SV-01..64 → handling map
core/synth.py          synthetic signature/document generator (tests, samples)
tests/run_selftests.py automatic scenario checks
tools/make_samples.py  demo data
data/                  library, samples, audit log, review queue (created at run time)
```
