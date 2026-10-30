"""Decision logic: per-signature verdicts and document-level verdict.
SV-08-11, 38-46, 50-59, 62-63."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .compare import compare
from .config import CUSTOMER_ROLES, NON_CUSTOMER_ROLES
from .detector import Candidate

BLOCKING = {"cut_off", "too_small", "blurred", "stamp_overlap", "cleaning_loss", "text_overlap"}


@dataclass
class VResult:
    cand_id: str
    page: int
    role: str
    customer: Optional[str]
    verdict: str                      # MATCH | MISMATCH | REVIEW | SUSPECT | EXCLUDED | NO_REFERENCE
    score: float = 0.0
    confidence: str = ""
    reasons: list = field(default_factory=list)
    best_specimen: Optional[dict] = None
    per_specimen: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    other: Optional[tuple] = None


@dataclass
class DocVerdict:
    verdict: str
    headline: str
    results: List[VResult]
    missing: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _against(c: Candidate, lib, cid, cache, try_flip=True):
    """Compare a questioned signature with ALL specimens of one customer (SV-41/42)."""
    specs = lib.specimens(cid)
    rows = []
    for s in specs:
        key = (c.id, cid, s["file"])
        if key not in cache:
            cache[key] = compare(c.mask, s["mask"], try_flip=try_flip)
        r = cache[key]
        rows.append({"file": s["file"], "date": s["date"], "quality": s["quality"], "low_quality": s["low_quality"],
                     "score": r["score"], "iou": r["iou"], "metrics": r["metrics"], "for": r["for"], "against": r["against"],
                     "angle": r["angle"], "crop": s["crop"]})
    return rows


def _combine(rows):
    sc = sorted((r["score"] for r in rows), reverse=True)
    return round(0.7 * sc[0] + 0.3 * (sum(sc[:2]) / len(sc[:2])), 1) if sc else 0.0


def verify_candidate(c: Candidate, lib, possible: List[str], params, cache, library_scan=True) -> VResult:
    r = VResult(c.id, c.page, c.role, None, "REVIEW")
    try:
        if c.kind == "thumb":                                              # SV-57
            r.reasons.append("Thumb impression - cannot be verified as a signature; manual review required (SV-57)")
            return r
        if c.role in NON_CUSTOMER_ROLES:                                   # SV-58
            r.verdict = "EXCLUDED"
            r.reasons.append(f"Signature of {c.role.replace('_', ' ')} - not compared with the customer specimen (SV-58)")
            return r
        if not possible:
            r.reasons.append("No customer is assigned to this signer role (SV-10/50)")
            return r
        best_cid, best_rows, best_sc = None, [], -1.0
        for cid in possible:
            rows = _against(c, lib, cid, cache)
            if not rows:
                continue
            sc = _combine(rows)
            if sc > best_sc:
                best_cid, best_rows, best_sc = cid, rows, sc
        if best_cid is None:                                               # SV-38
            r.verdict = "NO_REFERENCE"
            r.reasons.append("No specimen signature on file for " + ", ".join(possible) + " (SV-38)")
            return r
        r.customer, r.score, r.per_specimen = best_cid, best_sc, best_rows
        top = max(best_rows, key=lambda x: x["score"])
        r.best_specimen, r.metrics = top, top["metrics"]
        # --- base decision with metric agreement (SV-51/52/53)
        if best_sc >= params.accept_threshold and top["for"] >= 3:
            r.verdict, r.confidence = "MATCH", "high" if best_sc >= params.accept_threshold + 8 else "medium"
        elif best_sc <= params.reject_threshold and top["against"] >= 2:
            r.verdict, r.confidence = "MISMATCH", "high" if best_sc <= params.reject_threshold - 10 else "medium"
        else:
            r.verdict = "REVIEW"
            r.reasons.append(f"Score {best_sc} lies between reject ({params.reject_threshold}) and accept "
                             f"({params.accept_threshold}) or metrics disagree (SV-51/43)")
        # --- specimen quality / age (SV-39, 42, 41)
        if top["low_quality"] or top["quality"] < params.ref_min_quality:
            r.reasons.append("Best specimen is of low quality (SV-39)")
            if r.verdict != "MISMATCH":
                r.verdict = "REVIEW"
        if len(best_rows) > 1:
            dated = sorted(best_rows, key=lambda x: x["date"])
            newest = dated[-1]
            if top is not newest and newest["score"] < params.accept_threshold - 15 and top["score"] >= params.accept_threshold:
                r.reasons.append("Matches only an OLD specimen; newest specimen disagrees - signature may have changed over time (SV-42)")
                r.verdict = "REVIEW"
            r.reasons.append(f"{len(best_rows)} specimens compared; best {top['score']}, worst {min(x['score'] for x in best_rows)} (SV-41)")
        # --- pasted copy (SV-45)
        if top["iou"] >= params.copy_iou or "pixel_identical" in c.flags:
            r.verdict = "SUSPECT"
            r.reasons.append("Signature is pixel-identical to a specimen / another signature: pasted copy suspected (SV-45)")
        # --- image quality gates (SV-16-20, 12-14, 47-48)
        blocked = BLOCKING & set(c.flags)
        if c.quality < params.min_quality or blocked:
            if r.verdict in ("MATCH", "MISMATCH"):
                r.verdict = "REVIEW"
            r.reasons.append("Automatic decision withheld because of: " + ", ".join(sorted(blocked) or ["low image quality"]) +
                             f" (quality {c.quality:.2f})")
        # --- unknown signer role: never auto-accept (SV-10)
        if c.role == "unknown" and r.verdict == "MATCH":
            r.verdict = "REVIEW"
            r.reasons.append(f"Signer role unknown; best match is {best_cid} but role must be confirmed (SV-10)")
        # --- someone else's signature? (SV-40/44/56)
        if library_scan and r.verdict != "SUSPECT":
            others = [x for x in lib.customers() if x not in possible][:50]
            ob = (None, -1.0)
            for oc in others:
                rows = _against(c, lib, oc, cache, try_flip=False)
                if rows and _combine(rows) > ob[1]:
                    ob = (oc, _combine(rows))
            if ob[0] and ob[1] >= params.accept_threshold and ob[1] > best_sc + params.other_signer_margin:
                r.other = ob
                r.verdict = "MISMATCH" if best_sc < params.accept_threshold else "REVIEW"
                r.reasons.append(f"Signature is closer to a DIFFERENT customer ({ob[0]}: {ob[1]}) - wrong or unauthorised signer (SV-40/44/56)")
        if r.verdict == "MATCH" and not r.reasons:
            r.reasons.append("All metrics agree with the specimen")
    except Exception as e:                                                # SV-60
        r.verdict = "REVIEW"
        r.reasons.append(f"Comparison failed ({type(e).__name__}: {e}) - manual review (SV-60)")
    return r


def verify_document(analysis, lib, assignments: Dict[str, str], policy: Dict[str, int], params, digital_sigs=None) -> DocVerdict:
    cache: dict = {}
    warns = []
    if len(set(assignments.values())) < len(assignments):
        warns.append("The same customer is assigned to more than one role (check assignment, SV-50).")
    sigs = [c for c in analysis.candidates if c.included and c.kind in ("signature", "thumb")]
    results = []
    for c in sigs:
        if c.role in assignments:
            possible = [assignments[c.role]]
        elif c.role == "unknown":
            possible = sorted(set(assignments.values()))
        else:
            possible = [assignments[c.role]] if c.role in assignments else []
        results.append(verify_candidate(c, lib, possible, params, cache))
    # --- required signatures (SV-26, 54, 55)
    present: Dict[str, int] = {}
    for c in sigs:
        present[c.role] = present.get(c.role, 0) + 1
    missing = []
    spare_unknown = present.get("unknown", 0)
    for role, need in policy.items():
        have = present.get(role, 0)
        if have < need:
            use = min(spare_unknown, need - have)
            spare_unknown -= use
            if have + use < need:
                missing.append((role, need, have + use))
    counts = {k: sum(1 for r in results if r.verdict == k) for k in ("MATCH", "MISMATCH", "REVIEW", "SUSPECT", "EXCLUDED", "NO_REFERENCE")}
    ds_bad = [s for s in (digital_sigs or []) if s["problems"]]
    if not sigs:
        v, h = "NO SIGNATURE", "No signature found on any page (SV-26/54)"
    elif missing:
        v = "INCOMPLETE"
        h = "Missing required signature(s): " + "; ".join(f"{r.replace('_', ' ')} {have}/{need}" for r, need, have in missing) + " (SV-54/55)"
    elif counts["MISMATCH"] or counts["SUSPECT"]:
        v, h = "REJECT / ESCALATE", f"{counts['MISMATCH']} mismatch, {counts['SUSPECT']} suspected copy"
    elif counts["REVIEW"] or counts["NO_REFERENCE"] or ds_bad:
        v, h = "MANUAL REVIEW", f"{counts['REVIEW'] + counts['NO_REFERENCE']} signature(s) need a human decision (SV-63)"
    else:
        v, h = "PASS", "All required signatures match the bank specimens"
    if counts["MATCH"] and (counts["MISMATCH"] or counts["REVIEW"]):
        h += f" - partial: {counts['MATCH']} of {len(results)} matched (SV-59)"
    return DocVerdict(v, h, results, missing, warns, counts)
