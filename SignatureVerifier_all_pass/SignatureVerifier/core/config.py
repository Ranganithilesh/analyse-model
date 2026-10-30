"""Central configuration: thresholds, roles and label patterns."""
from dataclasses import dataclass, asdict


@dataclass
class Params:
    dpi: int = 150                    # render resolution for PDFs
    max_pages: int = 500              # SV-61 safety cap for huge PDFs
    page_timeout_s: int = 90          # SV-60 per-page timeout
    max_candidates_per_page: int = 12
    min_sig_score: float = 0.45       # detector confidence to call a region a signature
    min_quality: float = 0.35         # below this a signature is never auto-decided
    accept_threshold: float = 70.0    # >= and metric-agreement -> MATCH
    reject_threshold: float = 45.0    # <= and metric-agreement -> MISMATCH
    copy_iou: float = 0.97            # SV-45 pixel-identity limit (pasted copy)
    ref_min_quality: float = 0.40     # SV-39 specimen quality limit
    other_signer_margin: float = 6.0  # SV-40/44/56 margin vs another customer
    auto_reorder: bool = True         # SV-31
    auto_rotate: bool = True          # SV-03

    def as_dict(self):
        return asdict(self)


CUSTOMER_ROLES = ("account_holder", "borrower", "co_borrower", "guarantor",
                  "director", "authorised_signatory")
NON_CUSTOMER_ROLES = ("witness", "bank_officer")
ALL_ROLES = CUSTOMER_ROLES + NON_CUSTOMER_ROLES + ("unknown",)

# order matters: most specific first
ROLE_PATTERNS = [
    ("co_borrower", r"co[\s\-]?(borrower|applicant|obligor)"),
    ("guarantor", r"guarantor|surety"),
    ("borrower", r"borrower|applicant|obligor"),
    ("authorised_signatory", r"authori[sz]ed\s+signator"),
    ("director", r"director"),
    ("witness", r"witness"),
    ("bank_officer", r"bank\s+officer|verified\s+by|officer|manager|for\s+(bank|office)\s+use"),
    ("account_holder", r"account\s*holder|customer|depositor|holder"),
]
FIELD_PATTERN = r"^(full\s+)?(name|date|place|amount|address|phone|mobile|dob|pan|account\s*(no|number))\b"
