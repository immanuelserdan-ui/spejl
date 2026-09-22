"""Presentation-only summaries for existing QA flags and verification data."""

from __future__ import annotations

from dataclasses import dataclass

from spejl.models import Document
from spejl.qa.verify import VerifyReport


@dataclass(frozen=True)
class ReviewSummary:
    errors: int = 0
    warnings: int = 0
    information: int = 0

    @property
    def label(self) -> str:
        if self.errors:
            return f"{self.errors} error(s) need attention"
        if self.warnings:
            return f"{self.warnings} item(s) to review"
        return "Ready to save"


def summarize_document(document: Document) -> ReviewSummary:
    errors = warnings = information = 0
    for page in document.pages:
        for flag in page.flags:
            if flag.severity == "error":
                errors += 1
            elif flag.severity == "warn":
                warnings += 1
            else:
                information += 1
    return ReviewSummary(errors, warnings, information)


def summarize_verification(report: VerifyReport) -> ReviewSummary:
    if report.passed:
        return ReviewSummary(information=1)
    return ReviewSummary(errors=1, warnings=len(report.text_issues))
