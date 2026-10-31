"""
Recording-quality assessment.

A result computed from a bad recording must never be reported as "Normal".
Every analyzer builds a QualityReport first; if `ok` is False the analysis is
refused with a clear message instead of returning a misleading score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from neurotrack.core.signal import Sanitized, effective_rate, max_resolvable_hz


class AnalysisError(Exception):
    """Raised when the input cannot produce a trustworthy result (-> HTTP 422)."""

    def __init__(self, message: str, issues: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.issues = issues or []


@dataclass
class QualityReport:
    ok: bool
    n_samples: int
    duration_s: float
    effective_fps: float
    coverage: float                 # fraction of the expected duration with tracking
    longest_gap_s: float
    max_reliable_freq_hz: float     # highest frequency the sample rate can resolve
    reported_fps: float | None = None
    issues: list[str] = field(default_factory=list)   # failures (ok=False)
    warnings: list[str] = field(default_factory=list) # degrade but do not block

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("duration_s", "effective_fps", "coverage", "longest_gap_s", "max_reliable_freq_hz"):
            d[k] = round(float(d[k]), 3)
        return d


def assess(
    s: Sanitized,
    *,
    expected_duration_s: float | None,
    min_fps: float,
    min_coverage: float,
    max_gap_s: float,
    min_samples: int,
    reported_fps: float | None = None,
) -> QualityReport:
    t = s.t_s
    n = len(t)
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    fps = effective_rate(t)
    gaps = np.diff(t) if n > 1 else np.array([0.0])
    longest_gap = float(gaps.max()) if len(gaps) else 0.0

    expected = expected_duration_s if expected_duration_s and expected_duration_s > 0 else duration
    # Coverage = tracked time (excluding gaps longer than max_gap) / expected duration.
    tracked = float(gaps[gaps <= max_gap_s].sum()) if len(gaps) else 0.0
    coverage = min(1.0, tracked / expected) if expected > 0 else 0.0

    issues: list[str] = []
    warnings: list[str] = []
    if n < min_samples:
        issues.append(f"only {n} valid samples (need at least {min_samples})")
    if fps < min_fps * 0.97:   # small tolerance: a nominal 15 fps camera measures ~14.9
        issues.append(f"tracking rate {fps:.1f} fps is below the {min_fps:.0f} fps minimum")
    if coverage < min_coverage:
        issues.append(f"hand tracked for {coverage * 100:.0f}% of the test (need {min_coverage * 100:.0f}%)")
    if longest_gap > max_gap_s:
        issues.append(f"tracking was lost for {longest_gap:.1f}s at once (max {max_gap_s:.1f}s)")
    if s.n_dropped:
        warnings.append(f"{s.n_dropped} non-finite samples were discarded")
    if s.n_merged:
        warnings.append(f"{s.n_merged} duplicate-timestamp samples were merged")
    if s.reordered:
        warnings.append("timestamps arrived out of order and were sorted")

    return QualityReport(
        ok=not issues,
        n_samples=n,
        duration_s=duration,
        effective_fps=fps,
        coverage=coverage,
        longest_gap_s=longest_gap,
        max_reliable_freq_hz=max_resolvable_hz(fps),
        reported_fps=reported_fps,
        issues=issues,
        warnings=warnings,
    )


def require_ok(q: QualityReport, what: str) -> None:
    if not q.ok:
        raise AnalysisError(f"Recording quality too low for the {what}: " + "; ".join(q.issues) + ".", q.issues)
