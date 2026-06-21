"""Operational recommendation layer.

Turns the three model outputs into the concrete decisions the problem statement
asks for - manpower, barricading and diversion plans. This is a deliberately
transparent rules layer on top of the calibrated probabilities so that operators
can see *why* a recommendation was made (auditable, unlike a black box).

Manpower deliberately blends the priority signal with the duration and closure
signals. Priority is ~deterministic from the (excluded) corridor field, so we do
not rely on it alone; the duration-based tier is an independent corridor-free
check that keeps recommendations sensible even when the priority model is unsure.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Recommendation:
    closure_probability: float
    closure_expected: bool
    high_priority_probability: float
    expected_duration_min: float
    duration_low_min: float
    duration_high_min: float
    manpower_tier: str
    officers_suggested: int
    barricading: str
    diversion: str
    rationale: str

    def to_dict(self):
        return asdict(self)


def _duration_tier(expected_min: float) -> int:
    if expected_min < 60:
        return 0
    if expected_min < 240:        # < 4h
        return 1
    if expected_min < 1440:       # < 1 day
        return 2
    return 3                      # multi-day (construction-like)


def _manpower(closure_prob, priority_prob, duration_min):
    """Blend three independent signals into a 0-3 manpower tier."""
    dur_tier = _duration_tier(duration_min)
    score = 0.0
    score += 1.5 * closure_prob            # closures need traffic control
    score += 1.0 * priority_prob           # operator-assigned urgency
    score += 0.6 * dur_tier                # long events need relief shifts
    if score >= 2.2:
        tier, officers = "high", 6
    elif score >= 1.2:
        tier, officers = "medium", 3
    elif score >= 0.5:
        tier, officers = "low", 1
    else:
        tier, officers = "minimal", 0
    # A near-certain closure always warrants a crew regardless of the blend.
    if closure_prob >= 0.6 and officers < 3:
        tier, officers = "medium", 3
    return tier, officers


def recommend(closure_prob: float, closure_pred: int, priority_prob: float,
              duration_min: float, duration_low: float, duration_high: float,
              closure_threshold: float) -> Recommendation:
    closure_expected = bool(closure_pred)
    tier, officers = _manpower(closure_prob, priority_prob, duration_min)

    if closure_expected or closure_prob >= closure_threshold:
        barricading = "Full barricading at incident point + upstream warning signage"
        diversion = "Activate signed diversion via parallel corridor; pre-notify control room"
    elif closure_prob >= max(0.25, closure_threshold * 0.6):
        barricading = "Partial lane barricading; cones + advance warning"
        diversion = "Stand-by diversion plan; divert only if queue spillback observed"
    else:
        barricading = "Minimal - incident-side cones only"
        diversion = "No diversion required; monitor"

    rationale = (
        f"closure p={closure_prob:.2f} (thr {closure_threshold:.2f}); "
        f"priority p={priority_prob:.2f}; "
        f"expected clearance {duration_min:.0f} min "
        f"(80% interval {duration_low:.0f}-{duration_high:.0f}); "
        f"manpower blend -> {tier}."
    )
    return Recommendation(
        closure_probability=round(float(closure_prob), 4),
        closure_expected=closure_expected,
        high_priority_probability=round(float(priority_prob), 4),
        expected_duration_min=round(float(duration_min), 1),
        duration_low_min=round(float(duration_low), 1),
        duration_high_min=round(float(duration_high), 1),
        manpower_tier=tier,
        officers_suggested=officers,
        barricading=barricading,
        diversion=diversion,
        rationale=rationale,
    )
