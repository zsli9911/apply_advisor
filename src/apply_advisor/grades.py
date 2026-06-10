"""Grade-scale conversion.

Both sides of a match carry their OWN scale:
- the student's GPA is on 4.0 / 5.0 / 100 (see UserProfile.gpa_scale)
- a program's admission minimum is stored with its native scale
  (`grade_scale`), e.g. a French Licence requirement of 12 on a /20 scale.

`to_gpa4` maps any supported scale onto a common 4.0 scale so the rule engine
can compare like with like, while the raw value + scale are kept for display
(so we can say "requires 12/20 (≈GPA 3.0)").
"""
from __future__ import annotations

from typing import Optional

# scale -> ascending anchor points (raw_grade, gpa_on_4.0)
_ANCHORS: dict[float, list[tuple[float, float]]] = {
    # Chinese 100-point → US 4.0 (common admissions conversion)
    100: [(60, 2.0), (70, 2.7), (75, 3.0), (80, 3.3), (85, 3.7), (90, 4.0)],
    # French /20 (mention thresholds: passable 10, assez bien 12, bien 14, très bien 16)
    20: [(10, 2.3), (12, 3.0), (14, 3.5), (16, 3.8), (18, 4.0)],
}

SUPPORTED_SCALES = (4.0, 5.0, 20, 100)


def _interpolate(anchors: list[tuple[float, float]], g: float) -> float:
    if g <= anchors[0][0]:
        # linear from origin up to the first anchor
        x0, y0 = anchors[0]
        return round(max(0.0, y0 * g / x0), 2)
    if g >= anchors[-1][0]:
        return round(anchors[-1][1], 2)
    for (x1, y1), (x2, y2) in zip(anchors, anchors[1:]):
        if x1 <= g <= x2:
            return round(y1 + (g - x1) / (x2 - x1) * (y2 - y1), 2)
    return round(min(g, 4.0), 2)  # unreachable, keeps type-checkers happy


def to_gpa4(grade: Optional[float], scale: Optional[float]) -> Optional[float]:
    """Convert a raw grade on `scale` to the 4.0 scale. None-safe."""
    if grade is None:
        return None
    scale = scale or 4.0
    if scale == 4.0:
        return round(min(grade, 4.0), 2)
    if scale == 5.0:
        return round(min(grade / 5.0 * 4.0, 4.0), 2)
    if scale in _ANCHORS:
        return _interpolate(_ANCHORS[scale], grade)
    # unknown scale — treat as already-4.0 (callers should also warn)
    return round(min(grade, 4.0), 2)


def from_gpa4(gpa4: Optional[float], scale: Optional[float]) -> Optional[float]:
    """Inverse conversion, used when authoring sample data on a native scale."""
    if gpa4 is None:
        return None
    scale = scale or 4.0
    if scale == 4.0:
        return round(gpa4, 2)
    if scale == 5.0:
        return round(gpa4 / 4.0 * 5.0, 2)
    if scale in _ANCHORS:
        anchors = _ANCHORS[scale]
        if gpa4 <= anchors[0][1]:
            x0, y0 = anchors[0]
            return round(gpa4 * x0 / y0, 1)
        if gpa4 >= anchors[-1][1]:
            return round(anchors[-1][0], 1)
        for (x1, y1), (x2, y2) in zip(anchors, anchors[1:]):
            if y1 <= gpa4 <= y2:
                return round(x1 + (gpa4 - y1) / (y2 - y1) * (x2 - x1), 1)
    return round(gpa4, 2)


def describe(grade: Optional[float], scale: Optional[float]) -> str:
    """Human-readable '12/20 (≈GPA 3.0)' style label."""
    if grade is None:
        return "unspecified"
    scale = scale or 4.0
    if scale == 4.0:
        return f"GPA {grade}/4.0"
    return f"{grade}/{int(scale)} (≈GPA {to_gpa4(grade, scale)})"
