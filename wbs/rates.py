"""One helper, because `x / max(len(y), 1)` is how a broken run reports plausible
numbers.

R15 in the risk register was realised in `hypothesis_h1.py`: every record failed,
the relation list stayed empty, and `100 * n / max(len(relations), 1)` turned that
empty measurement into a per-100 rate of 0.0 for every defect class - a result
that reads as a clean ontology. A rate over nothing is not zero, it is undefined,
and the deliverable has to say so.

    rate(3, rows)        -> 0.0316 when len(rows) == 95
    rate(3, [])          -> None
    rate(3, 0)           -> None
"""
from __future__ import annotations

from typing import Sized


def rate(num: float, den: "int | Sized", nd: int = 4) -> float | None:
    """`num/den` rounded to `nd`, or None when the denominator is empty.

    `den` may be the count itself or the collection it was counted over.
    """
    n = den if isinstance(den, int) else len(den)
    return None if not n else round(num / n, nd)
