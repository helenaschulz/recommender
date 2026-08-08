"""Display-only rules for the demo's result list (milestone M15).

Nothing here scores, ranks, filters or reorders anything. Every function takes the list
:meth:`recommender.demo.DemoEngine.similar` already produced, in the order it produced it,
and answers a presentation question about it. That separation is deliberate and is an
acceptance condition of M15: the demo's ranking must stay *identical* to the scorer the
ledger measures, so anything that could change it does not belong in this module.

It lives beside ``demo.py`` rather than inside ``app/main.py`` because these rules have
edge cases worth a regression test, and the Streamlit layer is deliberately untested.
"""

from __future__ import annotations

from recommender.demo import Evidence, Suggestion

#: A suggestion is "thin" when fewer than this share of the anchor's own readers also read
#: it. **A UI choice, labelled as one** — exactly like ``demo.LOOKUP_TIE_MARGIN``, and it
#: touches nothing any published number is measured on.
#:
#: **A share, not an absolute count, and the correction is the argument.** The first
#: proposal was "< 5 shared readers", the column the anchor-support sweep already reports.
#: On the *Da Vinci Code* list that rule fires on **nothing** — not even on rank 2, which
#: has six shared readers out of the anchor's 905 and is the row that started the
#: discussion. It fails to fire exactly where the eye stops. The share fires on ranks 2, 7
#: and 9 and on nothing else, and it survives the small-anchor case that breaks an absolute
#: rule: *Fight Club* → *A Clockwork Orange* is 4 of 102 readers, 3.9%, and stays untagged,
#: correctly. The share is the anchor-normalised quantity, which is the whole point of the
#: calibration finding; the absolute count is not.
#:
#: **"Thin", not "wrong".** Every row on that list sits 24 to 60 times above chance overlap,
#: rank 2 included, so as an *association* it is real. What is thin is the evidence the
#: estimate rests on. Those are two different claims, and the tag says the second. (Lift
#: cannot serve as the criterion either — it does not separate these rows at all.)
THIN_EVIDENCE_SHARE = 0.02


def evidence_share(evidence: Evidence) -> float:
    """Co-readers as a share of the anchor's own readership, 0.0 when the anchor has none."""
    return evidence.co_readers / evidence.anchor_readers if evidence.anchor_readers else 0.0


def is_thin(evidence: Evidence) -> bool:
    """Does this row rest on less than :data:`THIN_EVIDENCE_SHARE` of the anchor's readers?"""
    return evidence_share(evidence) < THIN_EVIDENCE_SHARE


def bar_widths(suggestions: list[Suggestion]) -> list[float]:
    """Each row's evidence bar, as a fraction of the **strongest row in this same list**.

    Scaled within the list, never against a fixed maximum, and that is forced rather than
    chosen: similarity and co-reader counts are not comparable across anchors, so a bar on
    a common scale would render every thinly-read anchor as a row of empty tracks and would
    invite precisely the cross-anchor comparison the ledger forbids. Scaled within the
    list, the bar answers the one question it can answer honestly — which of *these ten*
    rests on the most readers.

    An all-zero list yields all-zero widths rather than a division by zero.
    """
    shares = [evidence_share(item.evidence) for item in suggestions]
    largest = max(shares, default=0.0)
    return [0.0 for _ in shares] if largest <= 0 else [share / largest for share in shares]


def divider_after(suggestions: list[Suggestion]) -> int | None:
    """Index of the last row above the thin threshold, or ``None`` for no divider.

    The divider marks where the evidence runs out. It is **not a filter and not a sort**:
    no row moves and none is removed, and everything below the line is thin *by
    construction*, because the line goes after the last row that is not.

    Rows above the line may still carry a thin tag — a strong row can sit below a weak one,
    which is itself one of this project's measured results and is the reason the ranking is
    left alone rather than corrected. The tag is per row; the divider is the boundary.

    Two cases render nothing, both on purpose:

    - **every row is above the threshold** — there is no "further out" to mark;
    - **the first row is already below it** — the whole list is thin, the tags say so, and
      a divider under nothing would imply a quality break that does not exist.
    """
    above = [index for index, item in enumerate(suggestions) if not is_thin(item.evidence)]
    if not above or above[-1] == len(suggestions) - 1:
        return None
    return above[-1] if above[0] == 0 else None
