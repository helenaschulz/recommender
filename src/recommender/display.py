"""Display-only rules for the demo's result list (milestones M15 and M17).

Nothing here scores, ranks, filters or reorders anything. Every function takes the list
:meth:`recommender.demo.DemoEngine.similar` already produced, in the order it produced it,
and answers a presentation question about it. That separation is deliberate and is an
acceptance condition of M15: the demo's ranking must stay *identical* to the scorer the
ledger measures, so anything that could change it does not belong in this module.

It lives beside ``demo.py`` rather than inside ``app/main.py`` because these rules have
edge cases worth a regression test, and the Streamlit layer is deliberately untested.

**M17 moved the bar from the evidence to the similarity, and deleted the divider.** Both
changes come from one finding: the list is ordered by the similarity, and M14.6 had taken
the similarity off the screen entirely. What was left was a ranking with nothing on screen
to justify it and one visible counter-argument — on *The Little Prince*, rank 1 showed 16
shared readers, rank 2 showed 17. The incomparability L63 measures is a claim *across*
anchors; within one list the cosine is exactly the sort key, and the app only ever shows one
list. The cross-anchor caveat is a sentence in the write-up, not a reason to hide the sort
key. The divider went for the sharper version of the same point: see the git history of
``divider_after`` for the argument (M17.3).
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
    """Each row's **similarity** bar, as a fraction of the strongest row in this same list.

    The bar is the sort key (M17.1). It was the evidence share until M17, which read
    honestly row by row and fought the list it was drawn on: the rows are ordered by
    similarity, so an evidence bar that lengthens partway down looks like a mistake in the
    ranking rather than the measured fact it is. One bar per row, and it is the one the
    order comes from; the evidence keeps its place beside it as text, where it reads as
    "how much this rests on" rather than as a competing ranking.

    **Scaled within the list, never against a fixed 0-1 axis**, and that is forced rather
    than chosen. L63 measures that a cosine is not comparable across anchors: similarity
    spans 0.33 to 0.40 on *The Little Prince* and 0.50 to 0.80 on *Interview with the
    Vampire*, so a fixed axis would draw the first list as a row of stubs and invite exactly
    the cross-anchor comparison the ledger forbids. Relative to the top row, the bar answers
    the one question it can answer honestly — how far each of *these ten* falls off the best
    one. The **absolute** value goes on screen as the number beside it, because within a
    list it is the sort key and it is honest.

    Negative similarities cannot reach here (a row only enters the list by being among the
    top-k of a cosine that the anchor itself scores 1.0 on), and a non-positive best would
    make the ratio meaningless, so that case yields all-zero widths rather than a division
    by zero or a bar pointing the wrong way.
    """
    scores = [item.evidence.score for item in suggestions]
    largest = max(scores, default=0.0)
    return [0.0 for _ in scores] if largest <= 0 else [max(score, 0.0) / largest for score in scores]
