"""Display rules for the result list (M15, M17): the thin-evidence tag and the bars.

Built from fixture evidence only — no assets, no data, no model. These are presentation
rules, and the point of having them in the package rather than in the Streamlit layer is
that their edge cases are worth pinning.

The divider's tests were **deleted** with it in M17.3, not skipped: the rule drew a
contiguous boundary from a criterion that is not monotone in the sort order, and every test
here that pinned that behaviour was pinning a specification a fresh anchor falsified.
"""

from __future__ import annotations

from recommender.demo import Evidence, Suggestion
from recommender.display import (
    THIN_EVIDENCE_SHARE,
    bar_widths,
    evidence_share,
    is_thin,
)


def suggestion(co_readers: int, anchor_readers: int = 1000, score: float = 0.5) -> Suggestion:
    evidence = Evidence(score=score, co_readers=co_readers, anchor_readers=anchor_readers, same_author=False)
    return Suggestion(
        isbn="w", title="t", author="a", year="1999", series="", evidence=evidence, reason=""
    )


class TestThinEvidence:
    def test_the_share_is_of_the_anchors_readers(self) -> None:
        assert evidence_share(suggestion(216, 905).evidence) == 216 / 905

    def test_an_anchor_with_no_readers_is_not_a_division_by_zero(self) -> None:
        assert evidence_share(suggestion(0, 0).evidence) == 0.0

    def test_the_row_that_started_the_discussion_is_tagged(self) -> None:
        """Da Vinci Code rank 2: six shared readers of 905. An absolute '< 5 readers' rule
        would not fire here, which is why the threshold is a share."""
        assert is_thin(suggestion(6, 905).evidence)

    def test_a_small_anchor_is_not_tagged_for_being_small(self) -> None:
        """Fight Club -> A Clockwork Orange: 4 of 102 readers is 3.9%, a good recommendation.
        An absolute rule tags it; the share correctly does not."""
        assert not is_thin(suggestion(4, 102).evidence)

    def test_strong_rows_are_untagged(self) -> None:
        assert not is_thin(suggestion(216, 905).evidence)

    def test_the_threshold_is_where_it_says_it_is(self) -> None:
        assert is_thin(suggestion(19, 1000).evidence)
        assert not is_thin(suggestion(20, 1000).evidence)
        assert THIN_EVIDENCE_SHARE == 0.02


class TestBars:
    def test_the_bar_is_the_similarity_relative_to_the_top_of_this_list(self) -> None:
        rows = [suggestion(1, score=0.80), suggestion(1, score=0.40), suggestion(1, score=0.20)]
        assert bar_widths(rows) == [1.0, 0.5, 0.25]

    def test_the_top_row_is_always_full(self) -> None:
        """A fixed 0-1 axis would draw The Little Prince (0.33-0.40) as a row of stubs."""
        for top in (0.40, 0.80):
            assert bar_widths([suggestion(1, score=top), suggestion(1, score=top / 2)])[0] == 1.0

    def test_the_bar_no_longer_follows_the_evidence(self) -> None:
        """The row that started M17: rank 1 has *fewer* shared readers than rank 2, and the
        bar must follow the sort key rather than contradict it."""
        rows = [suggestion(16, 174, score=0.40), suggestion(17, 174, score=0.38)]
        first, second = bar_widths(rows)
        assert first > second

    def test_a_list_with_no_similarity_has_no_bars(self) -> None:
        assert bar_widths([suggestion(0, score=0.0), suggestion(0, score=0.0)]) == [0.0, 0.0]

    def test_a_negative_score_does_not_draw_a_bar_backwards(self) -> None:
        assert bar_widths([suggestion(1, score=0.5), suggestion(1, score=-0.2)]) == [1.0, 0.0]

    def test_an_empty_list_is_not_an_error(self) -> None:
        assert bar_widths([]) == []
