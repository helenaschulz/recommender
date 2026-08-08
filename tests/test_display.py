"""Display rules for the result list (M15): the thin-evidence tag, the bars, the divider.

Built from fixture evidence only — no assets, no data, no model. These are presentation
rules, and the point of having them in the package rather than in the Streamlit layer is
that their edge cases are worth pinning.
"""

from __future__ import annotations

from recommender.demo import Evidence, Suggestion
from recommender.display import (
    THIN_EVIDENCE_SHARE,
    bar_widths,
    divider_after,
    evidence_share,
    is_thin,
)


def suggestion(co_readers: int, anchor_readers: int = 1000) -> Suggestion:
    evidence = Evidence(score=0.5, co_readers=co_readers, anchor_readers=anchor_readers, same_author=False)
    return Suggestion(
        isbn="w", title="t", author="a", year="1999", series="", image_url="", evidence=evidence, reason=""
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
    def test_the_bar_is_relative_to_the_strongest_row_in_the_list(self) -> None:
        assert bar_widths([suggestion(200), suggestion(100), suggestion(50)]) == [1.0, 0.5, 0.25]

    def test_a_list_with_no_shared_readers_has_no_bars(self) -> None:
        assert bar_widths([suggestion(0), suggestion(0)]) == [0.0, 0.0]

    def test_an_empty_list_is_not_an_error(self) -> None:
        assert bar_widths([]) == []


class TestDivider:
    def test_it_goes_after_the_last_row_above_the_threshold(self) -> None:
        rows = [suggestion(200), suggestion(100), suggestion(5), suggestion(4)]
        assert divider_after(rows) == 1

    def test_everything_below_the_line_is_thin_by_construction(self) -> None:
        rows = [suggestion(200), suggestion(5), suggestion(100), suggestion(4)]
        cut = divider_after(rows)
        assert cut == 2
        assert all(is_thin(row.evidence) for row in rows[cut + 1 :])

    def test_no_divider_when_every_row_is_above_the_threshold(self) -> None:
        assert divider_after([suggestion(200), suggestion(100)]) is None

    def test_no_divider_when_the_first_row_is_already_below_it(self) -> None:
        """The whole list is thin. The tags say so, and a line under nothing would imply a
        quality break that is not there."""
        assert divider_after([suggestion(5), suggestion(200), suggestion(4)]) is None

    def test_no_divider_for_an_all_thin_list(self) -> None:
        assert divider_after([suggestion(5), suggestion(4)]) is None

    def test_no_divider_for_an_empty_list(self) -> None:
        assert divider_after([]) is None

    def test_it_never_moves_or_removes_a_row(self) -> None:
        rows = [suggestion(200), suggestion(100), suggestion(5)]
        before = [id(row) for row in rows]
        divider_after(rows)
        assert [id(row) for row in rows] == before
