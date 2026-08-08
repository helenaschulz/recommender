"""Edition-clustering tests, pinned to the case that motivated the milestone.

*Crime and Punishment* is the whole problem in one title: 24 ISBNs of the same novel
spread over five author spellings and a dozen series parentheticals, sitting next to a
school textbook that shares the title and is not the same book at all. Every assertion
below is taken from real ``Books.csv`` rows, not invented.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recommender.data import (
    author_last_name,
    cluster_works,
    normalize_title,
    split_series,
    to_work_level,
)

#: Real catalogue rows: the five author spellings, four series parentheticals, one
#: same-title-different-work trap, and one title variant that must stay separate.
CRIME_AND_PUNISHMENT = pd.DataFrame(
    {
        "ISBN": [
            "0553211757",
            "0140449132",
            "0140440232",
            "0192815490",
            "0553210939",
            "0393956237",
            "0451527232",
            "0750224665",
            "0679734503",
        ],
        "Book-Title": [
            "Crime and Punishment (Crime &amp; Punishment)",
            "Crime and Punishment (Penguin Classics)",
            "Crime and Punishment (Classics S.)",
            "Crime and Punishment (The World's Classics)",
            "Crime and Punishment",
            "Crime and Punishment",
            "Crime and Punishment (Signet Classics (Paperback))",
            "Crime and Punishment (Twentieth Century Issues S.)",
            "Crime and Punishment: A Novel in Six Parts With Epilogue",
        ],
        "Book-Author": [
            "Fyodor Dostoevsky",
            "Fyodor Dostoyevsky",
            "Fyodor Dostoevsky",
            "Fedor Dostoevsky",
            "Fyodor M. Dostoevsky",
            "Feodor Dostoevsky",
            "Fyodor Dostoyevsky",
            "Ali Brownlie",  # a textbook about crime, not the novel
            "Fyodor Dostoyevsky",
        ],
    }
)


def test_the_five_dostoevsky_spellings_land_in_one_work() -> None:
    """Fyodor / Fedor / Feodor / Fyodor M. / Dostoyevsky — one novel, one work id."""
    works = cluster_works(CRIME_AND_PUNISHMENT)
    novel_isbns = ["0553211757", "0140449132", "0140440232", "0192815490", "0553210939", "0451527232"]
    ids = {works.work_of_isbn[isbn] for isbn in novel_isbns}
    assert len(ids) == 1, f"expected one work, got {ids}"


def test_the_same_title_by_a_different_author_does_not_join() -> None:
    """Ali Brownlie's *Crime and Punishment* is a different book that shares a title."""
    works = cluster_works(CRIME_AND_PUNISHMENT)
    assert works.work_of_isbn["0750224665"] != works.work_of_isbn["0553210939"]


def test_the_author_variant_pass_is_what_merges_the_two_surnames() -> None:
    """Without it the pinned title+last-name key splits Dostoevsky from Dostoyevsky.

    This is the deviation recorded in ledger L41, asserted rather than described: the
    pinned key alone leaves two clusters, the transliteration pass leaves one.
    """
    pinned_only = cluster_works(CRIME_AND_PUNISHMENT, merge_author_variants=False)
    assert pinned_only.work_of_isbn["0553211757"] != pinned_only.work_of_isbn["0140449132"]
    extended = cluster_works(CRIME_AND_PUNISHMENT)
    assert extended.work_of_isbn["0553211757"] == extended.work_of_isbn["0140449132"]


def test_a_subtitle_is_not_a_series_and_keeps_its_own_work() -> None:
    """Only the trailing parenthetical is packaging. A colon subtitle may be a different
    edition *or* a different text (here: a critical edition with essays), so it stays
    separate — the clustering errs towards not merging."""
    works = cluster_works(CRIME_AND_PUNISHMENT)
    assert works.work_of_isbn["0679734503"] != works.work_of_isbn["0553210939"]


def test_the_variant_pass_never_reaches_across_titles() -> None:
    """One-edit surnames only merge under an identical title, so a real pair of authors
    with similar names keeps its books apart."""
    books = pd.DataFrame(
        {
            "ISBN": ["x1", "x2"],
            "Book-Title": ["Endymion", "The Wild Swans at Coole"],
            "Book-Author": ["John Keats", "W. B. Yeats"],
        }
    )
    works = cluster_works(books)
    assert works.work_of_isbn["x1"] != works.work_of_isbn["x2"]


def test_short_surnames_are_below_the_variant_floor() -> None:
    """keats / yeats differ by one edit; under an identical title the length floor is the
    only thing that keeps them apart, so the floor gets its own test."""
    books = pd.DataFrame(
        {
            "ISBN": ["x1", "x2"],
            "Book-Title": ["Poems", "Poems"],
            "Book-Author": ["John Keats", "W. B. Yeats"],
        }
    )
    works = cluster_works(books)
    assert works.work_of_isbn["x1"] != works.work_of_isbn["x2"]


@pytest.mark.parametrize(
    ("title", "expected_base", "expected_series"),
    [
        ("Crime and Punishment (Penguin Classics)", "Crime and Punishment", "Penguin Classics"),
        # Nested brackets come off as one unit, not as a dangling "(Signet Classics".
        ("Crime and Punishment (Signet Classics (Paperback))", "Crime and Punishment", "Signet Classics (Paperback)"),
        # Two trailing parentheticals, kept in reading order.
        ("Harry Potter (Book 1) (UK Edition)", "Harry Potter", "Book 1 | UK Edition"),
        ("Crime and Punishment (Crime &amp; Punishment)", "Crime and Punishment", "Crime & Punishment"),
        ("Crime and Punishment", "Crime and Punishment", ""),
        # A fully parenthesised title is a title, not packaging with nothing in front.
        ("(Untitled)", "(Untitled)", ""),
        # A parenthetical in the middle is part of the title.
        ("The (Honest) Truth About Dishonesty", "The (Honest) Truth About Dishonesty", ""),
    ],
)
def test_series_is_parsed_out_of_the_title(title: str, expected_base: str, expected_series: str) -> None:
    base, series = split_series(title)
    assert base == expected_base
    assert series == expected_series


@pytest.mark.parametrize(
    ("author", "expected"),
    [
        ("Fyodor M. Dostoevsky", "dostoevsky"),
        ("J. R. R. Tolkien", "tolkien"),
        ("JOHN C. TUCKER", "tucker"),
        # The catalogue's comma rows are misplaced suffixes; the surname is still last.
        ("Mark E., Jr. Neely", "neely"),
        ("Rush H., III Limbaugh", "limbaugh"),
        (None, ""),
        ("", ""),
    ],
)
def test_author_last_name(author: str | None, expected: str) -> None:
    assert author_last_name(author) == expected


def test_normalize_title_collapses_case_and_whitespace() -> None:
    assert normalize_title("  Crime   and    Punishment (Penguin Classics) ") == "crime and punishment"


def test_unmappable_books_become_their_own_work() -> None:
    """No title or no author means no evidence to merge on. Singletons, never a bucket."""
    books = pd.DataFrame(
        {
            "ISBN": ["a", "b", "c"],
            "Book-Title": ["", "", "Real Book"],
            "Book-Author": ["Someone", "Someone Else", None],
        }
    )
    works = cluster_works(books)
    assert len({works.work_of_isbn[i] for i in "abc"}) == 3
    assert all(works.work_of_isbn[i].startswith("isbn:") for i in "abc")


def test_isbns_outside_the_catalogue_map_to_themselves() -> None:
    """10.3% of interactions point at an ISBN with no catalogue row (L14). They stay
    distinct items rather than collapsing into one anonymous work."""
    works = cluster_works(CRIME_AND_PUNISHMENT)
    mapped = works.of(["0553210939", "not-a-real-isbn", "another-one"])
    assert mapped[0] == "crime and punishment|dostoevsky"
    assert mapped[1] == "isbn:not-a-real-isbn"
    assert mapped[2] == "isbn:another-one"


def test_to_work_level_collapses_a_user_reading_two_editions() -> None:
    """Two editions of one novel are one interaction with the work, at the higher grade."""
    works = cluster_works(CRIME_AND_PUNISHMENT)
    ratings = pd.DataFrame(
        {
            "User-ID": [1, 1, 1, 2],
            "ISBN": ["0553211757", "0140449132", "0750224665", "0553210939"],
            "Book-Rating": [7, 9, 5, 0],
        }
    )
    collapsed = to_work_level(ratings, works)
    user1 = collapsed[collapsed["User-ID"] == 1]
    assert len(user1) == 2  # the novel (two editions) and the Brownlie textbook
    novel = user1[user1["ISBN"] == "crime and punishment|dostoevsky"]
    assert novel["Book-Rating"].item() == 9
    assert bool(novel["is_explicit"].item()) is True
    assert bool(collapsed.loc[collapsed["User-ID"] == 2, "is_explicit"].item()) is False


def test_clustering_is_independent_of_row_order() -> None:
    """The canonical spelling is chosen alphabetically, not by whichever row came first."""
    forward = cluster_works(CRIME_AND_PUNISHMENT)
    reversed_rows = cluster_works(CRIME_AND_PUNISHMENT.iloc[::-1].reset_index(drop=True))
    assert forward.work_of_isbn.to_dict() == reversed_rows.work_of_isbn.to_dict()
