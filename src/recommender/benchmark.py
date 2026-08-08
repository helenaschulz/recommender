"""Orchestration: assemble the evaluation universe once, at whichever item level.

Four call sites need exactly the same seven objects — the runner, the two tuning
scripts and notebook 02 — and M10 already recorded what happens when that setup is
written more than once: the notebook silently reported the explicit-only ablation as a
duplicate of the binarized row, because the "which matrix does this model fit on" logic
lived in the runner and the notebook had drifted. So it is written once, here.

**Why the order in :func:`build_bench` is not negotiable.** At work level the split has to
be drawn *before* the work-level catalogue is built, because the catalogue's canonical
per-work text is chosen from the most-interacted edition and that count must be taken on
train only (see :func:`recommender.data.work_level_catalog`). Building the catalogue first
would let the holdout decide which edition's title represents a work — a thin leakage
channel, but this project's contract is that there are none.

**Nested holdouts.** The tuning scripts carve a second leave-one-out split out of train
(seed 43). Their canonical text must exclude *both* holdouts, which is what
``extra_holdout`` is for. Tuning on a universe that has seen the test set would undo the
whole point of tuning on validation.

**Ceilings belong here too.** :func:`ceilings` recomputes ledger L20/L21 — what share of
held-out items any collaborative or content model could reach *at all* — from whatever
universe the :class:`Bench` describes. It lives beside the universe rather than in a
script because a HitRate read against the wrong ceiling is worse than one read against
none, and the work-level re-base moves both numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recommender.data import (
    CATALOG_SIZE,
    BookCrossing,
    Interactions,
    Works,
    build_interactions,
    load,
    to_work_level,
    work_level_catalog,
)
from recommender.gallery import ANCHORS
from recommender.split import SEED, Split, make_split


@dataclass(frozen=True)
class Bench:
    """Everything a model run needs, consistent with itself by construction."""

    catalog: BookCrossing  # keyed by ISBN, or by work id at work level
    split: Split
    train: Interactions
    catalog_ids: set[str]  # the Coverage@K numerator universe
    catalog_size: int  # the Coverage@K denominator — one value, used everywhere
    anchors: dict[str, str]  # gallery anchors, translated to the current item level
    works: Works | None  # None at ISBN level
    work_level: bool

    @property
    def item_level(self) -> str:
        return "work" if self.work_level else "ISBN"

    def describe(self) -> str:
        return (
            f"{self.item_level} level · {self.catalog_size:,} items in the coverage "
            f"denominator · {self.split.describe()}"
        )


def build_bench(
    *,
    work_level: bool,
    catalog: BookCrossing | None = None,
    seed: int = SEED,
    extra_holdout: pd.DataFrame | None = None,
) -> Bench:
    """Load, re-key if asked, split, and build the train matrix — in that order.

    Args:
        extra_holdout: further ``(User-ID, ISBN)`` rows to keep out of the canonical-text
            support counts, on top of ``split.test``. Used by the tuning scripts for their
            inner validation holdout.
    """
    catalog = catalog if catalog is not None else load()

    if not work_level:
        split = make_split(catalog.ratings, seed=seed)
        return Bench(
            catalog=catalog,
            split=split,
            train=build_interactions(split.train, weights="binary"),
            catalog_ids=set(catalog.books["ISBN"]),
            catalog_size=CATALOG_SIZE,
            anchors=dict(ANCHORS),
            works=None,
            work_level=False,
        )

    works = catalog.works
    split = make_split(to_work_level(catalog.ratings, works), seed=seed)
    holdout = split.test[["User-ID", "ISBN"]]
    if extra_holdout is not None and len(extra_holdout):
        holdout = pd.concat([holdout, extra_holdout[["User-ID", "ISBN"]]], ignore_index=True)
    work_catalog = work_level_catalog(catalog, works, holdout=holdout)
    return Bench(
        catalog=work_catalog,
        split=split,
        train=build_interactions(split.train, weights="binary"),
        catalog_ids=set(work_catalog.books["ISBN"]),
        catalog_size=works.n_works,
        anchors={works.of([isbn])[0]: label for isbn, label in ANCHORS.items()},
        works=works,
        work_level=True,
    )


@dataclass(frozen=True)
class Ceilings:
    """Ledger L20/L21 on one item universe: what perfect would look like, and it is not 1.0."""

    collaborative: float  # held-out items that appear at all in the train matrix
    content: float  # held-out items that have a catalogue row, so they can be embedded
    union: float  # reachable by one or the other — the bound on any hybrid
    n_holdouts: int
    item_level: str

    def describe(self) -> str:
        return (
            f"ceilings at {self.item_level} level over {self.n_holdouts:,} holdouts: "
            f"collaborative {self.collaborative:.2%}, content {self.content:.2%}, "
            f"union {self.union:.2%}"
        )


def ceilings(bench: Bench) -> Ceilings:
    """Recompute the structural ceilings for *bench*'s item universe.

    A collaborative model cannot rank an item that is absent from the train matrix, and a
    content model cannot vectorize an item with no catalogue row. Those two shares — and
    their union — bound every HitRate in the table from above, so they are recomputed on
    the same basis as the table rather than carried over from the ISBN-level run.
    """
    holdout = bench.split.test["ISBN"]
    in_train = holdout.isin(set(bench.train.item_ids))
    has_metadata = holdout.isin(bench.catalog_ids)
    return Ceilings(
        collaborative=float(in_train.mean()),
        content=float(has_metadata.mean()),
        union=float((in_train | has_metadata).mean()),
        n_holdouts=len(holdout),
        item_level=bench.item_level,
    )


def inner_bench(outer: Bench, *, seed: int = 43) -> Bench:
    """A validation split carved out of *outer*'s train, one level deeper.

    This is the only surface a hyperparameter may be chosen on. The outer holdout is
    never touched, and at work level the inner universe's canonical text excludes both
    holdouts, so nothing about the test set reaches a tuning decision.
    """
    inner_split = make_split(outer.split.train, seed=seed)
    train = build_interactions(inner_split.train, weights="binary")
    return Bench(
        catalog=outer.catalog,
        split=inner_split,
        train=train,
        catalog_ids=outer.catalog_ids,
        catalog_size=outer.catalog_size,
        anchors=outer.anchors,
        works=outer.works,
        work_level=outer.work_level,
    )
