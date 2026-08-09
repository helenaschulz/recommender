"""Evaluation layer: the same three metrics for every model, on the same split.

**The metrics, pinned once and used for every model.**

- **HitRate@K** — share of eligible users whose held-out book appears in their top-K.
  Under leave-one-out there is exactly one relevant item per user, so HitRate@K *is*
  Recall@K, and Precision@K = HitRate@K / K. One number carries all three; reporting
  them as three separate columns would be three views of the same measurement dressed
  up as corroboration.
- **Catalog-Coverage@K** — distinct catalogue books appearing in *any* user's top-K,
  divided by the size of the catalogue it was given: **271,360 ISBNs** on the ISBN basis,
  **235,824 works** on the work basis, and the two are never mixed in one table (the
  mistake ledger L46 records). Since M12 the published rows are work-level. This is how the
  "collaborative filtering can only reach 5.3% of the catalogue" argument (ledger L12) gets
  measured instead of asserted. Recommended ids that are not in ``Books.csv`` do not count
  towards the numerator: a book we cannot name is a book we cannot show.
- **Novelty@K** — mean self-information of recommended items,
  ``-log2((train_interactions + 1) / (total_train_interactions + catalog_size))``.
  Higher means deeper in the tail. The +1 smoothing is deliberate: content models can
  recommend books with zero interactions, and an unsmoothed share would make their
  novelty infinite rather than merely high.

**AnchorHitRate@K — the second question, measured (M20).** Everything above scores a
*user* query: given this reader's history, rank their held-out book. The demo asks an
**item** query: given this one book, what is like it. Those are different questions, the
project has said so since M13, and until M20 only the first had a number — the claim that
one model has "the best neighbourhoods" rested on three anchors read by eye, from before the
work-level re-base. :func:`neighbour_matrix` measures the second on the *same* split: one
anchor per user from :func:`recommender.split.pick_anchors`, the anchor's top-K neighbours
with that reader's train items filtered out, and a hit when the held-out book is among them.
Same readers, same held-out books, same row order — so the two columns pair under
:func:`mcnemar` and the difference between the questions becomes a measured quantity
instead of an assertion.

Two things it is not. It is not a better metric than HitRate@K, it is a different one, and a
model may honestly win either. And it is **not free of a judgement call**: which train book
becomes the anchor is a free parameter, pinned in :mod:`recommender.split` and reported with
the number.

Popularity is always computed on **train only** — a novelty score that used the full
data would leak the holdout into the metric.

Accuracy alone is not a verdict here. With 25.1% of interactions sitting in the top 1%
of books (ledger L9), a model can win HitRate by recommending bestsellers to everyone,
which is why coverage and novelty sit in the same row and the popularity baseline is
the reference every model is read against.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from recommender.data import CATALOG_SIZE, Interactions
from recommender.models.base import Recommender
from recommender.split import Split


def hit_vector(recommended: np.ndarray, holdout: np.ndarray) -> np.ndarray:
    """Per-user booleans: was that user's held-out item somewhere in their top-K list?

    HitRate@K is the mean of this vector, and everything that asks a *narrower* question —
    the support and profile-length strata (ledger L27, L28, L73) and the paired McNemar
    tests (L74) — is this vector grouped or compared rather than a second definition of a
    hit. One definition, so a stratum can never disagree with the aggregate it sits under.
    """
    if len(holdout) == 0:
        return np.zeros(0, dtype=bool)
    return (recommended == np.asarray(holdout).reshape(-1, 1)).any(axis=1)


#: 95% two-sided normal quantile. Named, because a bare 1.96 in three places is three
#: places to get it wrong.
Z95 = 1.959963984540054


def wilson_interval(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (ledger L74).

    Wilson rather than Wald because these proportions are small — HitRate runs 0.014 to
    0.069 here — and Wald is badly behaved near zero: it is symmetric by construction and
    can reach below it.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired exact McNemar test between two per-user hit vectors (ledger L74).

    Every model here is scored on the *same* users with the *same* held-out book, so the
    unpaired standard error is the wrong instrument: it discards the pairing, and on this
    data the pairing is strong — the users a model can hit at all are largely the users
    whose held-out work carries enough evidence to be reachable (L73). McNemar looks only
    at the users where the two models disagree: ``b`` users that A hit and B missed, ``c``
    the other way round. Under the null that either is equally likely, ``b`` is
    Binomial(b + c, 0.5), and the two-sided exact binomial p on that is the whole test.

    The interval on the difference uses the paired error ``sqrt(b + c) / n``, which is
    exactly the term the unpaired formula overstates for correlated models.
    """
    n = len(a)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    discordant = only_a + only_b
    diff = (only_a - only_b) / n if n else float("nan")
    half = Z95 * math.sqrt(discordant) / n if discordant and n else float("nan")
    return {
        "a_only": only_a,
        "b_only": only_b,
        "discordant": discordant,
        "p": float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else float("nan"),
        "diff": diff,
        "diff_lo": diff - half,
        "diff_hi": diff + half,
    }


def hit_rate_at_k(recommended: np.ndarray, holdout: np.ndarray) -> float:
    """Share of users whose single held-out item is somewhere in their top-K list."""
    if len(holdout) == 0:
        return float("nan")
    return float(hit_vector(recommended, holdout).mean())


def hit_rate_at_k_by_group(
    recommended: np.ndarray,
    holdout: np.ndarray,
    group_of: Callable[[list[str]], np.ndarray],
) -> float:
    """HitRate@K under a *coarser* notion of "correct": same group, not same id.

    ``group_of`` maps ids to group ids — in practice :meth:`recommender.data.Works.of`,
    so a recommended ISBN counts as a hit when it is any edition of the held-out book.

    This exists to answer one question and should not be used for anything else: **how
    much of the work-level re-base is the model getting better, and how much is the ISBN
    keyed metric having been unfair?** Scoring the identical ISBN-level model and the
    identical recommendations under work-level credit isolates the second part exactly,
    because nothing else about the run changes. See ``scripts/decompose_work_level_lift.py``
    and the ledger's decomposition line.

    It is deliberately *not* what the comparison table reports. Under this rule a model can
    fill all ten slots with ten editions of one book and still score a hit, which is a
    generous metric, not a better one — the table's job is to compare models, and this
    function's job is to explain a difference between two tables.
    """
    if len(holdout) == 0:
        return float("nan")
    flat = np.asarray(recommended, dtype=object)
    filled = [isbn for isbn in flat.ravel().tolist() if isbn is not None]
    lookup = dict(zip(filled, group_of(filled).tolist(), strict=True)) if filled else {}
    target = group_of(list(np.asarray(holdout).tolist()))
    hits = [
        any(lookup.get(isbn) == want for isbn in row if isbn is not None)
        for row, want in zip(flat, target.tolist(), strict=True)
    ]
    return float(np.mean(hits))


def catalog_coverage_at_k(recommended: np.ndarray, catalog_isbns: set[str], catalog_size: int = CATALOG_SIZE) -> float:
    """Distinct *catalogue* books recommended to anyone, over the whole catalogue."""
    distinct = {isbn for isbn in np.asarray(recommended).ravel().tolist() if isbn is not None}
    return len(distinct & catalog_isbns) / catalog_size


def novelty_at_k(
    recommended: np.ndarray,
    popularity: dict[str, int],
    total_interactions: int,
    catalog_size: int = CATALOG_SIZE,
) -> float:
    """Mean -log2 smoothed popularity share over every recommended slot."""
    flat = [isbn for isbn in np.asarray(recommended).ravel().tolist() if isbn is not None]
    if not flat:
        return float("nan")
    counts = np.array([popularity.get(isbn, 0) for isbn in flat], dtype=np.float64)
    share = (counts + 1.0) / (total_interactions + catalog_size)
    return float(np.mean(-np.log2(share)))


@dataclass
class EvalResult:
    """One row of the comparison table, with everything needed to reproduce it."""

    model: str
    hit_rate_at_10: float
    coverage_at_10: float
    novelty_at_10: float
    n_users: int
    k: int
    params: str
    seconds: float
    notes: str = ""
    recommendations: np.ndarray | None = field(default=None, repr=False)

    @property
    def precision_at_10(self) -> float:
        return self.hit_rate_at_10 / self.k

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            f"HitRate@{self.k}": round(self.hit_rate_at_10, 4),
            f"Coverage@{self.k}": round(self.coverage_at_10, 5),
            f"Novelty@{self.k}": round(self.novelty_at_10, 2),
            "users": self.n_users,
            "seconds": round(self.seconds, 1),
            "params": self.params,
        }

    def __str__(self) -> str:
        return (
            f"{self.model:<28} HitRate@{self.k}={self.hit_rate_at_10:.4f}  "
            f"Coverage@{self.k}={self.coverage_at_10:.3%}  Novelty@{self.k}={self.novelty_at_10:.2f}  "
            f"({self.n_users:,} users, {self.seconds:.0f}s)"
        )


def evaluate(
    model: Recommender,
    split: Split,
    train: Interactions,
    *,
    catalog_isbns: set[str],
    catalog_size: int = CATALOG_SIZE,
    k: int = 10,
    batch_size: int = 512,
    users: np.ndarray | None = None,
    notes: str = "",
    verbose: bool = False,
) -> EvalResult:
    """Score *model* on the pinned split and return one comparison-table row.

    Args:
        users: restrict evaluation to these User-IDs. Used only for the runtime
            guardrail on long runs; when set, the cap is recorded in
            :attr:`EvalResult.notes` so no ledger line can silently hide it.
        catalog_size: the Coverage@K denominator. 271,360 ISBNs by default; the
            work-level experiment passes its own item universe instead, because a
            coverage percentage is only meaningful against the catalogue it is measured
            over (ledger L44).
    """
    test = split.test if users is None else split.test[split.test["User-ID"].isin(set(users))]
    user_ids = test["User-ID"].to_numpy()
    holdout = test["ISBN"].to_numpy()

    popularity = dict(zip(train.item_ids.tolist(), train.item_popularity.tolist(), strict=True))
    total_interactions = int(train.item_popularity.sum())

    started = time.perf_counter()
    chunks = []
    for start in range(0, len(user_ids), batch_size):
        chunks.append(model.recommend(user_ids[start : start + batch_size], k=k))
        if verbose and (start // batch_size) % 10 == 0:
            print(f"  {model.name}: {min(start + batch_size, len(user_ids)):,}/{len(user_ids):,} users", flush=True)
    recommended = np.concatenate(chunks, axis=0) if chunks else np.empty((0, k), dtype=object)
    seconds = time.perf_counter() - started

    return EvalResult(
        model=model.name,
        hit_rate_at_10=hit_rate_at_k(recommended, holdout),
        coverage_at_10=catalog_coverage_at_k(recommended, catalog_isbns, catalog_size),
        novelty_at_10=novelty_at_k(recommended, popularity, total_interactions, catalog_size),
        n_users=len(user_ids),
        k=k,
        params=model.describe_params(),
        seconds=seconds,
        notes=notes,
        recommendations=recommended,
    )


def comparison_table(results: list[EvalResult]) -> pd.DataFrame:
    """The M10 comparison table: one row per model, identical split, all three metrics."""
    return pd.DataFrame([r.as_row() for r in results])


#: Ceiling on how deep a neighbour list is requested before the asking reader's own train
#: items are filtered out of it. Depth is ``k + len(owned)`` so that k slots survive the
#: filter even for a reader who owns every early neighbour; the cap stops a reader with
#: thousands of interactions from asking for a full sort of the catalogue.
NEIGHBOUR_DEPTH_CAP = 1000


def owned_items(train: Interactions, user_ids: np.ndarray) -> list[frozenset[str]]:
    """Each user's train items, in the order *user_ids* gives them.

    The item-to-item column has to exclude what the reader has already read, because the
    profile column does (:meth:`recommender.models.base.Recommender.recommend` excludes it
    by contract). Comparing a filtered list against an unfiltered one would test the filter
    rather than the query.
    """
    out: list[frozenset[str]] = []
    for user_id in user_ids:
        row = train.user_index.get(int(user_id))
        if row is None:
            out.append(frozenset())
            continue
        lo, hi = train.matrix.indptr[row], train.matrix.indptr[row + 1]
        out.append(frozenset(train.item_ids[train.matrix.indices[lo:hi]].tolist()))
    return out


def neighbour_matrix(
    model: Recommender,
    anchors: np.ndarray,
    *,
    k: int = 10,
    owned: list[frozenset[str]] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k neighbours of each anchor, already filtered, plus which anchors were answered.

    Returns ``(neighbours, answered)``: an ``(n, k)`` object array padded with ``None``,
    and a boolean vector that is True where the model returned *anything at all* for that
    anchor. ``answered`` is taken from the **raw** list, before filtering, so "this model
    has no representation for that book" stays distinguishable from "everything it returned
    was already on the reader's shelf".

    An unanswered anchor is a padded row and therefore a miss, never a dropped user: a
    model that declines to answer must not be rewarded with a smaller denominator, and
    dropping the row would silently break the pairing with the profile column
    (:func:`mcnemar` does not check that its two vectors are the same length).
    """
    n = len(anchors)
    out = np.full((n, k), None, dtype=object)
    answered = np.zeros(n, dtype=bool)
    for row, anchor in enumerate(anchors):
        if not isinstance(anchor, str):
            continue
        seen = owned[row] if owned is not None else frozenset()
        raw = model.similar_items(anchor, k=min(k + len(seen), NEIGHBOUR_DEPTH_CAP))
        answered[row] = bool(raw)
        picked = [item for item, _ in raw if item not in seen][:k]
        out[row, : len(picked)] = picked
        if verbose and row and row % 2000 == 0:
            print(f"  {model.name}: {row:,}/{n:,} anchors", flush=True)
    return out, answered


@dataclass
class AnchorEvalResult:
    """One row of the item-to-item table. Deliberately not an :class:`EvalResult`.

    A different query answers a different question, and giving it the same type would let
    the two be concatenated into one table by accident — which is the mistake ledger L46
    records for the two item bases. The field names say ``anchor`` for the same reason.
    """

    model: str
    anchor_hit_rate: float
    answered: float
    coverage: float
    novelty: float
    n_users: int
    k: int
    params: str
    seconds: float
    notes: str = ""
    anchors: np.ndarray | None = field(default=None, repr=False)
    neighbours: np.ndarray | None = field(default=None, repr=False)
    hits: np.ndarray | None = field(default=None, repr=False)

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            f"AnchorHitRate@{self.k}": round(self.anchor_hit_rate, 4),
            "answered": round(self.answered, 4),
            f"AnchorCoverage@{self.k}": round(self.coverage, 5),
            f"Novelty@{self.k}": round(self.novelty, 2),
            "users": self.n_users,
            "seconds": round(self.seconds, 1),
            "params": self.params,
        }

    def __str__(self) -> str:
        return (
            f"{self.model:<28} AnchorHitRate@{self.k}={self.anchor_hit_rate:.4f}  "
            f"answered={self.answered:.1%}  Coverage@{self.k}={self.coverage:.3%}  "
            f"Novelty@{self.k}={self.novelty:.2f}  ({self.n_users:,} anchors, {self.seconds:.0f}s)"
        )


def evaluate_anchors(
    model: Recommender,
    split: Split,
    train: Interactions,
    anchors: np.ndarray,
    *,
    catalog_isbns: set[str],
    catalog_size: int = CATALOG_SIZE,
    k: int = 10,
    owned: list[frozenset[str]] | None = None,
    notes: str = "",
    verbose: bool = False,
) -> AnchorEvalResult:
    """Score *model* on the item-to-item question and return one row of that table.

    *anchors* must be aligned row-for-row with ``split.test`` — see
    :class:`recommender.split.Anchors`. The hit is
    :func:`hit_vector` unchanged: one definition of a hit, so this column can never
    disagree with the one it is compared against.
    """
    holdout = split.test["ISBN"].to_numpy()
    if len(anchors) != len(holdout):
        raise ValueError(f"{len(anchors)} anchors against {len(holdout)} held-out items — the pairing is broken")
    popularity = dict(zip(train.item_ids.tolist(), train.item_popularity.tolist(), strict=True))
    total_interactions = int(train.item_popularity.sum())

    started = time.perf_counter()
    neighbours, answered = neighbour_matrix(model, anchors, k=k, owned=owned, verbose=verbose)
    seconds = time.perf_counter() - started

    hits = hit_vector(neighbours, holdout)
    return AnchorEvalResult(
        model=model.name,
        anchor_hit_rate=float(hits.mean()) if len(hits) else float("nan"),
        answered=float(answered.mean()) if len(answered) else float("nan"),
        coverage=catalog_coverage_at_k(neighbours, catalog_isbns, catalog_size),
        novelty=novelty_at_k(neighbours, popularity, total_interactions, catalog_size),
        n_users=len(holdout),
        k=k,
        params=model.describe_params(),
        seconds=seconds,
        notes=notes,
        anchors=anchors,
        neighbours=neighbours,
        hits=hits,
    )


def anchor_comparison_table(results: list[AnchorEvalResult]) -> pd.DataFrame:
    """The M20 table: one row per model, identical split, identical anchors."""
    return pd.DataFrame([r.as_row() for r in results])
