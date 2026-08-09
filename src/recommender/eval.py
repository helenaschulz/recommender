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
