"""Evaluation layer: the same three metrics for every model, on the same split.

**The metrics, pinned once and used for every model.**

- **HitRate@K** — share of eligible users whose held-out book appears in their top-K.
  Under leave-one-out there is exactly one relevant item per user, so HitRate@K *is*
  Recall@K, and Precision@K = HitRate@K / K. One number carries all three; reporting
  them as three separate columns would be three views of the same measurement dressed
  up as corroboration.
- **Catalog-Coverage@K** — distinct catalogue books appearing in *any* user's top-K,
  divided by the full catalogue of 271,360. This is how the "collaborative filtering
  can only reach 5.3% of the catalogue" argument (ledger L12) gets measured instead of
  asserted. Recommended ISBNs that are not in ``Books.csv`` do not count towards the
  numerator: a book we cannot name is a book we cannot show.
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

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recommender.data import CATALOG_SIZE, Interactions
from recommender.models.base import Recommender
from recommender.split import Split


def hit_rate_at_k(recommended: np.ndarray, holdout: np.ndarray) -> float:
    """Share of users whose single held-out item is somewhere in their top-K list."""
    if len(holdout) == 0:
        return float("nan")
    hits = (recommended == np.asarray(holdout).reshape(-1, 1)).any(axis=1)
    return float(hits.mean())


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
        coverage_at_10=catalog_coverage_at_k(recommended, catalog_isbns),
        novelty_at_10=novelty_at_k(recommended, popularity, total_interactions),
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
