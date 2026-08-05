"""Item-item collaborative filtering — the core hypothesis of this PoC.

The product takes a book and returns similar books, with no user identity at query time.
Item-item CF answers exactly that question natively: the model *is* a table of book
neighbourhoods. User-based CF would answer a different question and then need a user to
ask it, which the app does not have.

**Similarity.** Shrunk cosine over the binarized interaction matrix:

    sim(i, j) = co_occurrence(i, j) / (sqrt(support_i * support_j) + shrinkage)

Plain cosine is hostile on this dataset. 58% of books carry exactly one rating (ledger
L7), and two books that share their single reader score a perfect 1.0 — a similarity
built from one coin flip outranks a similarity built from four hundred. The shrinkage
term is the standard fix (Bell & Koren): it costs popular pairs almost nothing and
divides one-off pairs by a constant they cannot compete with. It is the prepared answer
to "how do you handle sparsity", and it is a one-line answer with a measurement behind it.

**Minimum support, and why it is 1.** A support threshold is the other common defence
against chance co-occurrence, and this implementation supports one — but the measured
cost of using it is severe. Raising it to 5 would drop 87% of the items and, with them,
the reachable share of held-out books from 84.8% to 64.5% (ledger L20, L23): twenty
points of achievable accuracy spent on a problem shrinkage already solves continuously.
So the default keeps every item that appears in train at least once and lets shrinkage
do the work.

**Scoring a user** is the sum of shrunk similarities from every book in the user's train
profile to the candidate, which is the standard item-KNN aggregation. Only each item's
``top_k_neighbours`` nearest neighbours are kept, both to bound memory and because the
tail of a similarity row is noise.

**The defaults were chosen on a validation split carved out of train**
(``scripts/tune_item_item.py``, seed 43), never on the evaluation holdout — tuning on the
test split and then reporting the best cell of the sweep is the offline equivalent of
marking your own homework. The sweep is also a result in itself (ledger L25): at
shrinkage 0 the model scores HitRate@10 0.0296 with 17.5% coverage, at shrinkage 10 it
scores 0.0532 with 7.6%. Damping coincidental co-occurrence nearly doubles accuracy and
costs more than half the catalogue reach — the accuracy/coverage tension in one table,
which is exactly why the hybrid argument is about coverage rather than accuracy.

**The explicit-only ablation** (``signal="explicit"``) is the same model fitted on a
matrix built from graded ratings alone. It exists to turn the pinned signal decision into
a measurement instead of an assumption. Note that it excludes only *explicit* items from
its recommendations: in the counterfactual world where we discarded the implicit data, we
would not know those interactions happened at all.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recommender.data import BookCrossing, Interactions
from recommender.models.base import Recommender


def shrunk_cosine_neighbours(
    matrix: sp.csr_matrix,
    *,
    shrinkage: float,
    top_k: int,
    min_support: int = 1,
    block: int = 4096,
) -> sp.csr_matrix:
    """Item-item shrunk-cosine similarity, truncated to *top_k* neighbours per item.

    Computed in blocks of item rows so the transient co-occurrence matrix never has to
    fit in memory whole. Returns a CSR matrix over the same item index space as
    *matrix*, with an all-zero row for any item below *min_support*.
    """
    n_items = matrix.shape[1]
    support = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.float64)
    norm = np.sqrt(support)
    eligible = support >= min_support

    columns = matrix.T.tocsr()
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    vals_out: list[np.ndarray] = []

    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        co = (columns[start:stop] @ matrix).tocsr()
        if co.nnz == 0:
            continue
        row_of = np.repeat(np.arange(start, stop), np.diff(co.indptr))
        sims = co.data / (norm[row_of] * norm[co.indices] + shrinkage)
        # An item is not its own neighbour, and sub-threshold items get no row.
        sims[row_of == co.indices] = 0.0
        sims[~eligible[row_of]] = 0.0
        sims[~eligible[co.indices]] = 0.0

        for local, global_row in enumerate(range(start, stop)):
            lo, hi = co.indptr[local], co.indptr[local + 1]
            row_sims = sims[lo:hi]
            keep = np.flatnonzero(row_sims > 0)
            if keep.size == 0:
                continue
            if keep.size > top_k:
                keep = keep[np.argpartition(-row_sims[keep], kth=top_k - 1)[:top_k]]
            rows_out.append(np.full(keep.size, global_row, dtype=np.int32))
            cols_out.append(co.indices[lo:hi][keep].astype(np.int32))
            vals_out.append(row_sims[keep].astype(np.float32))

    if not rows_out:
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    similarity = sp.coo_matrix(
        (np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=(n_items, n_items),
        dtype=np.float32,
    ).tocsr()
    return similarity


class ItemItemRecommender(Recommender):
    """Shrunk-cosine item-item CF over the binarized train matrix."""

    def __init__(
        self,
        *,
        shrinkage: float = 10.0,
        top_k_neighbours: int = 50,
        min_support: int = 1,
        signal: str = "binary",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name or "item-item"
        self.shrinkage = shrinkage
        self.top_k_neighbours = top_k_neighbours
        self.min_support = min_support
        self.signal = signal
        self.params = {
            "signal": signal,
            "shrinkage": shrinkage,
            "top_k_neighbours": top_k_neighbours,
            "min_support": min_support,
            "scoring": "sum of shrunk similarities over the user's train items",
        }
        self.similarity: sp.csr_matrix | None = None

    def fit(self, train: Interactions, catalog: BookCrossing) -> ItemItemRecommender:
        self.train = train
        self.similarity = shrunk_cosine_neighbours(
            train.matrix,
            shrinkage=self.shrinkage,
            top_k=self.top_k_neighbours,
            min_support=self.min_support,
        )
        return self

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        train = self._require_fit()
        rows = [train.user_index.get(int(u)) for u in user_ids]
        known = [r for r in rows if r is not None]
        out = np.full((len(user_ids), k), None, dtype=object)
        if not known:
            return out

        profiles = train.matrix[known]
        scores = (profiles @ self.similarity).tocsr()

        position = 0
        for row_index, user_row in enumerate(rows):
            if user_row is None:
                continue
            lo, hi = scores.indptr[position], scores.indptr[position + 1]
            position += 1
            candidates, values = scores.indices[lo:hi], scores.data[lo:hi]
            if candidates.size == 0:
                continue
            seen_lo, seen_hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
            unseen = ~np.isin(candidates, train.matrix.indices[seen_lo:seen_hi], assume_unique=False)
            candidates, values = candidates[unseen], values[unseen]
            if candidates.size == 0:
                continue
            take = min(k, candidates.size)
            best = np.argpartition(-values, kth=take - 1)[:take]
            best = best[np.argsort(-values[best], kind="stable")]
            out[row_index, :take] = train.item_ids[candidates[best]]
        return out

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        train = self._require_fit()
        item = train.item_index.get(isbn)
        if item is None:
            return []
        row = self.similarity[item]
        if row.nnz == 0:
            return []
        take = min(k, row.nnz)
        best = np.argpartition(-row.data, kth=take - 1)[:take]
        best = best[np.argsort(-row.data[best], kind="stable")]
        return [(train.item_ids[row.indices[i]], float(row.data[i])) for i in best]
