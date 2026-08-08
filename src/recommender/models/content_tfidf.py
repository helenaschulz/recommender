"""Content-based similarity over title + author — the coverage layer.

**The problem this exists to solve, quantified.** Collaborative filtering can only rank
books that somebody has already interacted with. On this dataset that ceiling is hard:
15.2% of held-out books appear nowhere in the train matrix (ledger L20), and after the
standard min-5 filter a CF model reaches 5.3% of the catalogue (L12). A content model
has no such limit — it can score a book nobody has ever touched, because it reads the
title. Coverage@10 is therefore the headline metric here, not HitRate.

**The representation.** TF-IDF over character n-grams of ``"title author"``. Character
n-grams rather than words, deliberately: the catalogue is multilingual (German, French
and Spanish titles sit next to English ones), riddled with edition variants of the same
work (ledger L40 — 24,392 works spread over 59,928 ISBNs), and full of punctuation and
subtitle noise like ``(Bestselling Backlist)``. Character n-grams degrade gracefully
across all three, where a word-level vocabulary would treat *Ringe* and *Rings* as
unrelated tokens and miss most edition duplicates.

**What the item is.** This module vectorizes whatever ``catalog.books`` gives it, one row
at a time, so it runs unchanged at either item level: at ISBN level that is one vector per
edition, at work level one vector per work carrying the title and author of its
most-interacted edition (:func:`recommender.data.work_level_catalog`). The difference is
not cosmetic for *this* model in particular — see the decomposition in ledger L58.

**The honest limitation.** 10.3% of ratings point at ISBNs with no row in ``Books.csv``
(ledger L14). Those books have no title to vectorize, so this model can never score them
— not badly, but *at all*. The content layer raises the reachable ceiling from 84.8% to
89.3% (L21); it does not remove it.

**Scoring a user** is the mean cosine similarity from the user's train books to the
candidate. The mean rather than the sum, because a user with 300 interactions should not
have their profile dominated by sheer volume, and because content similarity is already
scale-free.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recommender.data import BookCrossing, Interactions
from recommender.models.base import Recommender, top_k_from_scores


def content_text(books) -> np.ndarray:
    """``"title author"`` per catalogue row, lower-cased, missing fields left empty."""
    title = books["Book-Title"].fillna("").astype(str)
    author = books["Book-Author"].fillna("").astype(str)
    return (title + " " + author).str.lower().str.strip().to_numpy()


class TfidfRecommender(Recommender):
    """Character n-gram TF-IDF over title+author, cosine similarity, whole catalogue."""

    name = "content-tfidf"

    def __init__(
        self,
        *,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 3,
        max_features: int = 300_000,
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.batch_size = batch_size
        self.params = {
            "text": "title + author, lower-cased",
            "analyzer": analyzer,
            "ngram_range": str(ngram_range),
            "min_df": min_df,
            "max_features": max_features,
            "scoring": "mean cosine to the user's train items",
        }
        self.vectors: sp.csr_matrix | None = None
        self.item_ids: np.ndarray | None = None
        self._index: dict[str, int] = {}

    def fit(self, train: Interactions, catalog: BookCrossing) -> TfidfRecommender:
        self.train = train
        # The candidate universe is the whole catalogue, including books with zero
        # interactions -- that is the entire point of a content layer.
        books = catalog.books
        keep = books["Book-Title"].notna()
        books = books.loc[keep]
        self.item_ids = books["ISBN"].to_numpy()
        self._index = {isbn: i for i, isbn in enumerate(self.item_ids)}

        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.vectors = normalize(vectorizer.fit_transform(content_text(books)))
        self.params["catalogue_vectorized"] = len(self.item_ids)
        self.params["vocabulary"] = len(vectorizer.vocabulary_)
        return self

    def _profile(self, user_row: int) -> sp.csr_matrix | None:
        """Mean TF-IDF vector of the books a user interacted with in train."""
        train = self._require_fit()
        lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
        isbns = train.item_ids[train.matrix.indices[lo:hi]]
        rows = [self._index[isbn] for isbn in isbns.tolist() if isbn in self._index]
        if not rows:
            return None
        return sp.csr_matrix(self.vectors[rows].mean(axis=0))

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        train = self._require_fit()
        out = np.full((len(user_ids), k), None, dtype=object)

        profiles, targets, blocked = [], [], []
        for row, user_id in enumerate(user_ids):
            user_row = train.user_index.get(int(user_id))
            if user_row is None:
                continue
            profile = self._profile(user_row)
            if profile is None:
                continue
            lo, hi = train.matrix.indptr[user_row], train.matrix.indptr[user_row + 1]
            seen = train.item_ids[train.matrix.indices[lo:hi]]
            profiles.append(profile)
            targets.append(row)
            blocked.append(np.array([self._index[i] for i in seen.tolist() if i in self._index], dtype=np.int64))

        for start in range(0, len(profiles), self.batch_size):
            chunk = profiles[start : start + self.batch_size]
            scores = np.asarray((sp.vstack(chunk) @ self.vectors.T).todense())
            picked = top_k_from_scores(scores, k, blocked=blocked[start : start + self.batch_size])
            for local, row in enumerate(targets[start : start + self.batch_size]):
                chosen = picked[local]
                usable = chosen >= 0
                out[row, : usable.sum()] = self.item_ids[chosen[usable]]
        return out

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        self._require_fit()
        item = self._index.get(isbn)
        if item is None:
            return []
        scores = np.asarray((self.vectors[item] @ self.vectors.T).todense()).ravel()
        scores[item] = -np.inf
        take = min(k, scores.size - 1)
        best = np.argpartition(-scores, kth=take - 1)[:take]
        best = best[np.argsort(-scores[best], kind="stable")]
        return [(self.item_ids[i], float(scores[i])) for i in best if np.isfinite(scores[i])]
