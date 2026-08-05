"""Candidate models. One module per model, all behind the :class:`Recommender` interface.

The ladder, in the order the milestones build it:

- :mod:`~recommender.models.popularity` — the reference row. Everything else is only
  interpretable against it.
- :mod:`~recommender.models.item_item` — the core hypothesis: shrunk cosine over the
  binarized interaction matrix.
- :mod:`~recommender.models.content_tfidf` — the coverage layer over title+author.
- :mod:`~recommender.models.als` — the same story one rung up: item factors are
  embeddings, similarity is a dot product, and it carries to Spark for productionization.
- :mod:`~recommender.models.embeddings` — multilingual sentence embeddings, coverage of
  the whole catalogue from day one.
"""

import pandas as pd

from recommender.data import BookCrossing, Interactions, build_interactions
from recommender.models.base import Recommender, top_k_from_scores

__all__ = ["Recommender", "fit_model", "top_k_from_scores"]


def fit_model(
    model: Recommender,
    train: Interactions,
    catalog: BookCrossing,
    train_ratings: pd.DataFrame,
) -> Recommender:
    """Fit *model* on the right view of the training data.

    Two models want something other than the plain binarized matrix, and this is the one
    place that knows it. It exists because it once did not: the runner special-cased both
    while the notebook did not, so the notebook silently reported the explicit-only
    ablation as an exact duplicate of the binarized run. Any dispatch worth writing twice
    is worth writing once.

    - ``signal="explicit"`` (the item-item ablation) is fitted on a matrix built from
      graded interactions alone, over the *same* item and user index space so the two
      models stay comparable coordinate for coordinate.
    - ALS additionally needs the train ratings frame for its ``1 + alpha * rating``
      confidence weights. It is passed explicitly and never read from ``catalog.ratings``,
      which still contains the holdout.
    """
    fit_matrix = train
    if getattr(model, "signal", "binary") == "explicit":
        fit_matrix = build_interactions(
            train_ratings[train_ratings["is_explicit"]],
            weights="binary",
            item_ids=train.item_ids,
            user_ids=train.user_ids,
        )

    if type(model).__name__ == "ALSRecommender":
        return model.fit(fit_matrix, catalog, ratings=train_ratings)
    return model.fit(fit_matrix, catalog)
