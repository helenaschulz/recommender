"""Book-Crossing book recommender: the modelling and evaluation package.

Layers, deliberately separate: :mod:`~recommender.data` prepares,
:mod:`~recommender.split` defines the one evaluation split, :mod:`~recommender.models`
score, :mod:`~recommender.eval` measures, :mod:`~recommender.gallery` shows.

Nothing here computes a statistic from held-out data. That discipline is the reason the
numbers in ``docs/RESULTS.md`` are worth citing.
"""

__all__ = ["data", "eval", "gallery", "models", "split"]
