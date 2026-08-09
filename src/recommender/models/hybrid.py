"""The hybrid, measured rather than recommended from a ceiling (milestone M19).

**Why this module exists at all.** Since M10 the project's headline recommendation has been
"item-item as the scoring core, with the content layer serving the catalogue it structurally
cannot reach", and until M19 that sentence rested on two *bounds*: ledger L50 (the union of
both model classes raises the achievable ceiling from 86.66% to 95.34%) and L60 (the two
classes reach largely different parts of the catalogue). Neither is a HitRate. Recommending
a system on two bounds is defensible; being asked "so how does it score?" and having no row
is not.

**Three rules, because the choice between them is the actual question.** All three combine
the *same* two fitted models — item-item as the collaborative side, TF-IDF as the content
side — and none of them re-fits anything:

- :data:`CASCADE` — **backfill.** The collaborative model fills the list; where it returns
  fewer than k candidates, or where a slot would go to an item whose train support is below
  ``support_floor``, the content model fills the remainder. This is the rule the ledger's
  recommendation actually describes, so it is the one that must be measured whatever the
  others do. At ``support_floor=0`` it touches only the users item-item cannot fill at all,
  which makes it the most conservative of the three by construction.
- :data:`FUSION` — **score fusion**, ``α · norm(collaborative) + (1-α) · norm(content)``.
  The two scores are not comparable — one is a sum of shrunk cosines over a profile, the
  other a mean cosine in [0, 1] — so each is min-max normalized **within each user's own
  candidate list** before mixing. α is tuned on the inner validation split (seed 43), never
  on the test holdout, and the whole curve is reported rather than the argmax.
- :data:`RRF` — **reciprocal rank fusion**, ``Σ 1/(rrf_k + rank)``, the parameter-free
  reference. It ignores the scores entirely and uses only the ranks, which is precisely why
  it is the control: if a tuned fusion cannot beat RRF, the tuning bought nothing.

**Normalization is a choice with a failure mode, and it is stated rather than hidden.**
Min-max within a user's candidate list maps that user's best candidate to 1.0 and their
worst to 0.0 *for each model separately*. So a user whose collaborative candidates are all
weak still has a 1.0 at the top of that list. That is deliberate — the alternative, a global
scale, would make the fusion a function of profile length, which is the L28 defect wearing a
different hat — but it means the fused score answers "how good is this candidate *for this
user, relative to what this model found for them*", and nothing more. An item outside a
model's top ``candidates`` list scores 0 from that model rather than being treated as
missing, because a candidate the model never surfaced is evidence of absence at this depth.

**The candidate depth is a parameter and it bounds every rule.** All three see each model's
top ``candidates`` (default 100) rather than the full item universe: scoring 235,824 works
densely per user for two models is what makes the honest version of this expensive, and the
tail below rank 100 cannot reach a top-10 under any of these rules except by displacing
something already there. That is a real limit on score fusion at extreme α and it is
recorded in the ledger line rather than argued away.

**What this module deliberately does not do.** It does not re-fit, re-tune or touch either
base model — the six published rows must reproduce to the digit beside it, and
``scripts/measure_hybrid.py`` checks exactly that before it reports anything. It also does
not dedup: at work level the item already *is* a work (M12), so there is nothing to collapse.
"""

from __future__ import annotations

import numpy as np

from recommender.data import BookCrossing, Interactions
from recommender.models.base import Recommender

CASCADE = "cascade"
FUSION = "fusion"
RRF = "rrf"
RULES = (CASCADE, FUSION, RRF)

#: How deep each base model's candidate list goes before the rules see it. 100 is ten times
#: the reported k, which is enough for any of these rules to reorder a top-10 completely.
DEFAULT_CANDIDATES = 100

#: The constant in reciprocal rank fusion. 60 is the value from the original TREC work and
#: is left alone on purpose: RRF is here as the *parameter-free* reference, so tuning it
#: would defeat the reason it is in the comparison.
DEFAULT_RRF_K = 60


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max a 1-D score vector into [0, 1]; a flat or empty vector becomes all 1.0.

    Flat means every candidate this model found is equally good *for this user*, so 1.0 is
    the honest encoding — mapping them all to 0.0 would silently hand the whole list to the
    other model, which is a combination rule nobody chose.
    """
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def combine_user(
    cf_ids: np.ndarray,
    cf_scores: np.ndarray,
    ct_ids: np.ndarray,
    ct_scores: np.ndarray,
    *,
    rule: str,
    k: int,
    alpha: float = 0.5,
    rrf_k: int = DEFAULT_RRF_K,
    support: dict[str, int] | None = None,
    support_floor: int = 0,
) -> list[str]:
    """Combine one user's two candidate lists into one top-k list of ids.

    This is the whole ranking rule, and it is a free function on purpose: both
    :class:`HybridRecommender` and ``scripts/measure_hybrid.py`` call it, so the model class
    and the measurement can never drift into ranking two different ways. Inputs are already
    ranked best-first with ``None`` / ``-inf`` padding, exactly as
    :meth:`Recommender.recommend_scored` returns them.
    """
    cf = [(i, s) for i, s in zip(cf_ids.tolist(), cf_scores.tolist(), strict=True) if i is not None]
    ct = [(i, s) for i, s in zip(ct_ids.tolist(), ct_scores.tolist(), strict=True) if i is not None]

    if rule == CASCADE:
        # The collaborative list wins every slot it is entitled to; the content model is a
        # filler, not a voter. A slot is forfeited when the item is below the support floor,
        # which is the "thin evidence" half of the rule the ledger describes.
        kept = [i for i, _ in cf if support is None or support.get(i, 0) >= support_floor]
        out = kept[:k]
        if len(out) < k:
            seen = set(out)
            for i, _ in ct:
                if i not in seen:
                    out.append(i)
                    seen.add(i)
                    if len(out) == k:
                        break
        return out

    if rule == RRF:
        fused: dict[str, float] = {}
        for ranked in (cf, ct):
            for rank, (i, _) in enumerate(ranked, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
    elif rule == FUSION:
        cf_norm = _normalize(np.array([s for _, s in cf], dtype=np.float64))
        ct_norm = _normalize(np.array([s for _, s in ct], dtype=np.float64))
        fused = {}
        for (i, _), value in zip(cf, cf_norm.tolist(), strict=True):
            fused[i] = fused.get(i, 0.0) + alpha * value
        for (i, _), value in zip(ct, ct_norm.tolist(), strict=True):
            fused[i] = fused.get(i, 0.0) + (1.0 - alpha) * value
    else:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")

    # Ties broken by the collaborative model's order, then the content model's — a stable,
    # stated rule, because on this data ties are common and "whatever dict order gives" is
    # not a ranking anybody could reproduce.
    order = {i: rank for rank, (i, _) in enumerate(cf)}
    tiebreak = {i: rank for rank, (i, _) in enumerate(ct)}
    ranked_ids = sorted(fused, key=lambda i: (-fused[i], order.get(i, 10**6), tiebreak.get(i, 10**6)))
    return ranked_ids[:k]


class HybridRecommender(Recommender):
    """Two fitted models, one combination rule, no re-fitting."""

    def __init__(
        self,
        collaborative: Recommender,
        content: Recommender,
        *,
        rule: str = CASCADE,
        alpha: float = 0.5,
        rrf_k: int = DEFAULT_RRF_K,
        candidates: int = DEFAULT_CANDIDATES,
        support_floor: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
        self.collaborative = collaborative
        self.content = content
        self.rule = rule
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.support_floor = support_floor
        self.name = name or f"hybrid ({rule})"
        self.params = {
            "rule": rule,
            "collaborative": collaborative.name,
            "content": content.name,
            "candidates": candidates,
            **({"alpha": alpha} if rule == FUSION else {}),
            **({"rrf_k": rrf_k} if rule == RRF else {}),
            **({"support_floor": support_floor} if rule == CASCADE else {}),
        }
        # Wrapping two already-fitted models is the normal case: a hybrid is a decision
        # taken after training, and requiring a re-fit would change the thing measured.
        self.train = collaborative.train
        self._support: dict[str, int] | None = None

    def fit(self, train: Interactions, catalog: BookCrossing) -> HybridRecommender:
        self.collaborative.fit(train, catalog)
        self.content.fit(train, catalog)
        self.train = self.collaborative.train
        return self

    def item_support(self) -> dict[str, int]:
        """Train interaction count per item — what the cascade's support floor reads."""
        if self._support is None:
            train = self._require_fit()
            self._support = dict(zip(train.item_ids.tolist(), train.item_popularity.tolist(), strict=True))
        return self._support

    def describe_params(self) -> str:
        return f"{self.rule}; {self.collaborative.describe_params()} || {self.content.describe_params()}"

    def recommend(self, user_ids: np.ndarray, k: int = 10) -> np.ndarray:
        cf_ids, cf_scores = self.collaborative.recommend_scored(user_ids, k=self.candidates)
        ct_ids, ct_scores = self.content.recommend_scored(user_ids, k=self.candidates)
        support = self.item_support() if self.rule == CASCADE and self.support_floor else None

        out = np.full((len(user_ids), k), None, dtype=object)
        for row in range(len(user_ids)):
            picked = combine_user(
                cf_ids[row],
                cf_scores[row],
                ct_ids[row],
                ct_scores[row],
                rule=self.rule,
                k=k,
                alpha=self.alpha,
                rrf_k=self.rrf_k,
                support=support,
                support_floor=self.support_floor,
            )
            out[row, : len(picked)] = picked
        return out

    def similar_items(self, isbn: str, k: int = 10) -> list[tuple[str, float]]:
        """The same rule applied item-to-item, which is what the gallery and the app see.

        The score returned is the fused one for :data:`FUSION` and :data:`RRF`, and the
        contributing model's own score for :data:`CASCADE` — under a cascade nothing is
        fused, so inventing a combined number would be a worse answer than the real one.
        """
        cf = self.collaborative.similar_items(isbn, k=self.candidates)
        ct = self.content.similar_items(isbn, k=self.candidates)
        if not cf and not ct:
            return []

        cf_ids = np.array([i for i, _ in cf] + [None] * 0, dtype=object)
        cf_scores = np.array([s for _, s in cf], dtype=np.float64)
        ct_ids = np.array([i for i, _ in ct], dtype=object)
        ct_scores = np.array([s for _, s in ct], dtype=np.float64)
        picked = combine_user(
            cf_ids,
            cf_scores,
            ct_ids,
            ct_scores,
            rule=self.rule,
            k=k,
            alpha=self.alpha,
            rrf_k=self.rrf_k,
            support=self.item_support() if self.rule == CASCADE and self.support_floor else None,
            support_floor=self.support_floor,
        )
        known = dict(cf)
        known.update({i: s for i, s in ct if i not in known})
        return [(i, float(known.get(i, 0.0))) for i in picked]
