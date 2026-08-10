# Model cheat sheet

One page. Terms in **bold** are the ones to say out loud. Every number traces to `RESULTS.md`.

---

## 1 · The taxonomy, on two axes

**Axis 1 — what data does the model see?**

| | Sees | Blind to | Learns from |
|---|---|---|---|
| **Collaborative filtering (CF)** | the interaction matrix | what the book *is* | behaviour of other readers |
| **Content-based filtering** | item attributes (title, author) | other readers | similarity of the items themselves |

**Axis 2 — how does it process that data?** (only meaningful inside CF)

- **Memory-based** (a.k.a. **neighborhood-based**) — computes similarities directly on the raw matrix. No parameters are learned; the model *is* a similarity table.
- **Model-based** — compresses the matrix into learned parameters, usually **latent factors**. Fitting is an optimization problem.

**A third axis that is NOT a model category:** **explicit vs implicit feedback** is a property of the *data*, not of the model. Rating 8 is explicit; rating 0 is an interaction without a judgement, i.e. implicit. Every CF model handles both, differently: item-item **binarizes**, ALS uses the rating as a **confidence weight**.

---

## 2 · The ladder, with settings and results

Work level, 235,824 works, 1,129,755 train interactions, 13,580 held-out works. One split, one run.

| Model | Category | Key settings | HitRate@10 | Coverage@10 | Novelty@10 |
|---|---|---|---:|---:|---:|
| popularity | non-personalized **baseline** | — | 0.0155 | 0.027% | 10.54 |
| **item-item CF** | CF, memory-based | `shrinkage λ=10`, `top_k_neighbours=50`, binarized | **0.0644** | 8.190% | 14.17 |
| ALS | CF, model-based (**matrix factorization**) | `factors=128`, `α=1`, `regularization=0.05`, `iterations=20`, `similar_min_support=20` | 0.0545 | 0.897% | 12.29 |
| item-item, explicit-only | CF, memory-based (**ablation**) | same, `signal="explicit"` | 0.0486 | 10.036% | 15.97 |
| content TF-IDF | content-based, **sparse** | `analyzer="char_wb"`, `ngram_range=(3,5)`, `min_df=3`, `max_features=300k` | 0.0405 | 16.806% | 17.07 |
| content embeddings | content-based, **dense** | `paraphrase-multilingual-MiniLM-L12-v2`, `center=True` | 0.0141 | **26.143%** | **18.34** |
| *structural ceiling* | — | — | *0.8666* | — | — |

**The shape is the finding:** accuracy and coverage run in opposite directions. No model wins both. The choice is a product decision, not a leaderboard.

---

## 3 · What each one actually does

**Popularity** — sort by interaction count, hand everyone the same list. Not a recommender, a **reference point**. Without it you cannot say whether 0.0644 is good.

**Item-item CF** — two books are similar if the same people read both (**co-occurrence**), normalized as **shrunk cosine**:

```
sim(i,j) = co(i,j) / (sqrt(support_i · support_j) + λ)
```

λ is the **shrinkage**, and it is why this works. Without it, two books sharing their single reader score a perfect 1.0 and outrank a pair with 400 shared readers. Fits the product question natively: the model *is* a table of neighbourhoods.

**ALS (Alternating Least Squares)** — **matrix factorization**: approximate `R ≈ U · Vᵀ` with 128 **latent factors**. Every book becomes a 128-dim vector; similarity is a dot product. "Alternating" = hold `U` fixed and solve `V` in closed form, then swap. That is why it parallelizes, and why it has a first-class Spark implementation (the Part 3 argument). Ratings enter as **confidence weights** `1 + α·rating` — note what this does *not* claim: a low rating is not a negative signal, only weaker positive evidence.

**Item-item explicit-only** — same model on a graded-ratings-only matrix. Not an alternative, an **ablation**: it turns "keep the implicit interactions" from an assumption into a measurement. Costs ~25% of hit rate, gains coverage.

**TF-IDF** — **Term Frequency × Inverse Document Frequency**: each book becomes a sparse vector over text features, rare features weigh more. Uses **character n-grams** rather than words because the catalogue is multilingual and full of edition variants: *Ringe* and *Rings* share n-grams, as words they are unrelated.

**Embeddings** — a **sentence transformer** maps title+author into a dense space; multilingual training puts *Der Herr der Ringe* near *The Lord of the Rings* with no shared characters. Known trap, measured here: sentence embeddings share a large **common component** (**anisotropy**), so averaged profile vectors all point the same way. Fix is **centering** — subtract the global mean, renormalize.

---

## 4 · Why the category predicts the weakness

| | CF | Content-based |
|---|---|---|
| **Item cold start** (new book, no interactions) | structurally blind | works |
| Fine-grained quality | strong, learns real behaviour | weak, a title is not quality |
| **Popularity bias** | amplifies it, MF worst | neutral |
| **Long tail** reach | poor | good |

The **ceilings** are exactly this table as a number: 86.66% of held-out works are reachable by *any* CF model, 95.34% by the union of both classes. That gap is the hybrid argument stated as a bound, not a slogan.

---

## 5 · Hybrid — three rules, one taxonomy

All three combine the **same two fitted models** (item-item + TF-IDF). Nothing is re-fitted. Candidate depth `100` for every rule.

| Rule | What it does | Setting | HitRate@10 | Coverage@10 |
|---|---|---|---:|---:|
| **Cascade** (backfill) | CF fills the list, content fills only the gaps. Content is a *filler*, not a voter | `support_floor=0` | 0.0650 | 8.529% |
| **Weighted / score fusion** | `α · norm(CF) + (1−α) · norm(content)`, both vote everywhere | `α=0.6`, per-user min-max | **0.0690** | **11.912%** |
| **RRF** (Reciprocal Rank Fusion) | `Σ 1/(60 + rank)` — ignores scores, uses ranks only | `rrf_k=60`, no tuning | ≈ fusion | — |

Two results worth saying out loud:

- **Cascade is tiny and free**: +9 readers of 13,580, and **0 losses** — structural, because it only acts where item-item is silent. 8 of the 9 gains are at zero support.
- **The tuning bought nothing**: fusion at a tuned α cannot be told apart from parameter-free RRF (p = 0.867). So the honest description is "combine the two rankings", not "combine them at 0.6".

**Normalization matters and has a failure mode.** Min-max is applied *within each user's own candidate list, per model*, because the two scores live on different scales. A global scale would make fusion a function of profile length.

---

## 6 · The evaluation setup

**Split** — per-user **leave-one-out**, `seed=42`. Not LOO cross-validation: one item is held out *per user*, all at once, one training run.

- eligible = `≥5 explicit ratings` **and** `≥1 rating ≥8` → 13,580 readers
- holdout = 1 item per eligible user, drawn from their ratings ≥8
- train = everything else, including all implicit interactions and all non-eligible users
- no temporal split possible: `Ratings.csv` has no time column

**Metrics** — `k=10` throughout.

- **HitRate@10** — share of users whose held-out book is in the top-10. Under leave-one-out this **equals Recall@10**, and Precision@10 = HitRate@10 / 10. One measurement, three names.
- **Coverage@10** — distinct items ever recommended, over the full 235,824-work catalogue.
- **Novelty@10** — mean `−log2` smoothed popularity share. Higher = deeper into the tail.
- **AnchorHitRate@10** — same readers, same held-out books, but the query is **one book** instead of the whole profile. This is the query the demo actually makes.

**Uncertainty** — 95% **Wilson** intervals (not Wald: the proportions are small) and paired **McNemar** over per-user hit vectors. Resolution of this split: **±0.002 to ±0.004**. A change smaller than that has not been shown to be a change.

**Hyperparameters** were tuned on an inner validation split carved out of train (`seed=43`), never on the evaluation holdout.

---

## 7 · Serving settings (the demo)

| Setting | Value | Why |
|---|---|---|
| anchor support floor | 50 readers | below that the neighbourhoods are thin; 35.4% of anchors qualify |
| candidate support floor | 20 | a recommended item needs evidence behind it |
| ALS `similar_min_support` | 20 | without it ALS scores **below the popularity baseline** on the item query |
| score truncation `τ` | 0.0 (off) | the rule works but ends lists at 4 items, which reads as a broken app |
| item level | work, not ISBN | one *Crime and Punishment*, not 21 |

---

## 8 · Three sentences that pre-empt the obvious questions

1. **"Why only HitRate?"** — under leave-one-out, Recall@10 and Precision@10 are the same measurement. Reporting all three would be one number dressed up as corroboration.
2. **"Why isn't a random split leaking?"** — it would be, temporally, if the data had time. It does not. With production logs we would split temporally and say so.
3. **"Which model wins?"** — none. Accuracy and coverage move in opposite directions, so the answer is a product decision. The measured recommendation is item-item as the scoring core with a content layer for the catalogue it structurally cannot see.
