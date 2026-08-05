# Book-Crossing: what is already known, and which approaches make sense

As of 2026-08-03. Every number below was recomputed on the local CSVs in `data/`, not
copied from someone else's notebook. Sources are listed at the end.

## 1. Provenance and character

The Kaggle dataset [`arashnic/book-recommendation-dataset`](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
is the classic **Book-Crossing dataset**: a crawl of bookcrossing.com from August and
September 2004, published by Ziegler et al. (WWW 2005, ["Improving Recommendation Lists
Through Topic Diversification"](https://doi.org/10.1145/1060745.1060754)). 278,858 users,
271,360 books, 1,149,780 ratings. It has been a standard benchmark for twenty years, so
there is a lot of prior knowledge to build on.

**The structural property almost every public notebook ignores: there are no
timestamps.** `Ratings.csv` holds only `User-ID`, `ISBN`, `Book-Rating`. A time-based
train/test split is therefore impossible. The honest alternatives are a per-user random
holdout or per-user leave-N-out, and saying openly that a production system with real
timestamps would be split temporally instead. That limitation is worth stating rather
than hiding: the split follows what the data can support, not what the textbook prefers.

## 2. Data quality, recomputed on these files

| Finding | Number |
|---|---|
| Implicit zeros in `Ratings.csv` | 716,109 of 1,149,780 (62.3%) |
| Explicit ratings | 433,671 (37.7%), mean 7.6, strongly left-skewed (mode 8) |
| Matrix density (all interactions) | 0.0032% |
| Books with exactly 1 rating | 57.9% (explicit-only: 69.7%) |
| Books with fewer than 5 ratings | 87.1% |
| Users with exactly 1 rating | 56.2% |
| Top 1% of books | 25.1% of all interactions |
| Ratings whose ISBN is **not** in `Books.csv` | 118,644 (10.3%) |
| Users in `Users.csv` without a single rating | 173,575 (62%) |
| `Age` missing | 39.7%; outliers from 0 to 244 (0.74% implausible) |
| `Year-Of-Publication` | 4,618 zeros, 23 later than 2006, 3 non-numeric (broken rows) |
| Edition duplicates (same title+author, several ISBNs) | 17,554 works, 40,675 ISBNs |
| After the typical filter (explicit, user ≥5, book ≥5) | 152,280 ratings, 13,305 users, 14,513 books, density 0.079% |

Three consequences follow:

1. **The coverage argument for a hybrid.** After the usual 5/5 filter, ~14.5k of 271k
   books remain. Pure collaborative filtering can therefore only ever recommend ~5% of
   the catalogue. Content-based modelling is not a nice-to-have fallback, it is the only
   coverage for the other 95%. Quantified, that is a strong argument.
2. **Join loss.** 10.3% of ratings lose their metadata when joined to `Books.csv` (ISBN
   variants, typos). Still usable for collaborative filtering, not usable for content
   features or display. This belongs in the data-preparation layer (ISBN normalization,
   possibly ISBN-10/13 canonicalization).
3. **`Users.csv` carries little.** 62% of users have no rating, `Age` is 40% missing and
   unreliable, `Location` is free text. Leaving demographics out deliberately, and saying
   why, is stronger than half-heartedly wiring them in.

## 3. What the community and the research have done with this dataset

**Kaggle and GitHub notebooks** (many dozens) mostly converge on one pattern: filter
aggressively (often users with >200 ratings and books with >50, which boils the dataset
down to a small dense core), build a pivot matrix, then run k-NN with cosine similarity
(`sklearn` `NearestNeighbors` on a CSR matrix) or SVD. Results are judged by eye ("looks
plausible"), precision/recall/coverage are rarely measured cleanly, and the implicit
zeros are usually discarded without comment. That is the gap: **clean evaluation plus a
deliberate decision about the 62% implicit interactions is what separates this project
from practically every public notebook.**

**Research** uses Book-Crossing as a standard benchmark for sparsity and popularity bias:

- The popularity-bias study by Naghiaei, Rahmani and Deldjoo (2022), ["The Unfairness of
  Popularity Bias in Book Recommendation"](https://arxiv.org/abs/2202.13446), compares 11
  algorithms on Book-Crossing. **WMF (implicit ALS) and VAECF** give the best balance of
  accuracy and fairness; **MostPop, BPR and NeuMF amplify popularity bias** the most. More
  accurate models tend to be less fair towards niche readers. This is the direct
  justification for reporting coverage and novelty alongside Precision@K — and a
  bookseller earns on the long tail (cross-sell), not on the hundredth Harry Potter sale.
- Consensus across many papers: on extremely sparse explicit data, simple methods
  (item-KNN, well-regularized matrix factorization, linear models such as SLIM and EASE)
  regularly beat neural approaches. Deep learning is not the lever here, and that is worth
  saying plainly.

## 4. Approaches that fit this use case (book in, similar books out)

The model ladder from the earlier notes holds. Refinements:

1. **Decide on the signal first, not on the model.** The most important design decision is
   implicit vs. explicit: 62% of the data are interactions without a grade. Recommendation:
   use both. Binarize (interaction yes/no) for candidate similarity, and use explicit
   ratings as a confidence weight (in the direction of implicit ALS / WMF). Explicit-only
   throws away two thirds of the signal.
2. **Item-item CF as the core stays right** (the use-case argument is unchanged): cosine on
   the binarized user-item matrix, a minimum interaction threshold, and shrinkage against
   coincidental co-occurrence for rare books.
3. **ALS is not an alternative but the second stage of the same story.** ALS item factors
   *are* item embeddings, and similarity is a dot product. So ALS serves the same use case
   (similar books) AND gives personalization for free later, plus the Spark/Databricks
   bridge for the productionization story. That unifies the model ladder.
4. **Mention EASE as a cheap, strong candidate** — a closed-form linear item-item model,
   one matrix inversion, close to state of the art on sparse data (Steck 2019,
   ["Embarrassingly Shallow Autoencoders for Sparse Data"](https://arxiv.org/abs/1905.03375)).
   Worth naming even if it only appears as "we would test this next".
5. **Content-based over title/author** (TF-IDF or sentence embeddings) **as a coverage
   layer**, justified by the 5% argument from section 2 rather than as a cold-start
   platitude.
6. **Evaluation:** per-user random or leave-N-out split (there are no timestamps),
   Precision@K/Recall@K on explicit-positive holdouts (rating ≥8 as "relevant" is common
   and fits the skewed distribution), plus catalogue coverage and a popularity/novelty
   measure. Popularity baseline as the benchmark, everything recorded in
   [`RESULTS.md`](RESULTS.md).
7. **Not doing in this project:** user-based CF (wrong for the use case), NeuMF and deep
   models (the evidence does not support them here), demographic features (data quality),
   timestamp-based splits (no timestamps exist).

## 5. Sources

- Kaggle: [`arashnic/book-recommendation-dataset`](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) —
  the dataset page, and its [code tab](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/code)
  for the representative community notebooks described in section 3.
- Ziegler, McNee, Konstan, Lausen (WWW 2005): ["Improving Recommendation Lists Through
  Topic Diversification"](https://doi.org/10.1145/1060745.1060754) — the original
  Book-Crossing paper (provenance, dataset sizes).
- Naghiaei, Rahmani, Deldjoo (2022): ["The Unfairness of Popularity Bias in Book
  Recommendation"](https://arxiv.org/abs/2202.13446) — the popularity-bias benchmark on
  Book-Crossing.
- Hu, Koren, Volinsky (ICDM 2008): ["Collaborative Filtering for Implicit Feedback
  Datasets"](https://doi.org/10.1109/ICDM.2008.22) — the weighted matrix factorization
  (implicit ALS) formulation used here via the
  [`implicit`](https://github.com/benfred/implicit) library.
- Steck (WWW 2019): ["Embarrassingly Shallow Autoencoders for Sparse
  Data"](https://arxiv.org/abs/1905.03375) — EASE.
- Own profiling of the local CSVs, script run on 2026-08-03 (the numbers in section 2).
