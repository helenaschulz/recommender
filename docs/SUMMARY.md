# The argument in two pages

The full evidence is in [`RESULTS.md`](RESULTS.md) — 91 numbered lines, every one recording
how it was measured. This page is the argument those lines support, with the ID to check each
claim against.

## The product question

Paste a book, get ten similar books, with a reason for each. No login, no reading history, no
user identity at query time. That shape is the whole design constraint: the system is asked
*given this one item, what is like it*, which is **not** the question a recommender benchmark
usually scores.

## What the data forces

Book-Crossing is 1,149,780 ratings over 271,360 books and 278,858 users, and three properties
decide the modelling before any model is chosen:

- **62.3% of the interactions carry no grade** (L4) — a rating of 0 marks that someone touched
  the book, not that they liked it. The matrix is implicit far more than it is explicit.
- **It is extremely sparse and extremely long-tailed**: density 0.0032% (L6), 57.9% of books
  rated exactly once (L7). After the standard min-5 filter, collaborative filtering can reach
  **5.3% of the catalogue** (L12) — which is the quantified argument for a content layer, not
  a cold-start footnote.
- **The same book appears under many ISBNs.** 59,928 ISBNs collapse into shared works (L40),
  and the catalogue's own title strings are the only key available — there is no work id.

There are also **no timestamps** (L18), so a chronological split is impossible and the
evaluation has to be leave-one-out per user.

## How it was evaluated

One split, pinned once and never re-drawn for a result: per-user leave-one-out, seed 42, a
held-out book counting as relevant at a rating ≥8, over **13,581 eligible users** (L19) — or
13,580 once editions are merged into works, because one user's graded editions collapse into a
single book (L44). Three metrics, because accuracy alone would pick a bestseller list:
**HitRate@10**,
**Coverage@10** (share of the catalogue a model ever recommends) and **Novelty@10**.

The ceiling matters as much as the score. No collaborative model can exceed **0.8666** on
this split, because that is the share of held-out books reachable through co-occurrence at
all (L50).

## The comparison

Six models, one split, item = *work* rather than ISBN:

| Model | HitRate@10 | Coverage@10 | Novelty@10 | Ledger |
|---|---:|---:|---:|---|
| popularity (baseline) | 0.0155 | 0.027% | 10.54 | L52 |
| **item-item CF** | **0.0644** | 8.190% | 14.17 | L53 |
| ALS / weighted MF | 0.0545 | 0.897% | 12.29 | L55 |
| item-item, explicit only | 0.0486 | 10.036% | 15.97 | L54 |
| content TF-IDF | 0.0405 | 16.806% | 17.07 | L56 |
| content embeddings | 0.0141 | 26.143% | 18.34 | L57 |

Accuracy and coverage run in opposite directions down that table, which is the trade the
product has to choose a point on rather than a ranking to read off. Every pairwise comparison
carries a 95% Wilson interval and a paired McNemar test; the intervals run **±0.002 to
±0.004**, and one comparison fails to separate — embeddings against the popularity baseline,
p = 0.342, published as a tie (L74). The ordering survives **five independent split draws**,
though three near-ties change their verdict between draws (L86).

## The four findings that actually matter

**1 · The largest single accuracy gain was data preparation, not modelling.** Merging editions
into works before training lifts item-item from **0.0546 to 0.0644, +18%** (L44), and raises
the structural ceiling from 84.81% to 86.66%. No model change in this project comes close. The
text-based models gain far more from it than the collaborative ones, and the reason is measured
rather than asserted (L58).

**2 · The demo asks a different question from the table, so the table cannot settle it.**
HitRate@10 scores how well a model ranks a held-out book in a *user's* history. The app asks
what is similar to *one book*. Measured on its own metric over the same 13,580 anchors, ALS
(0.0308) and item-item (0.0297) are **not distinguishable** (L80), and the fusion rules that
win the aggregate — RRF 0.0371, score fusion 0.0355 (L85) — buy every point of their win
**below the support floor the app refuses to serve**.

**3 · The support floor is worth more than the model choice.** ALS scores 0.0308 with a floor
of 20 interactions and **0.0130 without it** — below the popularity baseline — a difference of
+0.0177 at p = 2.5e-41, **larger than any difference between two models in this project**
(L82). In the band the app actually serves, the engines go 41/42 at p = 1.000 — statistically
indistinguishable (L85) — and the band a lower floor would open is too underpowered to settle
the question either way (L88). The floor's price is reach: raising the anchor floor to 50
leaves 2,508 askable works, 26.6% of all interactions (L65).

**4 · The hybrid is real and modest.** The rule this project recommended for four milestones
before measuring it is worth **nine users out of 13,580** — HitRate 0.0644 → 0.0650, 9 wins,
0 losses, p = 0.0039 (L76). The two higher-scoring fusion rules were rejected on a hand read:
they fill roughly a third of a reader's list with another edition of a book they already own
(L79).

## What ships

Two engines behind a picker, **A** (ALS item factors) as default and **B** (item-item with a
shrunk cosine) beside it, both at the same anchor floor of 50 so the switch moves exactly one
variable: the model. Reasons come from structured evidence only — co-reader counts, shared
author, similarity.

The pair is backed by two hand audits rather than by a metric: **240 slots** read by hand in
the band the floor would open (L89) and **440 slots** across 20 anchors in the band the app
serves (L90), where B produced **0 bad slots in 220** against A's 7. A is default because it is
the rehearsed engine, not because it won anything.

Cost: cold start **8.8 s** (A) / 8.5 s (B), warm query 21/22 ms, no network and no fitting at
query time (L91). The demo ships **890 MB** of assets, of which the recommender itself is
**17.5%** — the rest is the free-text search box (L71). For the whole product a precomputed
answer table is **1.5 MB** (L72), which is the number that decides the serving architecture:
at that size a key-value store is sufficient and neither a vector database nor a live model
server is required.

## What this is not

- **One dataset, one draw.** The numbers are seed 42's unless stated; L86 bounds how much that
  matters, it does not remove it.
- **The split's resolution is about ±0.002–0.004** (L74). Differences smaller than that are
  not findings, and are not reported as findings.
- **Translations and alternate titles still defeat the work key** (L47, L59). It is the known
  ceiling on the edition clustering and the first thing an external work identifier would fix.
- **Two of twenty famous titles do not resolve to the book they name** (L90) — a published,
  audited property of the lookup tie rule.

## Where to check

| For | Read |
|---|---|
| every number, and how it was measured | [`RESULTS.md`](RESULTS.md) |
| why each model was chosen or rejected | [`model_selection.md`](model_selection.md) |
| what the dataset is, and prior work on it | [`dataset_findings.md`](dataset_findings.md) |
| the two hand audits, slot by slot | [`floor_band_audit.md`](floor_band_audit.md), [`anchor_set_audit.md`](anchor_set_audit.md) |
| the journey, with charts | [`../notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb), [`../notebooks/02_models.ipynb`](../notebooks/02_models.ipynb) |
