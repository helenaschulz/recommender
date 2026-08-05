# How the recommender was chosen, and what would be built

Written after the model comparison run of 2026-08-04. Every number traces to a line in
[`RESULTS.md`](RESULTS.md); nothing here is from memory.

---

## 1 · The question the product actually asks

The brief is: paste a book, get books like it. That single sentence decides more than any
metric does, because it fixes what the model is asked to compute.

- There is **no user identity at query time**. Whoever is typing is anonymous. A model
  that needs to know who you are cannot answer.
- The answer is a **neighbourhood of one item**, not a personalized feed.
- The catalogue is **271,360 books**, and a bookseller earns on the long tail — the
  hundredth Harry Potter sale is not what a recommender is for.

So the natural hypothesis is **item-item collaborative filtering**: the model is literally
a table of "readers of this book also read that one". It was treated as a hypothesis, and
the alternatives were measured against it rather than assumed away.

**A gap not papered over.** The offline harness scores *user histories*: hold out one of a
user's books, ask the model for ten, check whether the held-out book is among them. The
product asks something different — *item to item*. A good HitRate is evidence that a
model's neighbourhoods are informative, not proof that they look sensible to a reader.
That is why every model also produces a **face-validity gallery** (§6), and why the two
disagree in a way that turned out to be one of the more useful findings here.

## 2 · What the data forced

Three properties of Book-Crossing drove every subsequent choice
([`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb), ledger L1–L18).

**62.3% of the rows carry no grade** (L4). A `Book-Rating` of 0 is an interaction, not a
score. Discarding them is the default in almost every public notebook on this dataset,
and it throws away two thirds of the signal. They are kept here, binarized, with the
explicit ratings doing two jobs: relevance labels in evaluation, and confidence weights
in ALS. The decision was then *measured* rather than asserted — see §5.

**The catalogue is extremely long-tailed** (L6–L9). 0.0032% density; 58% of books rated
exactly once; the top 1% of books absorbing 25.1% of all interactions. This is why
catalogue coverage and novelty are reported next to accuracy: on a distribution this
skewed, a model can win on accuracy by recommending bestsellers to everybody, and that
model is worthless to a bookseller.

**There are no timestamps at all** (L18). `Ratings.csv` has three columns: user, ISBN,
rating. A temporal split is not expressible on this data, so evaluation uses per-user
leave-one-out and says so. On a real interaction log with timestamps the split would be
temporal, because a random split there leaks future behaviour into training and flatters
the model. Whether the log carries timestamps and sessions is the first question to ask
of any production dataset: the answer changes evaluation, and it decides whether sequence
models are possible at all.

## 3 · The evaluation split, pinned once

Everything is measured on one split, defined in
[`src/recommender/split.py`](../src/recommender/split.py) and cited by every row of the
ledger (L19).

> Per-user leave-one-out, **seed 42**. A user is eligible with ≥5 explicit ratings **and**
> ≥1 rating ≥8 — the first so a profile survives the holdout, the second so there is
> something worth predicting. **13,581 eligible users.** One held-out book each, drawn at
> random from their ratings ≥8. Everything else is train: all 716,109 implicit
> interactions, and every interaction of the users who are not eligible.

**Metrics, identical for every model.** HitRate@10 — under leave-one-out this *equals*
Recall@10, and Precision@10 = HitRate@10/10, so one number carries all three.
Catalog-Coverage@10 — distinct catalogue books appearing in anybody's top-10, over all
271,360. Novelty@10 — mean `-log2` smoothed popularity share.

**Leakage discipline, enforced rather than promised.** Item-item PoCs classically fail by
computing similarities on the full matrix before splitting. Here: fitting only ever sees
`split.train`; the holdout is chosen in exactly one place; the notebook *runs* a leakage
check rather than asserting one (0 of 13,581 held-out pairs appear in the train matrix);
and [`tests/test_split.py`](../tests/test_split.py) pins the invariants, including that
the split does not change when the input rows are shuffled — so "seed 42" describes the
split rather than describing pandas' row order.

**Hyperparameters were chosen on a validation split carved out of train** (seed 43, one
level deeper), never on the evaluation holdout. Sweeping shrinkage on the test split and
reporting the best cell is the offline equivalent of marking your own homework.

### The ceiling nobody can beat

Before any model: **15.2% of held-out books appear nowhere in the train matrix** (L20).
That book's only interaction in the entire dataset was the one held out. No co-occurrence
model can rank an item it has never seen.

| Model class | Ceiling on HitRate@10 |
|---|---:|
| any collaborative model | **84.81%** |
| a content model (needs only a title) | 89.31% |
| the union of both | **95.37%** |

That 10.6-point gap between collaborative-only and the union is the coverage argument for
a hybrid, expressed as a bound on achievable accuracy rather than as a slogan.

## 4 · The ladder, and what each rung cost

One command, one split, one run (`python scripts/run_model.py --all --gallery`, 8m35s).

| Model | HitRate@10 | Coverage@10 | Novelty@10 | Ledger |
|---|---:|---:|---:|---|
| popularity (baseline) | 0.0145 | 0.019% | 10.81 | L22 |
| **item-item CF** | **0.0546** | 9.064% | 14.91 | L24 |
| ALS / weighted MF | 0.0451 | 0.835% | 12.63 | L33 |
| item-item, explicit-only | 0.0379 | 10.739% | 16.59 | L26 |
| content TF-IDF | 0.0228 | 16.616% | 17.63 | L30 |
| content embeddings | 0.0109 | **23.911%** | **18.42** | L35 |

**The table has a shape, and the shape is the finding.** The ranking by accuracy is almost
exactly the reverse of the ranking by reach, with ALS the only exception. There is no
single best model, so "which model" is the wrong question — "which model for which job" is
the right one.

**The baseline is narrow, not weak** (L27). It scores 0.0145 overall, 491× better than
random. But broken down by the held-out book's popularity: it scores **exactly 0.0000**
for every user whose target has fewer than 50 interactions — 73% of them. Its entire hit
rate comes from users who were going to be handed a bestseller anyway, and it ever
recommends **51 distinct books** across all 13,581 users. Any aggregate metric hides
this, which is a good reason never to report just one.

**Item-item wins on both axes at once**: 3.8× the baseline's accuracy *and* 477× its
coverage. It is not trading reach for hit rate.

## 5 · The decisions that can be defended, because they were measured

**Using the implicit interactions was worth 31% of the hit rate** (L26). The identical
model fitted on graded ratings alone scores 0.0379 against 0.0546. An ungraded
interaction is weaker evidence than a 10, but it is not noise. The honest
counter-current, which belongs in the same breath: explicit-only *wins* on coverage
(10.7% vs 9.1%) and novelty, because a sparser matrix spreads its recommendations more
thinly.

**Shrinkage is what makes item-item work on data this sparse** (L25). With 58% of books
rated exactly once, two books sharing their single reader score a perfect cosine of 1.0 —
one coincidence outranking four hundred readers. Adding the shrinkage term nearly doubles
accuracy (0.0296 → 0.0532 on validation) and costs more than half the catalogue reach
(17.5% → 7.6%). That trade is the accuracy/coverage tension in one line, and it is the
substantive answer to "how do you handle sparsity".

**No minimum-support threshold** (L23), the other common defence, because its cost was
measured: raising it to 5 drops the reachable share of held-out books from 84.8% to
64.5%. Twenty points of achievable accuracy to solve a problem shrinkage already handles
continuously.

**Dense embeddings needed a fix that is invisible unless you look for it** (L36). The
first embedding run scored 0.0042 — worse than the baseline. The cause: averaging a
user's book vectors produces almost the same vector for every user (mean cosine to the
global profile centroid 0.883). Sentence embeddings share a large common direction and
averaging amplifies it. Centering the item vectors drops that to 0.193 and, on
validation, lifts HitRate from 0.0036 to 0.0095 and coverage from 3.7% to 20.4%. The
naive "embed everything and take cosine" recipe underperforms for a reason that is one
line to fix once diagnosed.

## 6 · Where the offline metric and the product disagree

Every model answered the same three books. *The Da Vinci Code* (853 interactions) is the
easy case; *The Lovely Bones* (1,248) checks that one genre cluster is not simply being
reproduced; **Harry Potter and the Sorcerer's Stone** (101) is the diagnostic — medium
support, an obvious right answer, and 120 Harry Potter rows in the catalogue across
editions.

The Harry Potter column is the whole story in one table:

| Model | What it returns |
|---|---|
| popularity | bestsellers — no notion of similarity; the control |
| item-item | two unrelated obscure books first, *then* Chamber of Secrets (L29) |
| content TF-IDF | five ISBNs of *Sorcerer's Stone* itself (L39) |
| content embeddings | five editions of *Sorcerer's Stone* itself (L39) |
| **ALS** | ***Fellowship of the Ring*, then Harry Potter 3, 2 and 4** (L34) |

**The model that scores worst on the metrics has the best neighbourhoods.** ALS is beaten
by item-item on all three numbers, and it is the only model that answers the question a
reader would actually ask. That is not a curiosity — it is direct evidence that HitRate
and the product surface measure different things, which is exactly why both are reported.

Two failures worth understanding rather than hiding:

**Item-item is under-damped for the item-to-item surface** (L29). λ=10 was chosen to
maximize HitRate, and HitRate is dominated by *popular* held-out books. For a
medium-support anchor, two books sharing 4 readers out of 6 score 0.116 and outrank
*Chamber of Secrets* at 0.097. The fix is straightforward — a minimum co-occurrence
floor, or a higher λ on the similarity endpoint than on the ranking one — but it is a
product decision, so it was measured and left open.

**Both content models return the same book again** (L31, L39). 31.6% of TF-IDF's
recommendation slots are another *edition* of a book the user already has, affecting
73.4% of users. Textually a reprint and the original are identical, so no text model can
distinguish "same work" from "similar work". **Edition clustering is a precondition for
shipping a content model as the app's similarity engine**, not a data-hygiene nicety, and
it also means every model's coverage number is inflated by roughly a fifth (L32).

## 7 · What would be built, and why

**Item-item CF as the scoring core.** Best accuracy by a clear margin, answers the
product's question natively, trains in 21 seconds, and is explainable in a sentence a
customer understands: *readers of this book also read that one*. On a short build, an
approach a team can debug beats one it can only tune.

**A content layer beside it, not behind it.** Not as a cold-start footnote — collaborative
filtering is structurally blind to 15.2% of held-out books and 95% of the catalogue after
standard filtering. The two model classes overlap on only 7,451 of the ~75,000 books they
reach between them, and their union raises the achievable ceiling from 84.8% to 95.4%.

**ALS kept in the plan for what the metrics do not show.** Free personalization from the
same fit, the best item-to-item neighbourhoods of any model here, and the only model that
ports to Spark without a rewrite — which makes productionization a port rather than a
second project.

**Edition clustering before any of it ships.** It is the one fix that improves every
model's product surface at once.

### Deliberately not built, and why

- **User-based CF** — answers a different question. There is no user identity at query time.
- **Deep scoring models (NeuMF, LightGCN)** — the evidence on this dataset says
  neighbourhood methods and well-regularized linear models win on extreme sparsity, and
  the [popularity-bias study run *on Book-Crossing*](https://arxiv.org/abs/2202.13446)
  flags MF-family and neural models as bias amplifiers. The ALS row here is a small
  confirmation. **Mult-VAE/VAECF** is the one neural candidate with evidence on this
  dataset; it belongs on the list as a stretch goal after a standing item-item benchmark,
  not as a first build.
- **Sequence models (SASRec, BERT4Rec)** — impossible. No timestamps.
- **Demographic features** — 62% of users in `Users.csv` never rated anything, Age is 40%
  missing and self-reported up to 244 years. Leaving them out by decision is a stronger
  position than a half-working feature.
- **An LLM as the recommender itself** — latency and cost per request, not offline
  evaluable, popularity bias from pretraining, and no knowledge of this catalogue. The
  LLM earns its place in the *layers* (metadata enrichment, explanation), not in the core.

## 8 · What these numbers are not

One dataset, one split, one draw — seeded and reproducible, but no confidence intervals,
so differences of a few tenths of a percent are not real. And every metric is a proxy:
"was the held-out book in the top ten" stands in for "would a reader click, buy, or
enjoy this". A recommendation the reader has never heard of scores zero whether it was a
brilliant discovery or a mistake — which is precisely the outcome a long-tail recommender
exists to produce.

**A live A/B test is the only thing that settles the real question**, and that stays true
however favourable the offline table looks. What the offline work buys is the right to
choose which two or three candidates go into that test, and the confidence that they were
not chosen by accident.

## 9 · Open questions

1. **Edition clustering: serving-layer dedup, or a data-prep fix?** It is the single
   highest-value change (L31). Done properly in data prep it changes the comparison basis
   for every model, so every number would need re-measuring. The pragmatic path is dedup
   in the serving layer first, with the ledger numbers left as they are and the
   data-prep-level clustering named as the proper fix.
2. **Which model should drive the app?** Item-item has the best numbers; **ALS has by far
   the best neighbourhoods** (L34), and the app is an item-to-item surface. Showing ALS
   *and* the table where it loses is a better account than either number alone — it just
   requires explaining why the metric and the demo disagree.
3. **Re-tune item-item for the similarity endpoint?** (L29) A higher λ or a minimum
   co-occurrence floor would fix the Harry Potter neighbourhood. Not done, because
   choosing a parameter to make a demo look better is exactly the move this project argues
   against — but there is a legitimate version: tune the *similarity* endpoint on its own
   validation objective rather than on HitRate.
4. **Confidence intervals.** Currently none. A handful of seeds per model would give error
   bars for perhaps twenty minutes of compute, and would pre-empt "is that difference
   real?".
5. **Cross-lingual lookup is weak** (L38) — `"herr der ringe"` finds nothing. Title+author
   is too thin for a multilingual encoder to bridge. This is the concrete, now-measured
   argument for an LLM metadata-enrichment layer.
