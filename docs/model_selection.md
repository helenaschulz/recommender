# How the recommender was chosen, and what would be built

Written after the model comparison run of 2026-08-04 and re-based onto the work-keyed table
of 2026-08-08 (milestone M12, §11). Every number traces to a line in
[`RESULTS.md`](RESULTS.md); nothing here is from memory.

**One thing to know before reading any number below.** The item is a **work**, not an ISBN:
*Crime and Punishment* is one item, not the 21 editions Book-Crossing ships it as. The
earlier ISBN-keyed table is kept alongside, always labelled, because it is the record of how
the decision was made — and because the distance between the two tables turned out to be a
finding in its own right (§11).

---

## 1 · The question the product actually asks

The brief is: paste a book, get books like it. That single sentence decides more than any
metric does, because it fixes what the model is asked to compute.

- There is **no user identity at query time**. Whoever is typing is anonymous. A model
  that needs to know who you are cannot answer.
- The answer is a **neighbourhood of one item**, not a personalized feed.
- The catalogue is **271,360 ISBNs — 235,824 actual books** once editions are merged
  (L40) — and a bookseller earns on the long tail; the hundredth Harry Potter sale is not
  what a recommender is for.

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
> something worth predicting. **13,580 eligible users.** One held-out book each, drawn at
> random from their ratings ≥8. Everything else is train: all 716,109 implicit
> interactions, and every interaction of the users who are not eligible.

**The item is a *work*, not an ISBN** (milestone M12, L49). Book-Crossing keys everything
by edition, so *Crime and Punishment* is 21 items sharing 141 interactions. The
interactions are re-keyed to works **before** the split is drawn, so holding out a work
removes every edition of it from that user's profile at once. On the ISBN basis the same
rule produced 13,581 eligible users (L19); one loses eligibility when their graded editions
merge. §11 is why the whole table moved, and what it cost to check.

**Metrics, identical for every model.** HitRate@10 — under leave-one-out this *equals*
Recall@10, and Precision@10 = HitRate@10/10, so one number carries all three.
Catalog-Coverage@10 — distinct catalogue items appearing in anybody's top-10, over the
whole catalogue: **235,824 works**, the same denominator in every cell of §4. Novelty@10 —
mean `-log2` smoothed popularity share.

**Leakage discipline, enforced rather than promised.** Item-item PoCs classically fail by
computing similarities on the full matrix before splitting. Here: fitting only ever sees
`split.train`; the holdout is chosen in exactly one place; the notebook *runs* a leakage
check rather than asserting one (0 of 13,580 held-out pairs appear in the train matrix, and
0 of 13,581 on the ISBN basis);
and [`tests/test_split.py`](../tests/test_split.py) pins the invariants, including that
the split does not change when the input rows are shuffled — so "seed 42" describes the
split rather than describing pandas' row order.

**Hyperparameters were chosen on a validation split carved out of train** (seed 43, one
level deeper), never on the evaluation holdout. Sweeping shrinkage on the test split and
reporting the best cell is the offline equivalent of marking your own homework. They were
**re-swept at work level** when the table re-based, and both sweeps re-selected the same
values (L51) — a null result, recorded rather than left to look like an omission.

### The ceiling nobody can beat

Before any model: **13.3% of held-out works appear nowhere in the train matrix** (L50).
That work's only interaction in the entire dataset was the one held out. No co-occurrence
model can rank an item it has never seen.

| Model class | Ceiling on HitRate@10 | on the ISBN basis (L20/L21) |
|---|---:|---:|
| any collaborative model | **86.66%** | 84.81% |
| a content model (needs only a title) | 88.98% | 89.31% |
| the union of both | **95.34%** | 95.37% |

That 8.7-point gap between collaborative-only and the union is the coverage argument for
a hybrid, expressed as a bound on achievable accuracy rather than as a slogan. Note the
right-hand column: merging editions moves these ceilings by at most 1.9 points, so **the
hybrid argument is invariant to the re-base** — which is worth knowing before leaning on
it, and is not something one could assume without measuring it twice.

## 4 · The ladder, and what each rung cost

One command, one split, one run
(`python scripts/run_model.py --all --gallery --work-level`, ~8 min).

| Model | HitRate@10 | Coverage@10 | Novelty@10 | Ledger |
|---|---:|---:|---:|---|
| popularity (baseline) | 0.0155 | 0.027% | 10.54 | L52 |
| **item-item CF** | **0.0644** | 8.190% | 14.17 | L53 |
| ALS / weighted MF | 0.0545 | 0.897% | 12.29 | L55 |
| item-item, explicit-only | 0.0486 | 10.036% | 15.97 | L54 |
| content TF-IDF | 0.0405 | 16.806% | 17.07 | L56 |
| content embeddings | 0.0141 | **26.143%** | **18.34** | L57 |

**The table has a shape, and the shape is the finding.** The ranking by accuracy is almost
exactly the reverse of the ranking by reach, with ALS the only exception. There is no
single best model, so "which model" is the wrong question — "which model for which job" is
the right one.

**The baseline is narrow, not weak** (L27, L52). It scores 0.0155 overall — hundreds of
times better than random. But broken down by the held-out book's popularity it scores
**essentially 0.0000** for every user whose target has fewer than 50 interactions, which is
two thirds of them. Its entire hit rate comes from users who were going to be handed a
bestseller anyway, and it ever recommends **64 distinct works** across all 13,580 users.
Any aggregate metric hides this, which is a good reason never to report just one.

**Item-item wins on both axes at once**: 4.2× the baseline's accuracy *and* 302× its
coverage. It is not trading reach for hit rate.

### The same ladder on the ISBN basis — the journey record

This was the published table from M4 to M11. It is kept because it is how the choice was
made, and because §11 is about the distance between the two. **No cell here is comparable
with a cell above**: different items, different coverage denominator, one more eligible
user.

| Model | HitRate@10 | Coverage@10 | Novelty@10 | Ledger |
|---|---:|---:|---:|---|
| popularity (baseline) | 0.0145 | 0.019% | 10.81 | L22 |
| **item-item CF** | **0.0546** | 9.064% | 14.91 | L24 |
| ALS / weighted MF | 0.0451 | 0.835% | 12.63 | L33 |
| item-item, explicit-only | 0.0379 | 10.739% | 16.59 | L26 |
| content TF-IDF | 0.0228 | 16.616% | 17.63 | L30 |
| content embeddings | 0.0109 | **23.911%** | **18.42** | L35 |

## 5 · The decisions that can be defended, because they were measured

**Using the implicit interactions was worth 24% of the hit rate** (L54). The identical
model fitted on graded ratings alone scores 0.0486 against 0.0644. An ungraded
interaction is weaker evidence than a 10, but it is not noise. The honest
counter-current, which belongs in the same breath: explicit-only *wins* on coverage
(10.0% vs 8.2%) and novelty, because a sparser matrix spreads its recommendations more
thinly. On the ISBN basis the same ablation cost 31% (L26) — same direction, larger
penalty, because fragmentation was hurting the thin explicit-only matrix hardest.

**Shrinkage is what makes item-item work on data this sparse** (L51). With 58% of books
rated exactly once, two books sharing their single reader score a perfect cosine of 1.0 —
one coincidence outranking four hundred readers. Adding the shrinkage term lifts accuracy
by half again (0.0379 → 0.0573 on validation) and costs more than half the catalogue reach
(17.0% → 6.8%). That trade is the accuracy/coverage tension in one line, and it is the
substantive answer to "how do you handle sparsity". The sweep was re-run at work level and
re-selected the same λ=10 and 50 neighbours, so the parameter is not an artefact of the
item key.

**No minimum-support threshold** (L23), the other common defence, because its cost was
measured: raising it to 5 drops the reachable share of held-out books from 84.8% to
64.5%. Twenty points of achievable accuracy to solve a problem shrinkage already handles
continuously. (Measured on the ISBN basis; L43 is the same argument from the other side —
per-ISBN filtering deletes 23,429 editions of works that clear the threshold.)

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

The Harry Potter column is the whole story in one table. It is given twice, because the
re-base changed it and the change is the point.

| Model | On the ISBN basis (M4–M10) | On the work basis (M12, L59) |
|---|---|---|
| popularity | bestsellers — no notion of similarity; the control | unchanged; still the control |
| item-item | two unrelated obscure books first, *then* Chamber of Secrets (L29) | **Chamber of Secrets, Azkaban, Goblet of Fire, Order of the Phoenix — in order** |
| content TF-IDF | five ISBNs of *Sorcerer's Stone* itself (L39) | the pop-up book, the illustrator re-credit, *Philosopher's Stone*, the Welsh edition |
| content embeddings | five editions of *Sorcerer's Stone* itself (L39) | *Pietra Filosfale*, *à l'école des sorciers*, *Philosopher's Stone* |
| **ALS** | ***Fellowship of the Ring*, then Harry Potter 3, 2 and 4** (L34) | Chamber of Secrets, Azkaban, Goblet of Fire, Order of the Phoenix |

**Two things changed and one did not.** Item-item's neighbourhood went from embarrassing to
correct — its evidence had been split over 120 Harry Potter rows, and merging them was
enough (L53). The content models stopped returning the anchor's own ISBNs and started
returning the anchor's own *titles in other languages*, which is L47's wall, still
standing. What did not change: ALS remains the model whose neighbourhoods you would show a
reader, and it is still beaten on every metric in §4.

**That divergence is the reason both numbers and galleries are reported.** A HitRate cannot
see whether a neighbourhood is sensible, and a sensible-looking neighbourhood cannot see
whether it is predictive.

Two failures worth understanding rather than hiding:

**Item-item was under-damped for the item-to-item surface — and the fix turned out to be
data, not tuning** (L29, L53). λ=10 was chosen to maximize HitRate, and HitRate is
dominated by *popular* held-out books; for a medium-support anchor, two books sharing 4
readers out of 6 scored 0.116 and outranked *Chamber of Secrets* at 0.097. The proposed
remedies were a co-occurrence floor or a separate λ for the similarity endpoint. Neither
was needed: on the work basis the same model with the same λ returns *Chamber of Secrets*
at 0.477. The anchor was never under-damped so much as under-evidenced.

**Both content models return the same book again** (L31, L39, L47). At ISBN level 39.1% of
TF-IDF's recommendation slots were another *edition* of a book the user already had,
affecting 81.5% of users. Textually a reprint and the original are identical, so no text
model can distinguish "same work" from "similar work". The re-base removes that failure by
construction — those ISBNs are now one item — but **it does not remove the failure
underneath**: *Philosopher's Stone*, *Harry Potter à l'école des sorciers* and *Harry Potter
E la Pietra Filosfale* are one book with three names, and no string key finds that. §11 has
the count on the work basis, and it is the measured argument for the enrichment layer.

## 7 · What would be built, and why

**Item-item CF as the scoring core.** Best accuracy by a clear margin, answers the
product's question natively, trains in 22 seconds, and is explainable in a sentence a
customer understands: *readers of this book also read that one*. On a short build, an
approach a team can debug beats one it can only tune.

**A content layer beside it, not behind it.** Not as a cold-start footnote — collaborative
filtering is structurally blind to 13.3% of held-out works and most of the catalogue after
standard filtering, and their union raises the achievable ceiling from 86.7% to 95.3%
(L50). The two classes are also **nearly disjoint in what they reach**: of the 52,151 works
item-item and TF-IDF touch between them, only 6,794 — 13% — are reached by both (L60). §11
sharpens this further: the ISBN-keyed table had been *under-rating* the content layer, so
the gap between the two model classes is smaller than the first run suggested.

**ALS kept in the plan for what the metrics do not show.** Free personalization from the
same fit, the best item-to-item neighbourhoods of any model here, and the only model that
ports to Spark without a rewrite — which makes productionization a port rather than a
second project.

**Edition clustering in data prep, before any of it ships.** It is the single largest
accuracy gain in the project — **+18% on item-item's hit rate for a data-prep change**
(L44, L53) — and it is the fix that turned item-item's *Harry Potter* neighbourhood from
two obscure books sharing four readers into the four sequels in order, without touching a
hyperparameter. See §10 and §11.

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

1. ~~**Edition clustering: serving-layer dedup, or a data-prep fix?**~~ ~~The remaining
   decision is whether the whole comparison table moves to work level.~~ **Both answered.**
   They do different jobs (§10), and the table did move (§11, M12). Serving dedup stays in
   the serving layer for the app, because the app's engine is fitted on the full
   interaction matrix and still has to collapse editions on the way to the screen.
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

## 10 · Edition clustering, measured (M11)

§6 ended on a failure that was named but not fixed: a third of the content model's output
is another edition of a book the reader already has, and no text model can tell a reprint
from a similar book because the two are textually identical. It was left unpatched on
purpose — fixing it moves every number in the table, so it needed measuring first.

**The catalogue is 13% smaller than it looks.** 271,360 ISBNs are **235,824 works**;
24,392 works carry more than one ISBN, covering 59,928 ISBNs — 22% of the catalogue
(L40). The clustering key is the normalized title with its trailing parenthetical
stripped, plus the author's surname; the parenthetical is not thrown away but parsed into
a `series` field (74,233 books have one — a series name, but also often a volume number,
a format or an imprint, which is why stripping it needed its own check: L48, 0.023% of
merged interactions at risk). The earlier estimate (L15) counted 40,675 ISBNs;
this key finds **47% more**, because an exact author string cannot see that "Fyodor
Dostoevsky", "Fedor Dostoevsky" and "Fyodor M. Dostoevsky" are one person. L15 was a
lower bound, and is now labelled as one.

**Validated by hand, not asserted.** 30 seeded-random multi-ISBN clusters were inspected
one by one in [`edition_clusters_sample.md`](edition_clusters_sample.md): **0 wrong
merges**. One extension to the key — merging surnames that differ by a single character
under an identical title, which is what finally joins *Dostoevsky* to *Dostoyevsky* —
touches only 223 clusters, so a random sample cannot audit it; 20 of those were drawn
separately and **1 was wrong** (Anne Hampson and Georgia Hampton both wrote a *Desire*).
Two ISBNs, three interactions. Raising the length floor would remove that error and also
un-merge Rendell/Rendall, Elliott/Elliot and Searls/Searles, so the error is reported
rather than tuned away (L42).

**The finding that changes the build order.** Clustering *before* training — the same
item-item model on a work-keyed matrix, same split mechanics, same metrics — lifts
HitRate@10 from **0.0546 to 0.0644, +18%** (L44). That is the largest single accuracy gain
in this project, and it comes from data preparation, not from a model. Three things had to
be ruled out before believing it: the structural ceiling moves only 84.81% → 86.66%, so it
is not an easier target; the held-out work cannot leak in through a second edition,
because `to_work_level` collapses each (user, work) pair before the split; and only 0.38%
of ISBN-level holdouts were a second edition of something the user already had, so the old
number was not being flattered either. ~~It is also a *lower bound* — λ and the
neighbourhood size are still the ones tuned on the ISBN-level split.~~ **That caveat is
retired: the sweep was re-run at work level in M12 and re-selected the same λ=10 and 50
neighbours (L51), so the +18% is not a lower bound for that reason.**

**Why the standard recipe makes this worse.** The usual min-5 filter is applied per ISBN,
so it deletes **23,429 editions carrying 49,649 interactions that belong to works which
clear the threshold** (L43). *Crime and Punishment* loses 19% of its evidence that way and
is then treated as a book with 40 readers.

### Deduplication at serving time

The models still score ISBNs, and the app still has to show books. Collapsing the output
to one ISBN per work — dropping works the reader already has — is a presentation
decision, so it lives in the serving layer, not in the models. That keeps the comparison
table meaning what it meant and makes the fix measurable by switching it off.

**It is free, and for the content model it is better than free** (L45):

| Model | duplicate slots | users affected | HitRate@10 |
|---|---:|---:|---:|
| content TF-IDF | 39.1% → **0.0%** | 81.5% → 0% | 0.0228 → **0.0277** |
| content embeddings | 11.3% → 0.0% | 38.5% → 0% | 0.0109 → 0.0108 |
| ALS | 1.9% → 0.0% | 11.8% → 0% | 0.0451 → 0.0454 |
| item-item CF | 1.2% → 0.0% | 7.9% → 0% | 0.0546 → 0.0546 |

Four in five TF-IDF users were being handed a book they already owned. Removing those
slots does not cost accuracy — it *buys* 21% of it, because a wasted slot gets refilled
with a real candidate. The collaborative models barely move, and the reason is worth
saying out loud: two ISBNs of one book are read by *different* people, so they never
co-occur, and collaborative similarity separates editions for free. Only the text models
ever had this problem.

**The number that is too clean to trust, and what it actually hides.** Those 0.0%s are
measured with the same key that did the deduplication, so they are zero by construction —
not evidence. The independent check is the gallery, and it is less flattering (L47). After
dedup, item-item and ALS are genuinely clean: **0 of 30** gallery slots are the anchor
again. TF-IDF still returns **7 of 30** and the embedding model **9 of 30** — and every
survivor is the same book under a different *title*: *Desde Mi Cielo* and *In meinem
Himmel* for *The Lovely Bones*, *Philosopher's Stone* and the French, Spanish, Italian and
German editions for *Harry Potter*. Asked for books like *Harry Potter and the Sorcerer's
Stone*, the embedding model answers with seven Harry Potter and the Sorcerer's Stones.

No amount of string normalization finds those. It is the same wall as the failed
cross-lingual lookup in §5 (L38), hit from the other side, and it has the same fix: more
text per book — LLM-generated descriptions, themes, genre tags — or an external work
identifier that already knows these are one book. **Two independent measurements now point
at the enrichment layer**, which is a better reason to build it than the fact that it
involves an LLM.

### What this changes

- **Ship the serving dedup.** It costs nothing, it fixes the most visible defect in the
  demo, and it is switchable so the comparison table stays interpretable.
- **Work-level clustering belongs in data prep, not just at serving.** +18% on the one
  model measured both ways is too large to leave on the table. ~~The open decision is
  whether the whole comparison table re-bases to works.~~ **Decided: it did. §11.**
- **A content model still cannot be the app's similarity engine.** §6 said edition
  clustering was the precondition. It was necessary and it was not sufficient.

## 11 · The re-base to works, and the mechanism behind it (M12)

§10 ended on an open decision: clustering before training was worth +18% on the one model
measured both ways, so did the whole comparison table move to work level? It did. §4 is
now measured on works, and the ISBN-keyed table sits beside it as the journey record.

This gets its own section because **every model's number changed, and not by the same
amount.** Five of the six moved between +7% and +30%. TF-IDF moved **+77%**, far outside the
plausibility band the milestone had set for itself in advance (−20% to +40%). That gate
fired, the branch was held unmerged, and the number was taken apart before anything was
published.

### What the gate caught — and what it did not

Not a bug. Leakage was ruled out mechanically first: zero held-out (user, work) cells appear
in the train matrix, and the path where a held-out edition hides behind a sibling edition in
the same reader's profile is closed by construction — `to_work_level` collapses each
(user, work) pair *before* the split, so every edition of a held-out work leaves train
together. The canonical title each work carries into the content models is chosen from train
only, and a test asserts the holdout could otherwise have changed which edition won, so that
discipline is enforced rather than claimed.

The first explanation offered on the branch was that the re-base "removes a defect that was
suppressing the text models specifically", evidenced by the lift being monotone in each
model's duplicate-slot rate. That is half right, and the decomposition (L58) shows which
half.

### Splitting the lift in two

Re-basing does two separable things, and only one of them is the model getting better.

1. **Evaluation fairness.** At ISBN level, recommending the Penguin edition when the reader's
   held-out book was the Vintage edition scores **zero** — the model named the right book and
   the metric called it wrong. Isolating this needs no re-fit: same model, same top-10 lists,
   only the definition of a hit changes.
2. **Merged signal.** Everything else — co-occurrence counts split across editions become one
   count, one canonical text per work replaces a pile of near-duplicate strings, the item
   universe shrinks, and a slot spent on a second edition of a book the reader already has
   goes to a real candidate.

| Model | ISBN basis | + work credit | work basis | evaluation fairness | merged signal | total | ISBN dup slots |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0145 | 0.0155 | 0.0155 | +6.6% | +0.5% | +7.1% | 0.3% |
| item-item CF | 0.0546 | 0.0588 | 0.0644 | +7.5% | +9.5% | +17.8% | 1.2% |
| ALS | 0.0451 | 0.0503 | 0.0545 | +11.4% | +8.4% | +20.7% | 1.9% |
| item-item, explicit-only | 0.0379 | 0.0408 | 0.0486 | +7.6% | +19.1% | +28.2% | 3.9% |
| content embeddings | 0.0109 | 0.0115 | 0.0141 | +5.4% | +22.4% | +29.1% | 11.3% |
| **content TF-IDF** | **0.0228** | **0.0250** | **0.0405** | **+9.4%** | **+62.3%** | **+77.4%** | **39.1%** |

Ledger L58. A fourth column in the script repeats the fairness measurement with any slot the
reader already owns blanked out — work credit must not be allowed to reward recommending a
book they demonstrably have, because the work-level table cannot do that. The correction is
negligible everywhere (TF-IDF 0.0250 → 0.0245).

### The mechanism, named

**ISBN-level evaluation double-penalized the text models.** They were charged twice for the
same property of the data:

- on the **output** side — a slot spent on another edition of a book the reader already had
  is a wasted recommendation: **39.1% of TF-IDF's slots and 81.5% of its users** (L45);
- and on the **scoring** side — when the edition they *did* recommend was the right book
  under the wrong ISBN, the metric scored it zero.

Collaborative models were barely charged the first way at all, and the reason is worth saying
out loud: two ISBNs of one book are read by *different* people, so they never co-occur, and
collaborative similarity separates editions for free. It is exactly the models whose
similarity is textual that suffer, because textually a reprint and its original are the
**same document**.

**The decomposition says which charge carries the number, and it is not the one you would
guess.** The scoring-side penalty is small and *roughly uniform across the whole table* —
+5.4% to +11.4%, with ALS the largest and the embedding model the smallest, ordered by
nothing in particular. The ISBN key was mildly unfair to everybody. What varies by two orders
of magnitude is the output-side effect, and that is what tracks the duplicate-slot rate:
+0.5% for popularity, +62.3% for TF-IDF. L45 prices that path independently — serving dedup
alone, nothing else changed, was worth +21% on TF-IDF — which is a lower bound on its share.

### What it changes, and what it does not

**The ranking is unchanged.** item-item > ALS > explicit-only > TF-IDF > popularity >
embeddings on accuracy, at both item levels, and the accuracy/coverage tension survives. The
recommendation in §7 does not move.

**The gap between the layers is smaller than the first table suggested.** TF-IDF was
under-rated by the ISBN-keyed metric by a factor, not a rounding error. Anyone reading the
M4–M10 table alone would overstate how far behind the content layer sits, so the hybrid
argument is stronger on the work basis, not weaker.

**The item-to-item surface improved where no metric could see it.** §6 has the gallery: the
same item-item model with the same λ went from two obscure books sharing four readers to the
four Harry Potter sequels in order. That is the most demoable result of the milestone and it
does not appear in any cell of §4.

**The gate was right and the band was wrong, at the same time.** A plausibility band is worth
having precisely because it forces this examination; it is not worth obeying once the
examination finds a reason. The band assumed a re-parameterisation. For a text model the
re-base is also a defect fix, and no band calibrated on the first assumption could have
passed it.

### What is still open

- **Title equality is still the ceiling** (L47, L59). On the work basis the content models
  still answer *Sorcerer's Stone* with *Philosopher's Stone*, the Italian and French
  editions, and the Welsh one — 7 of 30 gallery slots for TF-IDF, 6 of 30 for embeddings.
  Only more text per book or an external work identifier fixes that.
- **L27 and L28 were not recomputed on the work basis.** They are per-stratum findings about
  the ISBN-level rows, quoted as such.
- **Still one split, still offline.** §8 applies unchanged, and re-basing does not make an
  offline proxy any less of a proxy.

## 12 · The demo, and what it deliberately contradicts (M13)

`streamlit run app/main.py`: paste a book, get ten similar books, each with one sentence of
reason drawn from countable evidence — co-reader count, shared author, shared series,
similarity value. No language model anywhere in the hot path. It starts in **9.4 s** and
answers in **21 ms** (L61), with no network and no fitting at query time.

**It runs on ALS, which loses §4.** That is the point rather than an oversight. §6 measured
the divergence: HitRate@10 scores how well a model ranks a held-out book in a *user's*
history, and the app asks a different question — given this one book, what is like it. ALS
is third of six on the first and best in the project on the second (L34, L55). Building the
demo on the model that wins the table would have meant building it on the model with worse
neighbourhoods, so the sidebar shows the table where ALS loses, next to the results it
produces. A panel can then ask the question, and there is an answer.

**It runs on works, like §4**, so the *Harry Potter* anchor returns *Chamber of Secrets*,
*Prisoner of Azkaban*, *Goblet of Fire* and *Order of the Phoenix* rather than a shelf of
editions of itself. That is §11's finding made visible.

**The input path needed two serving rules, and they are audited** (L62). Free-text lookup
resolved only 3 of 9 queries at rank 1 on raw cosine — *Hoopla — Harry Stein* beat Harry
Potter, exactly as L38 recorded. A support floor (never offer an anchor the engine would
refuse to answer for) takes it to 7 of 9; a 0.06-cosine tie margin that prefers the
better-read work among near-equal text matches takes it to 9 of 9. Neither is a model
change and neither touches a published number — but the second is a UI judgement chosen on
nine queries, and it is recorded as such rather than presented as a result.

**What the demo cannot hide.** `"herr der ringe"` and `"hobit tolkien"` still find nothing,
under every rule. Title+author is three to five words, and no amount of serving logic turns
that into enough signal for a multilingual encoder to bridge. It is the same wall as §10's
gallery and L59's count, now hit from a third direction — and the third independent
argument for the metadata-enrichment layer.


