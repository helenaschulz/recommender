# How the recommender was chosen, and what would be built

Written after the model comparison run of 2026-08-04, re-based onto the work-keyed table of
2026-08-08 (milestone M12, §11), and carried through to the two-engine demo of 2026-08-10
(M23.10, §13). Every number traces to a line in [`RESULTS.md`](RESULTS.md); nothing here is
from memory. For the argument without the workings, see [`SUMMARY.md`](SUMMARY.md).

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
and it throws away 62.3% of the rows. They are kept here, binarized, with the
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

**The table has a shape, and the shape is the finding.** Among the four real models the
ranking by accuracy is exactly the reverse of the ranking by reach: item-item > ALS >
explicit-only > TF-IDF on HitRate, TF-IDF > explicit-only > item-item > ALS on Coverage.
**The baseline sits outside that trade-off rather than at one end of it** — last on accuracy
among the real models and last on coverage too, so it buys nothing in either direction and
only marks the zero point. There is no single best model, so "which model" is the wrong
question — "which model for which job" is the right one.

**One row is a tie, not a loss.** Content embeddings at 0.0141 against the baseline's 0.0155
is **19 users out of 13,580**, z ≈ 1.0 — inside sampling noise, so the two are not
distinguishable and the row must be read as "matches the baseline on accuracy while reaching
963× more of the catalogue", never as "worse than the baseline". Every other gap in the table
clears three standard errors. The derivation is under the primary table in `RESULTS.md`; §8
is where the general caveat lives.

**The baseline is narrow, not weak** (L27 on the ISBN basis, L52 on the work basis). It
scores 0.0155 overall — hundreds of times better than random. But broken down by the
held-out book's popularity it scores **essentially 0.0000** for every user whose target has
fewer than 50 interactions, which is **73% of them at ISBN level and 65% at work level**.
Note the two bases in that sentence: the stratification is L27's at ISBN level and
**L73's at work level** (measured 09.08.2026, M18.2), and the share is given both ways
rather than rounded into one. The work-level strata say the same thing more sharply — the
baseline scores 0.0000 in all three strata below 50 interactions and 0.0443 in the fourth. Its entire hit rate comes from
users who were going to be handed a bestseller anyway, and it ever recommends **64 distinct
works** across all 13,580 users.
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

**Dense embeddings needed a fix that is invisible unless you look for it** (L36). Before
the fix the model scored **0.0036** on the validation split against **0.0095** after it —
worse than the popularity baseline either way. The cause: averaging a
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
| content TF-IDF | **eight** ISBNs of *Sorcerer's Stone* itself, its whole top 8 (L31) | the pop-up book, the illustrator re-credit, *Philosopher's Stone*, the Welsh edition |
| content embeddings | five editions of *Sorcerer's Stone* itself (L39) | *Pietra Filosfale*, *à l'école des sorciers*, *Philosopher's Stone* |
| **ALS** | ***Fellowship of the Ring*, then Harry Potter 3, 2 and 4** (L34) | Chamber of Secrets, Azkaban, Goblet of Fire, Order of the Phoenix |

**Two things changed and one did not.** Item-item's neighbourhood went from embarrassing to
correct — its evidence had been split over 120 Harry Potter rows, and merging them was
enough (L53). The content models stopped returning the anchor's own ISBNs and started
returning the anchor's own *titles in other languages*, which is L47's wall, still
standing. What did not change: ALS is still beaten on every metric in §4.

**What this table used to conclude, and what M20 measured instead.** It read "ALS remains the
model whose neighbourhoods you would show a reader" — off *this* column, three anchors, by
eye. Read it again: after the re-base **item-item and ALS return the same four books in the
same order**. The sentence outlived its own evidence by four milestones. M20 put a number on
the question it was answering: on 13,580 anchors the item query **cannot separate them**
(L80, 169/155, p = 0.47), and in the ≥50-reader band the app actually serves, item-item is
nominally ahead (L81). What the number *does* settle is L34's other half — the support floor
is worth more than the choice of model (L82).

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

**Both numbers behind that recommendation were *bounds* until 09.08.2026, and the hybrid
has now been run.** L50 is a ceiling on what a hybrid could reach and L60 is an overlap;
neither is a HitRate. Milestone M19 measured three combination rules — cascade, score fusion
and reciprocal rank fusion — with the prediction written down **before** the run, because a
hybrid is the kind of result everyone expects to win.

**The measured answer, and it is smaller and more interesting than the bound.** The rule this
document has been describing — collaborative first, content filling what it cannot reach —
moves HitRate@10 from **0.0644 to 0.0650** and Coverage@10 from **8.190% to 8.529%** (L76).
Nine readers out of 13,580, every one of them where item-item had nothing, and **no reader
loses a hit**. Paired McNemar p = 0.0039, so it is real; it is also about half a percent of
item-item's hits. Two other rules score higher — score fusion 0.0690, RRF 0.0687 — and
**neither is recommended**: L79 measures that they buy their accuracy by taking it from
well-evidenced readers (fusion is +48/−75 in the best-supported stratum) and that they fill
**28.8%** and **33.9%** of readers' lists with another edition of a book those readers
already own. On the three demo anchors that is 11 and 13 bad slots out of 30, against **0
for the cascade**. And the tuning that produced fusion's α = 0.6 cannot be distinguished
from parameter-free RRF at all (162/158, p = 0.867), so it bought nothing.

**The prediction was falsified as stated and held in its reasoning**, which is worth saying
in those words: it expected coverage to move a lot and accuracy barely, and no rule did both.
What it got right is the part that matters for the architecture — the extra *reachable*
ceiling is mostly unrankable, and the hybrid's real accuracy gains come from somewhere the
prediction never considered. L73 says the same thing from the other side: the works only a
content layer can reach are the works with no interaction evidence, and there TF-IDF scores
0.0304 against every collaborative model's 0.0000 — a real number, and a small one.

**ALS kept in the plan for what the metrics do not show.** Free personalization from the
same fit, item-to-item neighbourhoods that hold up where its HitRate does not (M20: level
with item-item on the item query, L80, and an advantage on thin anchors, L81), and the only
model that ports to Spark without a rewrite — which makes productionization a port rather than
a second project. Two superlatives have been removed from this paragraph by measurement: it
claimed the *best* neighbourhoods of any model until L80 found them level with item-item, and
it called the thin-anchor advantage the *only* one until L85 found RRF and fusion ahead of
item-item in the 1–4 band too. ALS's advantage there is real; it is not unique.

**None of this is expensive to run, and that is measured too.** The model artefact is
**155.6 MB** of float32 factors, the precomputed answer table for the whole product is
**1.5 MB**, and a full retrain of all six models is **about eight minutes** on a laptop
(L71, L72). Compute is not the constraint on this system; evidence is. The sizing argument
belongs to Part 3 and is only pointed at here so that "what would be built" and "what it
costs to run it" are not two disconnected claims.

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

One dataset, **one split, one draw** — seeded and reproducible. The uncertainty across users
is measured: 95% Wilson intervals run ±0.002 to ±0.004, and paired McNemar distinguishes
every pair in §4 except embeddings against the baseline (190 wins to 210 losses, p = 0.342);
the narrowest that does clear the bar is ALS against item-item at p = 2.7e-06 (L74). It
confirms the plain binomial standard error, ±0.0010 to ±0.0021, that was derived before it
was run.

**What that does not cover, said plainly because it is the kind of thing a reader should not
have to find.** Both tests hold the *drawn* held-out item fixed and ask about sampling across
users. Seed 42 also chooses **which** of a reader's favourites is held out, and on a
catalogue this long-tailed that draw carries real variance of its own — L73 shows why: hit
rates differ by a factor of seven between the lowest and highest support stratum, so it
matters a great deal which stratum the drawn book landed in. Several seeds would measure it,
and **M21 did: five draws (seeds 42, 44, 45, 46, 47), ledger L86.** The ordering holds on all
five — no row changes place — but the near-ties move, and the published cell sits at the
favourable end of its range. And every metric is a proxy:
"was the held-out book in the top ten" stands in for "would a reader click, buy, or
enjoy this". A recommendation the reader has never heard of scores zero whether it was a
brilliant discovery or a mistake — which is precisely the outcome a long-tail recommender
exists to produce.

**A live A/B test is the only thing that settles the real question**, and that stays true
however favourable the offline table looks. What the offline work buys is the right to
choose which two or three candidates go into that test, and the confidence that they were
not chosen by accident.

## 9 · Open questions, and how the closed ones closed

1. **Edition clustering: serving-layer dedup, or a data-prep fix? — both, and they do
   different jobs.** §10 has the comparison; the table itself moved to work level in §11.
   Serving dedup stays in the serving layer for the app, because the app's engine is fitted
   on the full interaction matrix and still has to collapse editions on the way to the screen.
2. **Which model should drive the app? — answered three times, and the third answer is the
   one that ships.** First ALS, on neighbourhoods judged better by eye (L34). Then M20
   measured the item query and found the two **not distinguishable** (L80), with item-item
   nominally ahead in the band the app serves (L81) — which retired the *evidenced* claim and
   left only a defensible one. M23.10 stopped trying to pick: **both ship, behind a picker, at
   the same anchor floor**, so the visitor moves one variable and the project does not have to
   assert a winner it cannot demonstrate (§13).
3. **Re-tune item-item for the similarity endpoint? — no re-tuning was needed.**
   L29 proposed a higher λ or a co-occurrence floor for the
   Harry Potter neighbourhood. On the work basis the same model with the same λ returns
   *Chamber of Secrets* at 0.477 (§6, L53): the anchor was under-*evidenced*, not
   under-damped, and the fix was data prep. The legitimate residue — tuning the similarity
   endpoint on its own validation objective rather than on HitRate — is still unbuilt, but
   it is now a refinement rather than a defect.
4. **Confidence intervals — measured, L74.** 95%
   Wilson intervals on all six cells, running **±0.002 to ±0.004**, plus the paired McNemar
   this item proposed. §8 carries the derived standard error that preceded it, and the
   measurement agreed with it: every pair in the table is distinguishable except embeddings
   against the baseline (190 wins / 210 losses, **p = 0.342**), which had already been
   reworded to a tie on the derivation alone.

   **What L74 does *not* cover, and it is the honest remaining gap.** A Wilson interval and
   McNemar both treat the **drawn held-out item as given** and ask about sampling across
   users. Seed 42 also decides *which* of a user's ≥8-rated books is held out, and on a
   catalogue this long-tailed it matters a great deal whether that book was a bestseller or a
   one-reader title. That variance is invisible to both tests. Re-running the split under
   several seeds is the way to measure it, it is cheap, and **it is now done — M21, ledger
   L86.** Five draws: the ordering holds on all five, and the caveat that survives is narrower
   and sharper than the worry was. ALS against item-item is separable on four draws of five,
   and the p = 2.7e-06 quoted above is seed 42's.

   **Why multi-seed runs are comparable at all**, since the obvious objection is that
   re-seeding changes which users are eligible: it does not. Eligibility in `split.py` is
   `≥5 explicit ratings and ≥1 rating ≥8`, a deterministic property of the data computed by
   an `intersect1d` with no seed near it. The seed enters at exactly one line, `rng.integers`,
   choosing the held-out item. The eligible set is **identical across seeds**, so the two
   measurements answer different questions and the project wants both: McNemar for "is this
   difference bigger than user-sampling noise" (L74), seeds for "does the conclusion survive
   a different draw" (L86).
5. **The hybrid was argued from bounds for four milestones, and M19 measured it.** It had
   rested on the union ceiling (L50) and the near-disjoint reach (L60), neither of which is a
   HitRate. Three rules were run with the prediction on record beforehand; §7 carries the
   result. What stays open is narrower and is worth keeping in view: the fusion rules'
   duplicate problem is L47's edition/translation ceiling arriving in a new place, so fixing
   that clustering is the first thing to re-measure fusion against; and the cascade's support
   floor was measured at 0, 5 and 20 with nothing in between, so the point where the trade
   turns negative is still unmeasured.
6. **Cross-lingual lookup is weak** (L38) — `"herr der ringe"` finds nothing. Title+author
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
number was not being flattered either. The +18% was first published as a *lower bound*,
because λ and the neighbourhood size were still the ones tuned on the ISBN-level split; the
sweep was then re-run at work level in M12 and re-selected the same λ=10 and 50 neighbours
(L51), so the caveat is retired and the figure stands on its own.

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
  model measured both ways was too large to leave on the table, so the whole comparison
  table re-based to works. §11 is what that cost to check.
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
- **The work-level strata are measured** (L73, M18.2): both of them, all six models, grouped
  from the same per-user hit vectors as L74. What the work-level version adds is the hybrid
  argument as a measurement rather than as a ceiling — the content models are the only ones
  that score at all where the held-out work has no train interactions (TF-IDF 0.0304,
  embeddings 0.0138, every collaborative model exactly 0.0000).
- **Still one split, still offline.** §8 applies unchanged, and re-basing does not make an
  offline proxy any less of a proxy.

## 12 · The demo, and what it deliberately contradicts (M13)

`streamlit run app/main.py`: paste a book, get ten similar books, each with one sentence of
reason drawn from countable evidence — **co-reader count and shared author** — beside the
similarity, shown as a bar scaled to the top of that list plus the absolute number. No
language model anywhere in the hot path. It starts in **8.8 s** and answers in **21 ms**
(L91; L69 measured 10.6 s / 20 ms for the single-engine app that preceded the picker, and
L61 9.4 s / 21 ms before the M15 surface rebuild), with no network and no fitting at query
time.

**Its default engine is ALS, which loses §4** — and since M23.10 the other engine is one
click away rather than one branch away (§13). That is the point rather than an oversight. §6 measured
the divergence: HitRate@10 scores how well a model ranks a held-out book in a *user's*
history, and the app asks a different question — given this one book, what is like it. ALS
is second of six on the first (L55), and on the second — measured in M20 on 13,580 anchors —
it is **level with the model that wins the first** (L80). The sidebar shows the table where
ALS loses, next to the results it produces, and says that the second question was measured
too. A reviewer can then ask the question, and there is an answer with a number in it rather
than a preference: *we could not tell them apart on the question the demo asks, so we kept
the one that also gives us personalization and a Spark port for free.*

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

**Two things the surface says explicitly, because it was caught not saying them** (M15, M17).
The list is sorted by similarity, and until M17 that quantity was nowhere on screen while the
only visible quantity — shared readers — contradicted the order (rank 1: 16 readers, rank 2:
17). It now shows both and says which one sorts. The picker got a measured relevance cutoff
in the same pass: a candidate more than **0.12 cosine** below the best match is not an
alternative reading of the query, which stops *A Little Princess* and a Stephen King being
offered beside *The Little Prince*, and the cutoff sits at twice the tie margin so it can
never change what a query resolves to (L70).

**The `series` field is computed and deliberately not shown** (M17.6). It holds the title's
trailing parenthetical, which is a real series often enough to have earned the name and a
publisher imprint or format note the rest of the time — *Penguin Classics* on 378 works,
*Dover Thrift Editions* on 268, *Harry Potter (Paperback)* on the demo's own anchor. A "same
series" tag built on exact string equality would therefore assert a publisher as a series and
miss the real ones, which appear as *Vampire Chronicles (Paperback)*, *The Vampire
Chronicles, Book 6* and *Vampire Chronicles, No 5*. A real series entity is a data-layer
project and is on the roadmap, not in the demo. The field's role in the clustering key (§10)
is unaffected — it is stripped before the key is built, and L48 prices that at 0.023%.

**What the demo cannot hide.** `"herr der ringe"` and `"hobit tolkien"` still find nothing,
under every rule. Title+author is three to five words, and no amount of serving logic turns
that into enough signal for a multilingual encoder to bridge. It is the same wall as §10's
gallery and L59's count, now hit from a third direction — and the third independent
argument for the metadata-enrichment layer.

## 13 · The demo's own question, and the floor underneath it (M14–M23.10)

§12 describes an app that runs one engine. It now offers two, and the four milestones
between those states changed what this document can claim — not by moving a cell in §4, but
by measuring the question §1 said the table could not answer.

### The score is not comparable across anchors

Reading eleven anchors instead of a table found what no cell shows: **the similarity score
means different things for a well-read book and a thinly-read one** (L63). A cosine of 0.49
is an excellent neighbour under one anchor and noise under another, because the number of
readers behind it differs by two orders of magnitude. That is a property of the surface, not
a bug, and it has one consequence the product cannot avoid: a demo that answers for every
book will sometimes answer confidently from four shared readers.

So the floor became **two** numbers (L65). A **candidate floor of 20** keeps noise
directions out of the neighbourhood — L34's original finding. An **anchor floor of 50**
decides which books the app will answer for at all: 2,508 works, 26.6% of all interactions.
Below it the app declines rather than inventing ten titles, and *The Kite Runner* at 39
readers (L90) is the pinned demonstration of that refusal.

### The item query, measured

M20 gave the product's question its own metric — **AnchorHitRate@10**: hold out a reader's
book, ask what is similar to one book they kept, check whether the held-out one comes back.
Over the same 13,580 anchors: **ALS 0.0308, item-item 0.0297, and the two cannot be told
apart** (L80, 169/155, p = 0.47). M22 added the fusion rules and they win the aggregate —
**RRF 0.0371, score fusion 0.0355** (L85) — but the win does not survive the band the app
serves: against item-item above the anchor floor they go **41/42, p = 1.000**. Every point
is bought below 50 readers, where the demo does not answer.

This is the same shape twice, so it is worth naming: **every engine this project has
measured is separated below the floor and indistinguishable above it.**

### The floor is worth more than the model

M23 put a number on that. Running ALS with and without L34's candidate floor: **0.0308
against 0.0130**, a difference of **+0.0177** at p = 2.5e-41 — larger than any difference
between two models anywhere in this project (L82). Without the floor ALS scores *below the
popularity baseline*. The support floor is not a hygiene detail; on this data it is the
single most consequential setting in the serving path, and it is a data-shape decision
rather than a modelling one — which makes it a sibling of §11's re-base, not a footnote.

The band a lower floor would open (20–49 readers, 1,860 anchors) cannot settle the engine
question either way: at ~3% hit rates it holds 23 and 13 discordant readers across two
draws, so it can only fail to separate (L88). Reported as underpowered rather than resolved
by taking the friendlier draw.

### Every number is conditional on one draw

M21 re-drew the split five times (seeds 42, 44, 45, 46, 47). **The ordering holds on all
five — no row of §4 changes place** — and three near-ties change their significance verdict
between draws (L86). §8's caveat is therefore bounded rather than removed: the table's
*shape* is robust, its *margins* are not, and the published cell sits at the favourable end
of its range.

### Two engines, one floor, and a hand read behind both

M23.10 ships **A** (ALS item factors) and **B** (item-item, shrunk cosine) behind a sidebar
picker, both at anchor floor 50, so the click moves exactly one variable: the model. A is
default because it is the rehearsed engine, not because it won anything, and one constant in
`app/main.py` removes the picker.

The pair is backed by reading, not by a metric — because §6's lesson is that a metric cannot
see whether a neighbourhood is sensible:

- **240 slots** in the band a lower floor would open, three configurations, every one read by
  hand (L89). It killed the third configuration: a text-blended engine that answered
  *Thank You for Smoking* by Christopher **Buckley** with two Christopher **Pike** novels at
  zero shared readers — 17 of 80 slots were that failure.
- **440 slots** across twenty anchors in the band the app serves (L90). **A: 7 bad slots of
  220. B: 0 of 220.** A's four unambiguous failures are one defect — the same book returned
  twice under two work keys a leading article apart, which B cannot make structurally,
  because two halves of a split work share almost no readers.

That second audit also found something that is not about engines at all (L90): **10,737
groups covering 23,091 works are one book under two work keys**, and 194 of them are
silenced at the floor — each half below 50 while the sum clears it. Merging them would add
7.7% to the askable catalogue without touching the floor or the model. It is L47's wall
again, measured from a third side.

**What the switch costs: nothing on a stopwatch** (L91). B's whole answer table is 25,080
rows / 0.22 MB, built in 21.8 s, and A comes back byte-identical on all 110 rehearsed slots
through both entry points. Cold start 8.8 s (A) / 8.5 s (B), warm query 21/22 ms, assets
unchanged at 890 MB because the table is additive.

### What this changes in §7

**Nothing in the model recommendation, and one thing in the order of work.** Item-item CF
plus a content layer is still what would be built. But the first lever to reach for is not a
model at all: it is the support floor, and after that the work key. Both are data-shape
decisions, both were worth more than any model swap measured here, and both are cheaper to
change than a training pipeline.
