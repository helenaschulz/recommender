# Book Recommender — research project on the Book-Crossing dataset

A personal research project: build a book recommender end to end on the
[Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
— explore the data, try the natural modelling approaches, evaluate them honestly, and
end in an interface where you paste a book and get recommendations back, with an
explanation of *why* those books.

Optimized for a clear, honest, demoable story and for learning something — not for
shipping a product. Offline metrics are treated as a proxy; a live A/B test would be
the real proof.

The work runs in three stages:

1. **Data understanding** — what this dataset actually is, and which modelling decisions
   it forces. See [`docs/dataset_findings.md`](docs/dataset_findings.md) and
   [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).
2. **Modelling and evaluation** — a ladder of candidate models on one pinned split, each
   measured on accuracy, catalogue coverage and novelty. See
   [`notebooks/02_models.ipynb`](notebooks/02_models.ipynb), the write-up
   [`docs/model_selection.md`](docs/model_selection.md) and the ledger
   [`docs/RESULTS.md`](docs/RESULTS.md).
3. **Interface and productionization** — paste a book, get recommendations with an
   explanation; plus how such a system would scale and stay fresh on a data platform.

## Getting the data

The raw CSVs are **not** in this repo (they are large, and licensed by their source).
Download them from Kaggle — dataset `arashnic/book-recommendation-dataset` — and place
the three files in `data/`:

```
data/Books.csv     # ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-*  (~271k)
data/Ratings.csv   # User-ID, ISBN, Book-Rating (0-10)                                            (~1.15M)
data/Users.csv     # User-ID, Location, Age                                                       (~279k)
```

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle datasets download -d arashnic/book-recommendation-dataset -p data --unzip
```

All files are comma-separated. Note that `Ratings.csv` has **no timestamp column** — this
shapes how evaluation splits are done (see [`docs/dataset_findings.md`](docs/dataset_findings.md)).

## Environment setup

Python 3.11, in a virtual environment:

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

To run the notebooks in this environment, register its Jupyter kernel once — otherwise
`jupyter` may silently execute them against a different interpreter:

```bash
python -m ipykernel install --user --name recommender --display-name "Python 3.11 (recommender)"
```

Then: `pytest` for the test suite, `ruff check .` for lint, and the model runner to
reproduce the **primary** comparison table. The item is a *work*, not an ISBN — editions of
the same book are merged before the split (see `docs/RESULTS.md` L49):

```bash
python scripts/run_model.py --all --gallery --work-level
```

Drop `--work-level` to reproduce the ISBN-keyed table instead. That one is kept in the
ledger as the journey record; the two are **not** cell-comparable, because they have
different items and different coverage denominators.

The edition-clustering work (same book, many ISBNs) has its own entry points:

```bash
python scripts/analyze_editions.py --write-sample docs/edition_clusters_sample.md
```

```bash
python scripts/analyze_dedup.py
```

And the measurement that explains why the two tables differ by so much more for the text
models than for the collaborative ones (ledger L58):

```bash
python scripts/decompose_work_level_lift.py
```

## The demo

Paste a book, get ten similar books, each with one grounded reason. Build the assets once
(~4 minutes; writes ~900 MB to the gitignored `artifacts/app/`), then the precomputed
answer table for configuration B (~22 seconds, adds 0.22 MB):

```bash
python scripts/build_app_assets.py
python scripts/build_answer_table.py
```

Then start it — cold start under 9 seconds, no network, no fitting:

```bash
streamlit run app/main.py
```

The app ships **two selectable engines** behind a sidebar picker. Configuration A, the
default, is ALS item factors over the work-keyed matrix. Configuration B is the comparison
table's accuracy winner, item-item collaborative filtering with a shrunk cosine, answering
from the precomputed answer table above. Both run on the same support floors (candidate
floor 20 per ledger L34, the line showing ALS needs one; anchor floor 50), so the picker
moves exactly one variable: the model. The switch costs nothing measurable — cold start
8.8 s (A) / 8.5 s (B), warm query 21/22 ms, assets unchanged (L91, the switch-cost
measurement). A single flag in `app/main.py` (`SHOW_ENGINE_PICKER`) rolls back to A-only.

The reason sentences come from structured evidence only — co-reader counts, shared author,
shared series, similarity — never from a language model. Screenshots of the three anchor
flows are in [`docs/img/`](docs/img/); `python scripts/measure_app_latency.py`
(per configuration via `--configuration A|B`) and `python scripts/audit_app_lookup.py`
reproduce the latency and lookup-audit ledger lines (L61/L91 and L62).

## Repository layout

| Path | What lives there |
|---|---|
| `data/` | The raw Book-Crossing CSVs (not committed — see above). |
| `notebooks/` | The journey: `01_eda.ipynb` (data understanding), `02_models.ipynb` (model comparison). |
| `docs/` | [`RESULTS.md`](docs/RESULTS.md) — the measurement ledger — plus [`dataset_findings.md`](docs/dataset_findings.md), the model-selection write-up [`model_selection.md`](docs/model_selection.md), the two hand audits [`floor_band_audit.md`](docs/floor_band_audit.md) and [`anchor_set_audit.md`](docs/anchor_set_audit.md), the hand-read samples [`edition_clusters_sample.md`](docs/edition_clusters_sample.md) and [`work_key_punctuation_sample.md`](docs/work_key_punctuation_sample.md), and figures under `docs/img/`. |
| `app/` | The Streamlit demo (`main.py`) — widgets only; its engines are `recommender.demo` and `recommender.engines`. |
| `src/recommender/` | One module per responsibility: `data.py` / `split.py` (prep and the pinned split), `models/` (the six candidates plus the hybrid rules), `eval.py` / `benchmark.py` (metrics and the one place the evaluation universe is assembled), `gallery.py`, `serving.py` / `display.py` (work-level dedup, and presentation rules that must not reorder anything), `demo.py` / `engines.py` / `answers.py` (the demo's engine, its selectable configurations, and the precomputed answer table). |
| `scripts/` | Entry points, grouped: `run_model.py` and `tune_*.py` (the comparison table), `measure_significance.py` / `measure_seed_sensitivity.py` (its statistical checks), `measure_hybrid.py` and `measure_anchor_hitrate.py` (the hybrid rules and the item-query metric), `analyze_editions.py` / `analyze_dedup.py` / `decompose_work_level_lift.py` / `analyze_work_key_punctuation.py` (the edition-clustering work), `audit_floor_band.py` / `audit_anchor_set.py` / `recut_anchor_bands.py` (the hand audits and the band the floor decision turns on), `build_app_assets.py` / `build_answer_table.py` / `verify_configuration_a.py` / `measure_app_latency.py` / `audit_app_lookup.py` / `capture_app_screenshots.py` (the demo), `run_demo_anchors.py` and the `analyze_*.py` reads that interrogate its output — anchor support, hit strata, recurrence, truncation, picker margin, surface stability — and `measure_serving_footprint.py` (the Part 3 sizing numbers). |
| `tests/` | Offline, deterministic tests — no network, no model downloads. |

[`docs/RESULTS.md`](docs/RESULTS.md) is the measurement ledger: every headline number used
anywhere in this project traces to a line there, including the negative results.

## Status

Data understanding, the model comparison and the interface are done. Six models —
popularity baseline, item-item CF (plus an explicit-only ablation), ALS, content TF-IDF
and multilingual sentence embeddings — are fitted on one pinned leave-one-out split and
measured on HitRate@10, Coverage@10 and Novelty@10; the table and every negative result
behind it are in [`docs/RESULTS.md`](docs/RESULTS.md), and the reasoning behind the choice
is written up in [`docs/model_selection.md`](docs/model_selection.md).

Since the table was published, the comparison has been stress-tested rather than extended.
Every pairwise comparison carries Wilson intervals and a paired McNemar test (L74, which
also sets the split's resolution at about ±0.002–0.004), and the ordering survives five
independent split draws, although three near-ties change their significance verdict
between draws (L86, the seed-sensitivity sweep). The hybrid is measured, not bounded: the
recommended cascade/backfill rule improves both accuracy and coverage with zero readers
losing a hit (L76), while the two higher-scoring rules, score fusion and RRF, fill roughly
a third of readers' lists with another edition of a book they already own and are not
recommended (L79, the gallery read that caught it).

On the demo's own question (one book in, ten out), the sharpest finding is that the
support floor is worth more than the model choice: ALS scores 0.0308 with the floor and
0.0130 without it, below the popularity baseline (L82), and in the band the app actually
serves, the engines are statistically indistinguishable (L85). That is why the demo ships
two engines behind a picker instead of declaring a winner. Two hand audits back the
shipped pair: 240 and then 440 hand-read recommendation slots (L89 and L90, the
floor-band and anchor-set audits in [`docs/`](docs/)), where B produced zero bad slots
and A stayed within one to seven of 220, with thinner evidence per slot.

The productionization write-up follows; its sizing evidence is already measured — the
demo's 890 MB of serving assets, of which the recommender itself is 17.5% (L71), against
a 1.5 MB precomputed answer table for the whole product (L72).

The published table is keyed by **work** rather than by ISBN — merging editions before
training is the largest single accuracy gain in the project, and it is a data-preparation
change rather than a model one. The ISBN-keyed table is kept beside it as the journey
record, and the reason the two differ far more for the text models than for the
collaborative ones is measured rather than asserted (ledger L58).

## References

The sources this project leans on, in full in
[`docs/dataset_findings.md`](docs/dataset_findings.md):

- Ziegler et al., WWW 2005 — [Improving Recommendation Lists Through Topic Diversification](https://doi.org/10.1145/1060745.1060754),
  the paper the Book-Crossing dataset comes from.
- Naghiaei, Rahmani, Deldjoo, 2022 — [The Unfairness of Popularity Bias in Book Recommendation](https://arxiv.org/abs/2202.13446),
  the popularity-bias benchmark on this dataset.
- Hu, Koren, Volinsky, ICDM 2008 — [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22),
  the weighted-MF formulation behind the ALS model, used via [`implicit`](https://github.com/benfred/implicit).
- Steck, WWW 2019 — [Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375) (EASE).
- [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  — the sentence-embedding model used for the content layer.

## License

The code and documentation in this repository are released under the MIT License — see
[`LICENSE`](LICENSE). The Book-Crossing data is **not** covered by it: it is not part of
this repo and stays under the terms of its own source (see
[Getting the data](#getting-the-data)).
