"""The demo: name a book, get similar books, each with the evidence behind it.

    streamlit run app/main.py

Deliberately thin. Every rule that could be *wrong* lives in :mod:`recommender.demo`
(ranking) and :mod:`recommender.display` (presentation), both tested offline; this file is
layout and copy. Run ``python scripts/build_app_assets.py`` once first — the app never fits
a model, never reads ``data/`` and never touches the network.

**One engine, on purpose.** The similar-items engine is ALS item factors over the
work-keyed matrix, with the support floors from ledger L34 (candidates) and L65 (anchors).
ALS places *third of six* on HitRate@10 in the published table (L55) and has the best
item-to-item neighbourhoods in the project. That the offline metric and the product surface
disagree is the finding, not an inconsistency — so the table where ALS loses is in the
sidebar rather than hidden. A switcher across the other four models was scoped and dropped:
each would need its own similarity artefact, which buys a dropdown and costs the cold start
this app is built around.

**Ledger codes appear in this file and never on the screen** (M15, pinned decision 1). They
are precise to us and are internal jargon to a client, and jargon in a demo reads as
unfinished rather than as rigorous. Comments are where they help.

**The ranking on screen is exactly the ranking the ledger measures.** M15 changes how the
list reads and never what is in it or in what order — no re-sort, no filter, no truncation.
The one boundary this file draws, the evidence divider, is a horizontal rule between two
rows that both stay exactly where the model put them.
"""

from __future__ import annotations

import html
import sys
import time

import streamlit as st

try:
    from recommender.demo import DemoEngine, load_assets
    from recommender.display import THIN_EVIDENCE_SHARE, bar_widths, divider_after, evidence_share, is_thin
    from recommender.gallery import DEMO_BUTTONS as ANCHORS  # M14.8 — one source, see its docstring
except ModuleNotFoundError as error:  # pragma: no cover - the wrong-interpreter path
    # `streamlit run` uses whichever interpreter is first on PATH, which on this machine is
    # a pyenv environment that has Streamlit but not this project. The raw traceback points
    # at line 36 of this file and says nothing about the actual problem, so it gets a
    # sentence and the command that fixes it — the same rule the stale-asset check follows.
    #
    # This is the second time an unrelated pyenv environment has produced a phantom failure
    # here; M10 recorded the first, when the Jupyter kernel resolved to one and notebook 01
    # was executed outside the pinned venv.
    st.set_page_config(page_title="Book recommender — demo", layout="wide")
    st.error(
        f"**{error.name} is not installed in the interpreter running this app.**\n\n"
        f"`streamlit run` used `{sys.executable}`, which is not the project's virtual "
        "environment. Start it from there instead:\n\n"
        "```\nsource .venv/bin/activate && streamlit run app/main.py\n```\n\n"
        "or, without activating anything:\n\n"
        "```\n.venv/bin/python -m streamlit run app/main.py\n```"
    )
    st.stop()

#: The free-text example, offered as a click rather than as a placeholder nobody types. It
#: is the query that took two serving rules to make work (L62), so it is worth being seen.
EXAMPLE_QUERY = "harry potter stein"

#: Everything visual that Streamlit's theme cannot express. Kept in one block so the layout
#: can be read in one place, and deliberately small: the theme in `.streamlit/config.toml`
#: does most of the work.
STYLE = """
<style>
  /* Full width, by Helena's decision on seeing it run (08.08.2026), reversing M15.2's
     46rem measure. M15.2's argument was typographic — body text at 1920px runs long — and
     the DoD item about a ~90-character measure goes with it; the deviation is recorded
     under M15 in STRATEGY.md rather than quietly dropped. What keeps it readable at full
     width is the row itself: the evidence line sits directly under the title in its own
     block, so a wide viewport stretches the whitespace to the right of each row rather
     than the text inside it. */
  .block-container {max-width: none; padding-top: 2.2rem; padding-left: 3rem; padding-right: 3rem;}
  section[data-testid="stSidebar"] {min-width: 21rem;}

  /* The button row: the labels wrap to different numbers of lines, which put the reader
     counts on different baselines. A fixed height squares the row. Selected by testid —
     the emotion class names are generated per build and `div.stButton > button` does not
     match the current DOM, which is why the first attempt silently did nothing. */
  [data-testid^="stBaseButton"] {min-height: 3.4rem; white-space: normal;
                                 line-height: 1.2; font-size: 14px;}
  /* ...except the tertiary "try this query" link, which is a line of text, not a target. */
  [data-testid="stBaseButton-tertiary"] {min-height: 0; height: auto; font-size: 13px;}

  .provenance {font-size: 13px; color: #6b6862; margin: -0.6rem 0 1.6rem 0;}
  .legend {font-size: 12px; color: #6b6862; margin: 0.2rem 0 0.9rem 0;}

  /* One row of the result list. No card, no fill, no border — a hairline and the type
     hierarchy carry it. */
  .row {display: flex; gap: 14px; padding-top: 14px; border-top: 0.5px solid #e2ded7;}
  .row .rank {width: 24px; flex: 0 0 24px; text-align: right; font-size: 14px;
              color: #9b968d; font-variant-numeric: tabular-nums; padding-top: 1px;}
  .row .body {flex: 1 1 auto; min-width: 0;}
  .row .title {font-size: 15px; font-weight: 500; color: #1c1c1e; line-height: 1.3;}
  .row .meta {font-size: 13px; color: #6b6862; margin-top: 7px;}
  .row .ev {display: flex; align-items: center; gap: 10px; margin-top: 7px; flex-wrap: wrap;}
  .row .track {width: 88px; height: 4px; border-radius: 2px; background: #e7e3dc; flex: 0 0 88px;}
  .row .fill {display: block; height: 4px; border-radius: 2px; background: #8a9aa5;}
  .row .evtext {font-size: 13px; color: #6b6862; font-variant-numeric: tabular-nums;}
  .row .tag {font-size: 12px; color: #4a5b66; background: #eceef0; border-radius: 999px;
             padding: 1px 8px;}
  .row .tag.thin {color: #7a6a55; background: #f3eee4;}

  /* The divider. Quiet on purpose: it is a label, not a verdict. */
  .cut {display: flex; align-items: center; gap: 12px; margin: 22px 0 4px 0;
        font-size: 12px; color: #9b968d;}
  .cut::after {content: ""; flex: 1 1 auto; height: 0.5px; background: #e2ded7;}
</style>
"""


@st.cache_resource(show_spinner="Loading the model assets…")
def get_engine() -> tuple[DemoEngine, float]:
    started = time.perf_counter()
    engine = DemoEngine(load_assets())
    return engine, time.perf_counter() - started


def corpus_line(engine: DemoEngine) -> str:
    """The one place in the app that says how big the corpus is.

    Counted off the assets at runtime rather than typed in. The work count moved from
    235,824 to 234,626 when the serving work key changed (L64), and a hard-coded number
    would have been stale within two days of being written.
    """
    assets = engine.assets
    return (
        f"{assets.readers.nnz:,} ratings from {assets.readers.shape[1]:,} readers "
        f"across {len(assets.books):,} works. Book-Crossing, crawled in 2004."
    )


def render_row(rank: int, suggestion, width: float) -> str:
    """One result row as a single block of markup, so the layout cannot drift between rows."""
    evidence = suggestion.evidence
    share = evidence_share(evidence)
    tags = []
    if evidence.same_author:
        tags.append(("", "same author"))
    if evidence.shared_series:
        tags.append(("", f"same series ({evidence.shared_series})"))
    if is_thin(evidence):
        tags.append(("thin", "thin evidence"))

    if evidence.co_readers:
        readers = f"{evidence.co_readers:,} shared reader" + ("s" if evidence.co_readers != 1 else "")
        evtext = f"{readers} · {share:.1%}"
    else:
        # No shared readers at all: the model is speaking from its own geometry. Say that
        # rather than printing "0 shared readers · 0.0%", which reads as a broken row.
        evtext = "no shared readers — from the model's geometry alone"

    meta = " · ".join(part for part in (suggestion.author, suggestion.year) if part)
    pills = "".join(f'<span class="tag {kind}">{html.escape(text)}</span>' for kind, text in tags)
    return (
        f'<div class="row"><div class="rank">{rank}</div><div class="body">'
        f'<div class="title">{html.escape(suggestion.title)}</div>'
        f'<div class="meta">{html.escape(meta)}</div>'
        f'<div class="ev"><span class="track"><span class="fill" style="width:{width:.0%}"></span></span>'
        f'<span class="evtext">{html.escape(evtext)}</span>{pills}</div>'
        f"</div></div>"
    )


def sidebar(engine: DemoEngine, asset_seconds: float) -> None:
    """The pinned M15.5 copy.

    **Register, which is the decision behind every line.** This is a demo shown to a client
    two weeks into an engagement, not a defence to an examiner. A client does not read a
    methodology essay before the first result, does not know what a ledger code is, and does
    not need the mechanism unless they ask. So: no internal codes, no argument the screen
    has not earned yet, plain nouns, and the model names written out. The substance stays —
    the floor, the refusal, the evidence, the table where this engine loses — in fewer words.
    """
    with st.sidebar:
        st.markdown("### What this is")
        st.markdown(
            "One book in, similar books out. No login and no reading history: the only "
            "input is the book you name."
        )
        st.markdown("### How it works")
        st.markdown(
            "Reading patterns from 1.1 million ratings, compressed into a short profile per "
            "book. Books whose profiles point in the same direction come back as similar. "
            "Editions of the same title are merged before anything is computed, so the "
            "results are books rather than reprints."
        )
        st.markdown("### Where it stops")
        st.markdown(
            f"Below {engine.assets.anchor_floor} readers the demo declines to answer. The "
            "model would still return ten titles and they would be noise. Every result "
            "names the readers behind it, so you can see how much it rests on."
        )
        st.markdown("### The engine")
        st.markdown(
            "This demo runs on matrix factorization. It comes third on the accuracy table "
            "below and produces the best \"books like this one\" lists of the six approaches "
            "we tried. Those are two different questions, and a demo like this one asks the "
            "second."
        )
        # The table keeps its published form and numbers (L52-L57); only the labels change,
        # from house abbreviations to written-out names. It stays expanded, because showing
        # the table where this engine loses is the point.
        st.markdown(
            "| Approach | Hit rate @10 |\n|---|---:|\n"
            "| Item-based collaborative filtering | 0.0644 |\n"
            "| **Matrix factorization (ALS) · this demo** | **0.0545** |\n"
            "| Item-based CF, explicit ratings only | 0.0486 |\n"
            "| Content-based, TF-IDF on title and author | 0.0405 |\n"
            "| Popularity baseline | 0.0155 |\n"
            "| Content-based, multilingual embeddings | 0.0141 |"
        )
        st.caption(
            "Hit rate @10: how often a reader's held-out book turns up in their top ten. "
            "Same data split for all six."
        )
        st.caption(
            "Cover images are omitted: the dataset's image links date from 2004 and no "
            "longer resolve."
        )
        # Deviation from the pinned copy, and the only one: it read "cold start 0.3 s". That
        # figure is the asset load, while the app's measured cold start is 9.4 s (L61) — most
        # of it the sentence encoder for the free-text box. Printing 0.3 s under the words
        # "cold start" would put a number on the surface that contradicts a ledger line, so
        # the label says what the number measures. Nothing else in the copy changed.
        st.caption(f"Runs offline from precomputed data · assets loaded in {asset_seconds:.1f} s.")
        st.caption("Every number on this screen is measured. The measurement ledger is in the repository.")


def main() -> None:
    st.set_page_config(page_title="Book recommender — demo", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    try:
        engine, asset_seconds = get_engine()
    except (FileNotFoundError, RuntimeError) as error:
        st.error(f"{error}\n\nRun `python scripts/build_app_assets.py` first.")
        return

    sidebar(engine, asset_seconds)

    st.title("Name a book, get books like it")
    st.markdown(f'<div class="provenance">{html.escape(corpus_line(engine))}</div>', unsafe_allow_html=True)

    columns = st.columns(len(ANCHORS))
    for column, (label, isbn) in zip(columns, ANCHORS.items(), strict=True):
        if column.button(label, use_container_width=True):
            st.session_state["query"] = label
            st.session_state["pinned_isbn"] = isbn
        # The reader count belongs on the button. Without it a thinly read anchor looks like
        # a broken app; with it, the calibration story (L63) is visible in the product
        # rather than only asserted on a slide.
        column.caption(f"{engine.describe(isbn).readers:,} readers")

    # Every writer of `st.session_state["query"]` has to run **before** the text input that
    # owns that key is instantiated — Streamlit raises `StreamlitAPIException` otherwise,
    # and the first version of this had the example button underneath the field, where it
    # crashed the app on click. The anchor buttons above were always on the right side of
    # that line; this one was not.
    if st.button(f"Try “{EXAMPLE_QUERY}”", type="tertiary"):
        st.session_state["query"] = EXAMPLE_QUERY

    query = st.text_input(
        "Book",
        key="query",
        label_visibility="collapsed",
        placeholder="Type a title however you remember it",
    )
    if not query:
        return

    started = time.perf_counter()
    pinned = st.session_state.pop("pinned_isbn", None)
    matches = engine.find(query, k=5)
    if not matches:
        st.warning("Nothing found. Try a title and an author.")
        return

    chosen = pinned or matches[0].isbn
    # The picker is reserved rather than conditional, so the result does not jump down the
    # screen the moment a second candidate exists.
    picker = st.container()
    if len(matches) > 1 and not pinned:
        with picker:
            labels = {f"{m.title} — {m.author}": m.isbn for m in matches}
            chosen = labels[st.radio("Did you mean", list(labels))]

    book = engine.describe(chosen)
    st.subheader(book.title)
    st.caption(
        f"{book.author} · {book.year}"
        + (f" · {book.series}" if book.series else "")
        + f" · {book.readers:,} readers"
    )

    suggestions = engine.similar(chosen, k=10)
    elapsed = time.perf_counter() - started
    if not suggestions:
        st.info(
            f"We do not know this book well enough. {book.readers:,} readers in this dataset "
            f"is below the {engine.assets.anchor_floor} we ask for. The model would still "
            "produce ten titles and they would be noise, so it declines instead."
        )
        return

    st.markdown(f"#### Because you liked *{book.title}*")
    st.markdown(
        f'<div class="legend">Bars compare shared readers <em>within this list only</em>. '
        f"“Thin evidence” marks a book fewer than {THIN_EVIDENCE_SHARE:.0%} of this book's "
        f"readers also read.</div>",
        unsafe_allow_html=True,
    )

    widths = bar_widths(suggestions)
    cut = divider_after(suggestions)
    for index, (suggestion, width) in enumerate(zip(suggestions, widths, strict=True)):
        st.markdown(render_row(index + 1, suggestion, width), unsafe_allow_html=True)
        if cut is not None and index == cut:
            st.markdown('<div class="cut">further out: few shared readers</div>', unsafe_allow_html=True)

    st.caption(f"Answered in {elapsed * 1000:.0f} ms.")


main()
