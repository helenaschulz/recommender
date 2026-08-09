"""The demo: name a book, get similar books, each with the evidence behind it.

    streamlit run app/main.py

Deliberately thin. Every rule that could be *wrong* lives in :mod:`recommender.demo`
(ranking) and :mod:`recommender.display` (presentation), both tested offline; this file is
layout and copy. Run ``python scripts/build_app_assets.py`` once first — the app never fits
a model, never reads ``data/`` and never touches the network.

**One engine, for now.** The similar-items engine is ALS item factors over the work-keyed
matrix, with the support floors from ledger L34 (candidates) and L65 (anchors). ALS places
*second of six* on HitRate@10 in the published table (L55). That the offline metric and the
product surface ask different questions is the finding, not an inconsistency — so the table
where ALS loses is in the sidebar rather than hidden.

**What M20 changed about the reason ALS is here** (L80–L82). This docstring used to say ALS
"has the best item-to-item neighbourhoods in the project", carried from L34 — three anchors
read by eye in M8, before the re-base that made item-item's and ALS's *Harry Potter*
neighbourhoods identical. Measured on 13,580 anchors, **ALS and item-item are not
distinguishable on the item query** (169/155, p = 0.47), and ALS's measured edge lives
entirely below 5 readers of anchor support — a band this app *declines to serve* (L65, L81).
What the measurement does vindicate is the **floor**, not the factorization: without it ALS
scores below the popularity baseline (L82). The engine choice is defensible, not evidenced,
and the honest version of that is now in the sidebar.

**Why there is no engine switcher yet, corrected.** M13.2 dropped one on the grounds that
each engine would need its own similarity artefact and would cost the cold start this app
is built around. **M16 measured that false and this docstring carried the dead reason for
two milestones.** With the anchor floor at 50 only 2,508 works are reachable as anchors at
all, so the artefact is not a second 155.6 MB factor matrix but a precomputed top-N table —
125,400 rows across five engines, 1.5 MB (ledger L72; the matrix is L71). Cold start goes
*down*: a table lookup replaces a 155.6 MB memory map. It is also the Part 3 Gold-table serving
pattern, built instead of drawn on a slide.

What actually holds the switcher back is a **schedule gate, not a cost**: M16 runs only
after the surface work is green and a first full write-up draft exists, because it changes a
demo that has already been rehearsed. That is a different sentence from the one this file
used to make, and the difference matters — the first is a priority, the second was a
measurement, and the measurement went the other way.

**Ledger codes appear in this file and never on the screen** (M15, pinned decision 1). They
are precise to us and are internal jargon to a client, and jargon in a demo reads as
unfinished rather than as rigorous. Comments are where they help.

**The ranking on screen is exactly the ranking the ledger measures.** M15 and M17 change how
the list reads and never what is in it or in what order — no re-sort, no filter, no
truncation. M17 removed the last thing this file drew on top of the list, the evidence
divider: it claimed a contiguous boundary from a criterion that is *not monotone in the sort
order*, and that non-monotonicity is one of this project's own findings, so on some anchor
the line had to be incoherent. A freshly typed anchor produced one immediately.

**M17 put the similarity back on screen** as a bar plus a number. M14.6 had taken it off
because a cosine is not comparable across anchors (L63), which left the list sorted by a
quantity that was nowhere to be seen and one that *was* on screen — shared readers —
visibly contradicting the order. L63 is a claim about comparing *two anchors*; the app shows
one list at a time, and within it the cosine is exactly the sort key. The bar is scaled to
the top of its own list for the same reason the caveat exists, and the caveat is a sentence
in the talk track.

**The one selection rule this file's engine gained is the picker's, and it is M17.4's.**
``find`` now returns only candidates within ``demo.PICKER_MARGIN`` of the best match, so the
"Did you mean" list is often a single row. It cannot change which book a query resolves to;
see the constant's docstring for why that is structural rather than lucky.
"""

from __future__ import annotations

import html
import sys
import time

import streamlit as st

try:
    from recommender.demo import DemoEngine, load_assets
    from recommender.display import THIN_EVIDENCE_SHARE, bar_widths, evidence_share, is_thin
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

#: Everything visual that Streamlit's theme cannot express. Kept in one block so the layout
#: can be read in one place, and deliberately small: the theme in `.streamlit/config.toml`
#: does most of the work.
STYLE = """
<style>
  /* Full width, by decided on seeing it run (08.08.2026), reversing M15.2's
     46rem measure. M15.2's argument was typographic — body text at 1920px runs long — and
     the DoD item about a ~90-character measure goes with it; the deviation is recorded
     with the milestone rather than quietly dropped. What keeps it readable at full
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
  /* The reader count belongs *to* its button, so it sits against it rather than floating in
     the middle of the gap Streamlit puts between two elements in a column. Scoped to the
     keyed container, so every other caption in the app keeps its normal spacing. */
  .st-key-anchor-row [data-testid="stElementContainer"]:has([data-testid="stCaptionContainer"])
    {margin-top: -0.85rem;}

  .legend {font-size: 12px; color: #6b6862; margin: 0.2rem 0 0.9rem 0;}

  /* The sidebar's first heading is the only place the app is named, so it is set as a name
     rather than as one more section label — larger than the "How it works" headings under
     it, which keep their size. Reached through a keyed container: Streamlit wraps every
     markdown block separately, so `h3:first-of-type` matches all four of them. */
  .st-key-app-name h3 {font-size: 1.65rem; margin-bottom: 0.15rem;}

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
  /* The sort key, so it is set a step above the evidence beside it rather than level with
     it: same size, darker, fixed width so the decimal points line up down the list. */
  .row .score {font-size: 13px; color: #1c1c1e; font-variant-numeric: tabular-nums;
               width: 2.4em; flex: 0 0 2.4em;}
  .row .sep {font-size: 13px; color: #c9c4bb;}
  .row .evtext {font-size: 13px; color: #6b6862; font-variant-numeric: tabular-nums;}
  .row .tag {font-size: 12px; color: #4a5b66; background: #eceef0; border-radius: 999px;
             padding: 1px 8px;}
  .row .tag.thin {color: #7a6a55; background: #f3eee4;}
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
    """One result row as a single block of markup, so the layout cannot drift between rows.

    The bar is the **similarity**, scaled to the top of this list, with the absolute value
    beside it; the evidence follows as text. That order is M17.1 and it is the sort key
    first: the number that decides the row's position leads, and the number that says how
    much the position rests on comes second.

    ``series`` is deliberately **not** here (M17.6). The field holds the title parenthetical
    — *Penguin Classics*, *Dover Thrift Editions*, *2nd Edition* — and a "same series" tag
    over a publisher imprint is a false claim printed as fact.
    """
    evidence = suggestion.evidence
    share = evidence_share(evidence)
    tags = []
    if evidence.same_author:
        tags.append(("", "same author"))
    if is_thin(evidence):
        tags.append(("thin", "thin evidence"))

    if evidence.co_readers:
        readers = f"{evidence.co_readers:,} shared reader" + ("s" if evidence.co_readers != 1 else "")
        evtext = f"{readers} · {share:.1%}"
    else:
        # No shared readers at all: the model is speaking from its own geometry. Say that
        # rather than printing "0 shared readers · 0.0%", which reads as a broken row.
        evtext = "no shared readers — from the model's geometry alone"

    # `"0"` is a non-empty string, so the `if part` filter passed it straight through and
    # the screen read "Antoine de Saint-Exupéry · 0" (M17.8). Year 0 is L3, the dataset's
    # missing-year encoding, known since the EDA and arriving here unfiltered; it is 2 of
    # the 10 slots on *The Little Prince*, because translations carry it far more often.
    year = suggestion.year if suggestion.year not in ("0", "") else ""
    meta = " · ".join(part for part in (suggestion.author, year) if part)
    pills = "".join(f'<span class="tag {kind}">{html.escape(text)}</span>' for kind, text in tags)
    return (
        f'<div class="row"><div class="rank">{rank}</div><div class="body">'
        f'<div class="title">{html.escape(suggestion.title)}</div>'
        f'<div class="meta">{html.escape(meta)}</div>'
        f'<div class="ev"><span class="track"><span class="fill" style="width:{width:.0%}"></span></span>'
        f'<span class="score">{evidence.score:.2f}</span>'
        f'<span class="sep">·</span>'
        f'<span class="evtext">{html.escape(evtext)}</span>{pills}</div>'
        f"</div></div>"
    )


def sidebar(engine: DemoEngine) -> None:
    """The pinned M15.5 copy.

    **Register, which is the decision behind every line.** This is a demo shown to a client
    two weeks into an engagement, not a defence to an examiner. A client does not read a
    methodology essay before the first result, does not know what a ledger code is, and does
    not need the mechanism unless they ask. So: no internal codes, no argument the screen
    has not earned yet, plain nouns, and the model names written out. The substance stays —
    the floor, the refusal, the evidence, the table where this engine loses — in fewer words.
    """
    with st.sidebar:
        # "Book Recommender" rather than the pinned copy's "What this is" (the project owner,
        # 09.08.2026): the sidebar's first line is the only place the thing gets named, and
        # a section label is not a name.
        with st.container(key="app-name"):
            st.markdown("### Book Recommender")
        # M15.6's provenance line, moved here from under the page title (review,09.08.).
        # Its rule is unchanged and is the reason it is a function rather than a string:
        # exactly one place in the app states the corpus size, and it is counted off the
        # assets at runtime.
        #
        # Body text, not a caption (review,09.08.): the two differed only in size, 13.1px
        # against 15px, and at caption size it read as a footnote to the name above it
        # rather than as the first thing the app tells you.
        #
        # The pinned copy's opening line — "One book in, similar books out. No login and no
        # reading history" — is cut with it. The page title says the first half and the
        # absence of any login field says the second.
        st.markdown(corpus_line(engine))
        st.markdown("### How it works")
        # "1.1 million ratings" was written out here, four lines under the corpus line that
        # counts the same fact at runtime and prints 1,143,125 (M17.10). Two statements of
        # one number, one of them hard-coded, is how a demo ends up contradicting itself
        # after a rebuild — the corpus line already survived exactly that when the work count
        # moved on the L64 re-base. The hard-coded one goes; the sentence reads better
        # without it anyway, since "those ratings" points at the line above.
        st.markdown(
            "Those reading patterns are compressed into a short profile per book. Books "
            "whose profiles point in the same direction come back as similar. Editions of "
            "the same title are merged before anything is computed, so the results are "
            "books rather than reprints."
        )
        st.markdown("### Where it stops")
        st.markdown(
            f"Below {engine.assets.anchor_floor} readers the demo declines to answer. The "
            "model would still return ten titles and they would be noise. Every result "
            "names the readers behind it, so you can see how much it rests on."
        )
        st.markdown("### The engine")
        # "second", not "third" (review,09.08.): the sentence said third and the table
        # underneath it said second, on the same screen. The ordinal is a leftover from M13,
        # where it was read off the ISBN-level ledger table — that one carries the work-level
        # item-item row as a pointer *next to* the ISBN-level one, so two item-item rows sat
        # above ALS (0.0644 and 0.0546 against 0.0451). Under the published work-level table
        # (L52-L57) there is one item-item row and ALS is second. No measurement moves; the
        # ordinal was never re-read after the re-base. Same correction in the two docstrings,
        # RESULTS.md and model_selection.md.
        #
        # The closing sentence's "asks the second" becomes "asks the latter" with it: the
        # pinned copy could carry two senses of "second" once the ordinal is right, and the
        # one that matters is which of the two questions the demo asks.
        #
        # **The superlative goes too (M20, 09.08.).** "produces the best 'books like this
        # one' lists of the six approaches we tried" was the pinned copy's strongest claim
        # and the only one on this screen with no number behind it. M20 measured it: on the
        # item query ALS and item-item are not distinguishable (L80, 169/155, p = 0.47).
        # The replacement says what was measured and keeps the reason the engine is here.
        # This is a deviation from M15.5's pinned copy, recorded with M20,
        # and it is open to reverse — but a superlative that a run of our own contradicts
        # is the one thing this demo cannot leave on screen.
        st.markdown(
            "This demo runs on matrix factorization. It comes second on the accuracy table "
            "below. We also measured the question this demo actually asks — one book in, "
            "similar books out — and on that one it is level with the approach that comes "
            "first. Those are two different questions, and a demo like this one asks the "
            "latter."
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
        # The sidebar ends on the table's own footnote. Three closing captions used to
        # follow — dead cover images, offline/cold start, and "every number here is
        # measured" — and all three were cut on seeing them run. The last of those was
        # M15's pinned decision 1; the deviation is recorded with the milestone.
        #
        # It reads better as a cut: each line answered a question no reader had
        # asked yet, and the ledger claim in particular is worth more said out loud in
        # response to "how do you know that" than printed pre-emptively where it looks
        # defensive.


def main() -> None:
    st.set_page_config(page_title="Book recommender — demo", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    try:
        engine, _ = get_engine()
    except (FileNotFoundError, RuntimeError) as error:
        st.error(f"{error}\n\nRun `python scripts/build_app_assets.py` first.")
        return

    sidebar(engine)

    st.title("Name a book, get books like it")

    # Keyed so the CSS above can pull the reader counts up against their buttons without
    # touching every other caption in the app.
    with st.container(key="anchor-row"):
        columns = st.columns(len(ANCHORS))
        for column, (label, isbn) in zip(columns, ANCHORS.items(), strict=True):
            if column.button(label, use_container_width=True):
                st.session_state["query"] = label
                st.session_state["pinned_isbn"] = isbn
            # The reader count belongs on the button. Without it a thinly read anchor looks
            # like a broken app; with it, the calibration story (L63) is visible in the
            # product rather than only asserted on a slide.
            column.caption(f"{engine.describe(isbn).readers:,} readers")

    # M15.7 offered "harry potter stein" as a clickable example here; it was cut on
    # 09.08. The query is still the one worth showing — it takes two serving rules to
    # resolve (L62) — but it is a thing to *say* while typing it live, not a control the
    # screen has to carry. The anchor buttons already cover "I do not want to type".
    #
    # Anything that writes `st.session_state["query"]` must still run **before** this
    # widget is created: Streamlit raises otherwise, and the example button crashed the
    # app on click for exactly that reason before it was moved above the field.
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
    # A button pins its work id, so there is nothing to look up (M17.5). `find()` used to run
    # here unconditionally and have its result discarded three lines later, which is the
    # sentence-encoder call — the most expensive part of a query — and it made "Answered in
    # N ms" mean one thing for a button and another for a typed title. One label, one
    # quantity.
    matches = [] if pinned else engine.find(query, k=5)
    if not pinned and not matches:
        st.warning("Nothing found. Try a title and an author.")
        return

    chosen = pinned or matches[0].isbn
    # The picker is reserved rather than conditional, so the result does not jump down the
    # screen the moment a second candidate exists. `find` now returns only candidates within
    # PICKER_MARGIN of the best match (M17.4), so "one match" is the common case and the
    # picker mostly stays empty.
    picker = st.container()
    if len(matches) > 1:
        with picker:
            labels = {f"{m.title} — {m.author}": m.isbn for m in matches}
            chosen = labels[st.radio("Did you mean", list(labels))]

    book = engine.describe(chosen)
    st.subheader(book.title)
    # No `series` (M17.6): the field is the title's parenthetical, so this line printed
    # "Harry Potter (Paperback)" as though it were the series, and *Dune* carries
    # "Remembering Tomorrow". A real series entity is a data-layer project, on the roadmap.
    # The year is filtered for the same reason as in `render_row` — L3's missing-year zero.
    anchor_parts = [book.author, book.year if book.year not in ("0", "") else "", f"{book.readers:,} readers"]
    st.caption(" · ".join(part for part in anchor_parts if part))

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
    # The legend says what the list is sorted by, and it is the first thing it says (M17.2).
    # Before M17 the sort key was nowhere on screen and the only quantity that *was* on
    # screen — shared readers — visibly contradicted the order, which reads as a bug rather
    # than as the measured finding it is.
    #
    # "compared with the top of this list" is the bar's scaling rule, said once. It is not
    # decoration: an absolute 0-1 axis would draw a whole legitimate list as stubs, because
    # a cosine is not comparable across anchors. The number next to each bar is the absolute
    # value, which within one list is exactly the sort key.
    st.markdown(
        f'<div class="legend">Sorted by similarity — the bar shows each book '
        f"compared with the top of this list, and the number is its actual score. "
        f"The readers behind it follow. “Thin evidence” marks a book fewer than "
        f"{THIN_EVIDENCE_SHARE:.0%} of this book's readers also read.</div>",
        unsafe_allow_html=True,
    )

    widths = bar_widths(suggestions)
    for index, (suggestion, width) in enumerate(zip(suggestions, widths, strict=True)):
        st.markdown(render_row(index + 1, suggestion, width), unsafe_allow_html=True)

    st.caption(f"Answered in {elapsed * 1000:.0f} ms.")


main()
