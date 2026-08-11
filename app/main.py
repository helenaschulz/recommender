"""The demo: name a book, get similar books, each with the evidence behind it.

    streamlit run app/main.py

Deliberately thin. Every rule that could be *wrong* lives in :mod:`recommender.demo`
(ranking) and :mod:`recommender.display` (presentation), both tested offline; this file is
layout and copy. Run ``python scripts/build_app_assets.py`` once first — the app never fits
a model, never reads ``data/`` and never touches the network.

**Two engines since M23.10, and A is the default.** The picker in the sidebar switches between
**A**, ALS item factors over the work-keyed matrix, which is what the rehearsed demo runs, and
**B**, the shrunk cosine over shared readers. Both carry the same support floors — L34 for
candidates, L65 for anchors — so the control moves the engine and nothing else. One constant,
:data:`SHOW_ENGINE_PICKER`, removes it and leaves A exactly as it was.

**What the click shows.** The same anchor, from the same 2,508 askable works, answered by a
calibrated similarity instead of an anti-calibrated one. L87 measured the difference in the
evidence behind a shown slot — 1.7% thin slots against 18.2% at this floor — and
``docs/anchor_set_audit.md`` is twenty anchors of it read by hand: A's median slot rests on 15
shared readers and B's on 46. What the click does **not** show is the floor. That is a separate
finding with a separate demonstration and no picker in it: type *The Kite Runner* and the demo
declines, because 39 readers is under 50, and it declines identically on both settings.

**What M20 changed about the reason ALS is the default** (L80–L82). This docstring used to say
ALS "has the best item-to-item neighbourhoods in the project", carried from L34 — three anchors
read by eye in M8, before the re-base that made item-item's and ALS's *Harry Potter*
neighbourhoods identical. Measured on 13,580 anchors, **ALS and item-item are not
distinguishable on the item query** (169/155, p = 0.47), and ALS's measured edge lives
entirely below 5 readers of anchor support — a band this app *declines to serve* (L65, L81).
What the measurement does vindicate is the **floor**, not the factorization: without it ALS
scores below the popularity baseline (L82). A is the default because it is the rehearsed one,
not because it won anything, and the sidebar says so.

**The switcher's cost was measured before it was built, and it is not what M13.2 assumed.**
M13.2 dropped one on the grounds that each engine needs its own similarity artefact and would
cost the cold start this app is built around. With the anchor floor at 50 only 2,508 works are
reachable as anchors at all, so B's artefact is a precomputed top-10 table of 25,080 rows and
0.22 MB (L91), written *beside* the shipped assets and never over them. It is also the Part 3
Gold-table serving pattern, built instead of sketched.

**Ledger codes appear in this file and never on the screen** (M15, pinned decision 1). They
are precise to us and are internal jargon to a visitor, and jargon in a demo reads as
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
in the write-up.

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
    from recommender.engines import CONFIGURATIONS, DEFAULT_CONFIGURATION
    from recommender.gallery import DEMO_BUTTONS as ANCHORS  # M14.8 — one source, see its docstring
    from recommender.models.embeddings import sentence_transformer_encoder
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
  /* Full width, decided on seeing it run (08.08.2026), reversing M15.2's
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

  /* The model picker. Both options are a model name plus a three-word gloss, so the longer
     one wraps in a 21rem sidebar — and Streamlit's row spacing assumes one line, which left
     the wrapped second option running into the first. Give the rows a gap and let the label
     text sit against the top of its button rather than centred on two lines. Scoped to the
     keyed container so every other radio in the app keeps Streamlit's own spacing. */
  .st-key-engine-picker [role="radiogroup"] {gap: 0.5rem;}
  .st-key-engine-picker [role="radiogroup"] > label {align-items: flex-start; line-height: 1.35;}
  .st-key-engine-picker [role="radiogroup"] > label > div:first-child {margin-top: 0.15rem;}

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


#: **The one flag** (M23 decision 2). Set it to ``False`` and the picker is gone, the sidebar
#: is what it was, and the demo runs :data:`~recommender.engines.DEFAULT_CONFIGURATION` — which
#: is A, the rehearsed engine, byte for byte. It is a constant rather than an environment
#: variable on purpose: flipping it is a diff, and shortly before a live demo the thing worth
#: having is a record of what was changed, not a shell that remembers.
SHOW_ENGINE_PICKER = True


@st.cache_resource(show_spinner="Loading the model assets…")
def get_assets():
    """The arrays, loaded once and shared by every configuration.

    Separate from :func:`get_engine` since M23.10: two configurations must not mean two copies
    of a 890 MB asset set, and — more to the point — must not mean two loads of the sentence
    encoder, which is 9.5 of L69's 10.6-second cold start.
    """
    return load_assets()


@st.cache_resource(show_spinner=False)
def get_encoder(model_name: str):
    """One encoder for the whole app. Cheap to build: the model itself loads on first use.

    ``sentence_transformer_encoder`` holds its model in a closure and only instantiates it when
    something is actually encoded, so calling this during engine construction costs nothing and
    keeps the encoder **lazy** — which is what stops a configuration switch from paying for a
    transformer load a second time.
    """
    return sentence_transformer_encoder(model_name)


@st.cache_resource(show_spinner="Loading the model assets…")
def get_engine(configuration: str) -> tuple[DemoEngine, float]:
    started = time.perf_counter()
    assets = get_assets()
    engine = DemoEngine(assets, configuration=configuration, encoder=get_encoder(assets.encoder_model))
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


def chosen_configuration() -> str:
    """Which configuration to build, read **before** the picker widget is drawn.

    The engine has to exist before the sidebar can be written — the corpus line counts off the
    assets and the engine section names the engine — but the picker belongs *in* the sidebar,
    next to the accuracy table it changes the reading of. Streamlit's own answer to that order
    is session state: the widget's value from the previous run is already there when this run
    starts, and a click reruns the script with the new value in place before a line renders. So
    this is not a workaround, it is the ordering Streamlit is built for.

    :data:`SHOW_ENGINE_PICKER` short-circuits it, and that is M23 decision 2's one flag: with it
    off, this returns the rehearsed default and no widget is ever created to disagree with.
    """
    if not SHOW_ENGINE_PICKER:
        return DEFAULT_CONFIGURATION
    return str(st.session_state.get("configuration", DEFAULT_CONFIGURATION))


def sidebar(engine: DemoEngine) -> None:
    """The pinned M15.5 copy.

    **Register, which is the decision behind every line.** This is a product surface, not a
    methods paper. A visitor does not read an essay before the first result, does not know
    what a ledger code is, and does not need the mechanism unless they ask. So: no internal codes, no argument the screen
    has not earned yet, plain nouns, and the model names written out. The substance stays —
    the floor, the refusal, the evidence, the table where this engine loses — in fewer words.
    """
    with st.sidebar:
        # "Book Recommender" rather than the pinned copy's "What this is" (copy read,
        # 09.08.2026): the sidebar's first line is the only place the thing gets named, and
        # a section label is not a name.
        with st.container(key="app-name"):
            st.markdown("### Book Recommender")
        # M15.6's provenance line, moved here from under the page title (09.08.).
        # Its rule is unchanged and is the reason it is a function rather than a string:
        # exactly one place in the app states the corpus size, and it is counted off the
        # assets at runtime.
        #
        # Body text, not a caption (09.08.): the two differed only in size, 13.1px
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
        #
        # **The mechanism sentence left this block in M23.10** and now lives under the picker,
        # because there are two mechanisms and the copy has to say which one is running. What
        # stays here is what is true of both settings — otherwise the sidebar would claim
        # "profiles point in the same direction" while the shared-reader engine was answering.
        #
        # It was first *reworded* rather than cut, into "Those reading patterns are what one
        # book is compared with another by" — a sentence with a stranded preposition that
        # nobody would write, saying something the picker three blocks down now says properly.
        # Cut on the copy read, 10.08.2026. The rule it illustrates: when a block loses
        # its job, delete it; a rewrite that keeps the slot is how filler survives a review.
        # The evidence sentence moved up here from "Where it stops" (copy read, 10.08.2026).
        # It describes how the app works, not where it stops, and under the old heading it
        # was the third sentence of a block that was supposed to state one rule.
        st.markdown(
            "Editions of the same title are merged before anything is computed, so the "
            "results are books rather than reprints. Every result names the readers behind "
            "it, so you can see how much it rests on."
        )
        # **"The anchor floor" (10.08.2026), replacing "Where it stops".** The copy read
        # asked for the functionality named rather than described, and this is the name the
        # ledger has used since L65 — so the screen, the write-up and the ledger say one
        # thing. It is the first house term this app has ever printed, which is why the
        # sentence defines it before it uses it: on this screen the anchor is simply the book
        # you typed, and six words of definition are cheaper than a visitor silently not
        # knowing what a floor is under.
        #
        # The number is read off the assets rather than typed, the same rule as `corpus_line`:
        # the floor has already moved once (M14.2 split it from the candidate floor) and a
        # hard-coded 50 here would have been stale within the day.
        st.markdown("### The anchor floor")
        st.markdown(
            f"The book you name is the anchor, and it needs {engine.assets.anchor_floor} "
            "readers in this dataset before the demo will answer for it. Below that the "
            "model would still return ten titles and they would be noise, so it declines "
            "instead."
        )
        st.markdown("### The engine")
        # "second", not "third" (09.08.): the sentence said third and the table
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
        # This is a deviation from M15.5's pinned copy, recorded with M20 and reversible —
        # but a superlative that a run of our own contradicts is the one thing this demo
        # cannot leave on screen.
        #
        # **M23.10 made this copy switch with the picker**, because under the other setting
        # every clause of it was false: the demo would be running item-based collaborative
        # filtering, which comes *first* on the table below rather than second. Two sentences
        # of copy that contradict the table on the same screen is the M17.8 failure mode, and
        # this screen has had it twice.
        key = engine.configuration.key if engine.configuration else DEFAULT_CONFIGURATION
        if key == "A":
            # The dash pair became brackets on the copy read (10.08.2026), so that the
            # two engine paragraphs punctuate an aside the same way. The words are M15.5's
            # pinned copy as M20 corrected them and are otherwise untouched.
            st.markdown(
                "This demo runs on matrix factorization. It comes second on the accuracy "
                "table below. We also measured the question this demo actually asks (one "
                "book in, similar books out) and on that one it is level with the approach "
                "that comes first. Those are two different questions, and a demo like this "
                "one asks the latter."
            )
        else:
            # Deliberately not "and it is better". On the item query the two are level on one
            # draw and this one is ahead on the other (L80, L81), which is a reason to be able
            # to show both and not a reason to claim a winner on a sidebar.
            #
            # **Rewritten on the copy read, 10.08.2026**, and the three things it was
            # reacting to are worth naming because they are the tells: "is *now* counting"
            # narrated a state where A simply says what it runs on; "the table is *not the
            # reason* to prefer either" defined by negation; and "*What separates them is* …"
            # is a cleft sentence, which was the most conspicuous machine-written construction
            # on the screen. Same four sentences, same claims, none of the three.
            st.markdown(
                "This demo counts shared readers. It comes first on the accuracy table "
                "below. On the question this demo actually asks (one book in, similar books "
                "out) the two settings are level, so the table does not decide between them. "
                "They differ in how much reading is behind each result, and that number sits "
                "beside every row."
            )
        # The table keeps its published form and numbers (L52-L57); only the labels change,
        # from house abbreviations to written-out names. It stays expanded, because showing
        # the table where this engine loses is the point.
        #
        # Which row carries "this demo" follows the picker, for the reason above: the marker
        # is a statement about what is running, and a fixed one would be wrong half the time.
        #
        # **The two switchable rows take their names from the configuration**
        # (10.08.2026) rather than from string literals here. They were literals for one
        # afternoon and that was already enough for the screen to disagree with itself: the
        # picker said "Shared readers" while the row it marked said "Item-based collaborative
        # filtering". One name per thing, from one place.
        def row(config_key: str, score: str) -> str:
            name = CONFIGURATIONS[config_key].short_label
            return (
                f"| **{name} · this demo** | **{score}** |"
                if config_key == key
                else f"| {name} | {score} |"
            )

        st.markdown(
            "\n".join(
                [
                    "| Approach | Hit rate @10 |",
                    "|---|---:|",
                    row("B", "0.0644"),
                    row("A", "0.0545"),
                    "| Item-based CF, explicit ratings only | 0.0486 |",
                    "| Content-based, TF-IDF on title and author | 0.0405 |",
                    "| Popularity baseline | 0.0155 |",
                    "| Content-based, multilingual embeddings | 0.0141 |",
                ]
            )
        )
        st.caption(
            "Hit rate @10: how often a reader's held-out book turns up in their top ten. "
            "Same data split for all six."
        )
        # The switch sits under the table rather than above it, because the table is what it
        # changes the reading of: you see where the running engine places, then you change it
        # and watch both the marker and the list move. `key="configuration"` is what
        # `chosen_configuration` read at the top of this run — see there for the ordering.
        if SHOW_ENGINE_PICKER:
            with st.container(key="engine-picker"):
                # "The model", not "What counts as similar" (10.08.2026). The heading
                # asked *how* and the options now answer *which*, and a question whose answer
                # is a different question reads as a mismatch. It also sits inside the sidebar
                # section already called "The engine", so the two agree about what is on offer.
                st.markdown("**The model**")
                keys = list(CONFIGURATIONS)
                st.radio(
                    "The model",
                    keys,
                    key="configuration",
                    index=keys.index(DEFAULT_CONFIGURATION),
                    format_func=lambda option: CONFIGURATIONS[option].label,
                    label_visibility="collapsed",
                )
                st.caption(CONFIGURATIONS[key].blurb)
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
        engine, _ = get_engine(chosen_configuration())
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
            # product rather than only asserted.
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
    #
    # **M23.10 swapped `find` for `resolve` here, and decision 5b is the whole reason.**
    # `resolve` returns the identical shortlist — same call, same rules, so no query resolves
    # anywhere new — plus the books the anchor floor removed from it. Those were dropped
    # silently from M13 until now, which is how *The Kite Runner* could vanish from a demo
    # about book recommendations without the screen ever saying so.
    resolution = engine.resolve(query, k=5) if not pinned else None
    matches = [] if resolution is None else resolution.matches
    below_floor = [] if resolution is None else resolution.below_floor
    if not pinned and not matches and not below_floor:
        st.warning("Nothing found. Try a title and an author.")
        return

    chosen = pinned or (matches[0].isbn if matches else below_floor[0].isbn)
    # The picker is reserved rather than conditional, so the result does not jump down the
    # screen the moment a second candidate exists. `find` returns only candidates within
    # PICKER_MARGIN of the best match (M17.4), so "one match" is the common case and the
    # picker mostly stays empty.
    #
    # The books under the floor go **after** the answerable ones and carry their reader count,
    # so the first option is still what the query resolved to and the default selection cannot
    # move. Picking one is what produces the refusal below — the app names the book, names the
    # number, and names the rule, rather than quietly answering about something else.
    picker = st.container()
    options = {f"{m.title} — {m.author}": m.isbn for m in matches}
    options.update(
        {
            # "below the N we ask for", the same words the refusal one click later uses.
            # It read "below the N this demo answers" for an afternoon, which is one rule
            # stated two ways a second apart (copy read, 10.08.2026).
            f"{b.title} — {b.author}  ·  {b.readers:,} readers, "
            f"below the {engine.assets.anchor_floor} we ask for": b.isbn
            for b in below_floor
        }
    )
    if len(options) > 1:
        with picker:
            chosen = options[st.radio("Did you mean", list(options))]

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
    #
    # **The number now says which similarity it is (M23 decision 6).** A's is a cosine between
    # learned profiles and B's is a shrunk cosine over shared readers; the two are on different
    # scales and mean different things, so a screen that prints one of them unlabelled invites
    # a reader who switches to compare 0.51 with 0.11 and conclude the wrong thing. The label
    # is the whole fix, and it is deliberately not accompanied by a "these are not comparable"
    # caveat — that answers a question no reader has asked yet (the M15.5 register
    # rule), and it belongs in the accompanying write-up.
    #
    # A colon rather than a dash, because what follows is a definition of the word before it
    # rather than an aside (copy read, 10.08.2026).
    st.markdown(
        f'<div class="legend">Sorted by similarity: {html.escape(engine.score_label)}. '
        f"The bar shows each book compared with the top of this list, and the number is its "
        f"actual score. The readers behind it follow. “Thin evidence” marks a book fewer "
        f"than {THIN_EVIDENCE_SHARE:.0%} of this book's readers also read.</div>",
        unsafe_allow_html=True,
    )

    widths = bar_widths(suggestions)
    for index, (suggestion, width) in enumerate(zip(suggestions, widths, strict=True)):
        st.markdown(render_row(index + 1, suggestion, width), unsafe_allow_html=True)

    st.caption(f"Answered in {elapsed * 1000:.0f} ms.")


main()
