"""The demo: paste a book, get ten similar books, each with one grounded reason.

    streamlit run app/main.py

Deliberately thin. Every rule lives in :mod:`recommender.demo`, which is tested offline;
this file is widgets and layout, so there is nothing here that can be wrong in a way a test
would not catch. Run ``python scripts/build_app_assets.py`` once first — the app never
fits a model, never reads ``data/`` and never touches the network.

**One engine, on purpose.** The similar-items engine is ALS item factors over the
**work-keyed** matrix, with the support floor from ledger L34. ALS places *third of six* on
HitRate@10 in the published table (L55) and has the best item-to-item neighbourhoods in the
whole project. That the offline metric and the product surface disagree is the finding, not
an inconsistency — so the table where ALS loses is shown in the sidebar rather than hidden.
A switcher across the other four models was scoped and dropped: each would need its own
similarity artefact, which buys a dropdown and costs the cold start this app is built
around.
"""

from __future__ import annotations

import time

import streamlit as st

from recommender.demo import DemoEngine, load_assets
from recommender.gallery import DEMO_BUTTONS as ANCHORS  # M14.8 — one source, see its docstring

#: The dataset ships cover URLs, and every one of them is a 2004 ``images.amazon.com`` link
#: that now answers **403**. Rendering them gives a page full of broken-image icons, which
#: reads as a bug rather than as a dead third party — so the app draws its own placeholder
#: from the title instead. Deterministic, offline, and honest about being a placeholder.
SPINE_COLOURS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860", "#da8bc3"]


def spine(title: str, height: int = 74) -> str:
    """A tiny coloured block carrying the title's initial. No network, no broken icons."""
    initial = next((character for character in title.upper() if character.isalnum()), "?")
    colour = SPINE_COLOURS[sum(map(ord, title)) % len(SPINE_COLOURS)]
    return (
        f"<div style='width:{height * 2 // 3}px;height:{height}px;background:{colour};"
        "border-radius:3px;display:flex;align-items:center;justify-content:center;"
        f"color:white;font-size:{height // 3}px;font-weight:600;'>{initial}</div>"
    )


@st.cache_resource(show_spinner="Loading the model assets…")
def get_engine() -> tuple[DemoEngine, float]:
    started = time.perf_counter()
    engine = DemoEngine(load_assets())
    return engine, time.perf_counter() - started


def main() -> None:
    st.set_page_config(page_title="Book recommender — demo", page_icon="📚", layout="wide")
    st.title("📚 Paste a book, get books like it")

    try:
        engine, cold_start = get_engine()
    except (FileNotFoundError, RuntimeError) as error:
        st.error(f"{error}\n\nRun `python scripts/build_app_assets.py` first.")
        return

    with st.sidebar:
        st.header("What this is")
        st.markdown(
            "An **item-to-item** surface: one book in, similar books out, with no user "
            "identity at query time — which is exactly the question the brief asks.\n\n"
            "**Engine:** ALS item factors over the **work-keyed** matrix (editions merged "
            "before fitting, **L49**), similarity = cosine.\n\n"
            "**Two support floors, not one** (**L63**, **L65**): a candidate needs 20 "
            "interactions (**L34**), an *anchor* needs 50. The similarity score is not "
            "comparable across anchors — below 30 readers, 72% of the books it would show "
            "share fewer than five readers with the anchor, at the same cosine. So the app "
            "shows counts and refuses anchors it cannot support, and never prints the "
            "similarity next to a number that *is* comparable.\n\n"
            "**Why ALS,** when it is only third of six on HitRate@10 (**L55**)? Because "
            "accuracy on held-out *user histories* and quality of *item neighbourhoods* "
            "are different questions, and this app asks the second one. ALS wins it "
            "clearly. The table where it loses is below — showing both is the point."
        )
        st.markdown(
            "| Model | HitRate@10 |\n|---|---:|\n"
            "| item-item CF | **0.0644** |\n"
            "| **ALS ← this app** | **0.0545** |\n"
            "| item-item, explicit-only | 0.0486 |\n"
            "| content TF-IDF | 0.0405 |\n"
            "| popularity | 0.0155 |\n"
            "| content embeddings | 0.0141 |"
        )
        st.caption(
            "Cover images are omitted on purpose: the dataset's are 2004 "
            "`images.amazon.com` links and every one of them now answers 403."
        )
        st.caption(f"Cold start {cold_start:.1f}s · assets loaded once, no network, no fitting.")

    st.caption("Type a title however you remember it — the lookup is embedding-based, not string matching.")
    # One button label runs to two lines and the others to one, which drops that button's
    # reader count half a line below its neighbours'. A fixed height keeps the row square:
    # the buttons are the first thing on screen and a ragged row reads as carelessness.
    st.markdown(
        "<style>div.stButton > button {height: 3.4rem; white-space: normal; line-height: 1.15;}</style>",
        unsafe_allow_html=True,
    )
    columns = st.columns(len(ANCHORS))
    for column, (label, isbn) in zip(columns, ANCHORS.items(), strict=True):
        if column.button(label, use_container_width=True):
            st.session_state["query"] = label
            st.session_state["pinned_isbn"] = isbn
        # The reader count belongs on the button, not in a footnote. Without it a thinly
        # read anchor looks like a broken app; with it, L63's calibration story is visible
        # in the product instead of only asserted on a slide.
        column.caption(f"{engine.describe(isbn).readers:,} readers")

    query = st.text_input("Book", key="query", placeholder="harry potter stein")
    if not query:
        return

    started = time.perf_counter()
    pinned = st.session_state.pop("pinned_isbn", None)
    matches = engine.find(query, k=5)
    if not matches:
        st.warning("Nothing found. Try a title and an author.")
        return

    chosen = pinned or matches[0].isbn
    if len(matches) > 1 and not pinned:
        labels = {f"{m.title} — {m.author}": m.isbn for m in matches}
        chosen = labels[st.radio("Did you mean", list(labels), horizontal=False)]

    book = engine.describe(chosen)
    left, right = st.columns([1, 9])
    with left:
        st.markdown(spine(book.title, height=110), unsafe_allow_html=True)
    with right:
        st.subheader(book.title)
        st.write(f"**{book.author}** · {book.year}" + (f" · {book.series}" if book.series else ""))
        st.caption(f"{book.readers:,} readers in the dataset")

    st.divider()
    suggestions = engine.similar(chosen, k=10)
    elapsed = time.perf_counter() - started
    if not suggestions:
        st.info(
            f"No neighbourhood for this book: {book.readers:,} interactions, below the "
            f"{engine.assets.anchor_floor} an anchor needs. Under that line the model's "
            "nearest neighbours are noise wearing a confident similarity score (ledger "
            "L34, L63). An honest empty answer beats a fabricated list."
        )
        return

    st.subheader(f"Because you liked *{book.title}*")
    for rank, suggestion in enumerate(suggestions, start=1):
        left, right = st.columns([1, 14])
        with left:
            st.markdown(spine(suggestion.title), unsafe_allow_html=True)
        with right:
            st.markdown(f"**{rank}. {suggestion.title}**  \n{suggestion.author} · {suggestion.year}")
            st.caption(suggestion.reason)

    st.caption(f"Query answered in {elapsed * 1000:.0f} ms.")


main()
