"""Screenshot the three anchor flows, so the demo exists as evidence even if it will not run.

    python scripts/capture_app_screenshots.py

Writes ``docs/img/app_<anchor>.png``. This is the fallback if both the live demo and the
screen recording fail on the day, and it is the evidence a reviewer can see without
installing anything — a PR that claims "the app works" and shows nothing is a claim, not a
demo.

Reproducible on purpose rather than a hand-taken screenshot: the script starts the app on a
private port, drives the same three buttons a presenter would click, waits for the results
to actually render, and shuts the server down. Re-running it after a change re-shoots the
evidence instead of leaving a stale picture in the repo.

Requires ``playwright`` and its Chromium (``python -m playwright install chromium``). Not a
runtime dependency of the app — only of this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request

from recommender.data import project_root
from recommender.gallery import DEMO_BUTTONS

#: Button label -> output slug, **derived** from the app's own button list rather than
#: retyped. The first version duplicated the labels, and when M14.8 changed them this
#: script sat waiting 30 seconds for a button that no longer existed.
ANCHORS = {label: label.lower().replace(" ", "_").replace("'", "") for label in DEMO_BUTTONS}
LOOKUP_QUERY = "harry potter stein"


def wait_for_health(port: int, timeout: float = 120.0) -> None:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.3)
    raise SystemExit(f"the app did not become healthy on port {port}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8597)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1100)
    args = parser.parse_args(argv)

    from playwright.sync_api import sync_playwright

    root = project_root()
    out = root / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable, "-m", "streamlit", "run", str(root / "app" / "main.py"),
            "--server.port", str(args.port), "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_health(args.port)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.goto(f"http://127.0.0.1:{args.port}", wait_until="networkidle")
            # The first render loads ~900 MB of assets behind st.cache_resource.
            page.wait_for_selector("text=Paste a book", timeout=120_000)

            for label, slug in ANCHORS.items():
                page.get_by_role("button", name=label, exact=True).click()
                page.wait_for_selector("text=Because you liked", timeout=120_000)
                page.wait_for_timeout(1_200)  # let the lazy cover images settle or fail
                path = out / f"app_{slug}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"wrote {path.relative_to(root)}")

            # And the input path, which is the part a panel will actually poke at.
            box = page.get_by_placeholder("harry potter stein")
            box.fill(LOOKUP_QUERY)
            box.press("Enter")
            page.wait_for_selector("text=Because you liked", timeout=120_000)
            page.wait_for_timeout(1_200)
            path = out / "app_lookup.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"wrote {path.relative_to(root)}  (free-text query {LOOKUP_QUERY!r})")

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
