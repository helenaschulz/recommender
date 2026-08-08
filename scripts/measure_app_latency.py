"""Measure what a presenter actually waits for: cold start, then per-query time.

    python scripts/measure_app_latency.py

The Arbeitsplan's DoD is "cold start < 30 s, query < 1 s", so both halves are measured
rather than estimated, and the cold start is measured **in a fresh interpreter** — timing
it in a process that has already imported numpy, torch and the assets would measure
nothing at all.

Cold start is reported in two parts because they are paid in parallel by a human but in
sequence by the machine, and only their sum is honest:

1. **Server ready** — ``streamlit run`` from process launch until ``/_stcore/health``
   answers. This is Streamlit's own start-up and has nothing to do with the model.
2. **First answer** — in a *new* Python process: load the assets, build the engine, load
   the sentence encoder, and answer one real query end to end. This is where the model
   assets and the encoder are paid for.

Per-query time is measured warm and reported as a median over the three gallery anchors,
because that is the number a presenter experiences after the first click.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from recommender.data import project_root
from recommender.gallery import ANCHORS

#: Run in a subprocess with a clean interpreter, so imports are part of the measurement.
FIRST_ANSWER = """
import json, time
started = time.perf_counter()
from recommender.demo import DemoEngine, load_assets
imported = time.perf_counter()
engine = DemoEngine(load_assets())
loaded = time.perf_counter()
book = engine.find("harry potter stein", k=1)[0]
found = time.perf_counter()
rows = engine.similar(book.isbn, k=10)
print(json.dumps({
    "import_s": imported - started,
    "load_assets_s": loaded - imported,
    "first_lookup_s": found - loaded,
    "first_similar_s": time.perf_counter() - found,
    "total_s": time.perf_counter() - started,
    "resolved": book.title,
    "n_suggestions": len(rows),
}))
"""


def server_ready_seconds(port: int, timeout: float = 120.0) -> tuple[float, str]:
    """Launch the app headless and time it to a healthy HTTP response."""
    root = project_root()
    started = time.perf_counter()
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable, "-m", "streamlit", "run", str(root / "app" / "main.py"),
            "--server.port", str(port), "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        while time.perf_counter() - started < timeout:
            if process.poll() is not None:
                return float("nan"), (process.stdout.read() if process.stdout else "")[-2000:]
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        return time.perf_counter() - started, "ok"
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.2)
        return float("nan"), "timed out"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-server", action="store_true")
    args = parser.parse_args(argv)

    root = project_root()
    ready, note = (float("nan"), "skipped") if args.skip_server else server_ready_seconds(args.port)
    print(f"1 · streamlit server ready : {ready:6.1f}s  ({note})", flush=True)

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", FIRST_ANSWER], cwd=root, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        print(completed.stdout[-2000:] + completed.stderr[-2000:])
        raise SystemExit("first-answer measurement failed -- have the assets been built?")
    cold = json.loads(completed.stdout.strip().splitlines()[-1])
    print(
        f"2 · first answer, fresh process: {cold['total_s']:6.1f}s"
        f"  (import {cold['import_s']:.1f}s, assets {cold['load_assets_s']:.1f}s,"
        f" lookup {cold['first_lookup_s']:.1f}s, similar {cold['first_similar_s']:.1f}s)"
    )
    print(f"    resolved 'harry potter stein' -> {cold['resolved']!r}, {cold['n_suggestions']} suggestions")
    total_cold = (0.0 if args.skip_server else ready) + cold["total_s"]
    print(f"\nCOLD START (1 + 2)         : {total_cold:6.1f}s   target < 30s")

    # Warm timings, in this process, after everything is paged in.
    from recommender.demo import DemoEngine, load_assets

    engine = DemoEngine(load_assets())
    for isbn in ANCHORS:
        engine.similar(isbn, k=10)  # page in the factor array before timing

    lookups: list[float] = []
    similars: list[float] = []
    for _ in range(args.repeats):
        for isbn, label in ANCHORS.items():
            started = time.perf_counter()
            engine.find(label, k=5)
            lookups.append(time.perf_counter() - started)
            started = time.perf_counter()
            engine.similar(isbn, k=10)
            similars.append(time.perf_counter() - started)

    print(
        f"WARM QUERY (median of {len(similars)}) : lookup {statistics.median(lookups) * 1000:5.0f} ms"
        f" + similar {statistics.median(similars) * 1000:5.0f} ms"
        f" = {(statistics.median(lookups) + statistics.median(similars)) * 1000:5.0f} ms   target < 1000 ms"
    )
    print(f"assets on disk: {sum(p.stat().st_size for p in Path(root / 'artifacts' / 'app').iterdir()) / 1e6:,.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
