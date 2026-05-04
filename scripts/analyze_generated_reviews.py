"""
Summarize archetype, model, and rating distributions for generated review JSONL.
Writes a PNG chart when matplotlib is available.

Usage:
  python scripts/analyze_generated_reviews.py
  python scripts/analyze_generated_reviews.py --input data/output/ai_reviews_realistic.jsonl
  python scripts/analyze_generated_reviews.py --output figures/my_dist.png
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = _PROJECT_ROOT / "data/output/ai_reviews_realistic.jsonl"
DEFAULT_CHART = _PROJECT_ROOT / "data/output/review_distributions.png"
TOP_ARCHETYPES_CHART = 18


def load_reviews(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skip invalid JSON at line {line_num}: {e}", file=sys.stderr)
    return rows


def _top_n_with_other(counter: Counter, n: int) -> tuple[list[str], list[int]]:
    items = counter.most_common()
    if not items:
        return [], []
    if len(items) <= n:
        return [str(k) for k, _ in items], [c for _, c in items]
    top = items[:n]
    other = sum(c for _, c in items[n:])
    labels = [str(k) for k, _ in top] + ["(other)"]
    counts = [c for _, c in top] + [other]
    return labels, counts


def save_distribution_chart(
    out_path: Path,
    archetypes: Counter,
    models: Counter,
    rating_values: list[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 12), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.0, 1.0])

    # Archetypes: horizontal bars (top-N + other)
    ax0 = fig.add_subplot(gs[0, :])
    alabels, acounts = _top_n_with_other(archetypes, TOP_ARCHETYPES_CHART)
    if alabels:
        y = list(range(len(alabels)))
        colors = plt.cm.tab20([i % 20 for i in y])
        ax0.barh(y, acounts, color=colors, edgecolor="white", linewidth=0.4)
        ax0.set_yticks(y, labels=alabels, fontsize=8)
        ax0.invert_yaxis()
        ax0.set_xlabel("Count")
        ax0.set_title("Archetype distribution (top categories + other)")
        ax0.grid(axis="x", alpha=0.35)
    else:
        ax0.text(0.5, 0.5, "No archetype data", ha="center", va="center", transform=ax0.transAxes)

    # Models
    ax1 = fig.add_subplot(gs[1, 0])
    if models:
        order = sorted(models.keys(), key=lambda k: models[k], reverse=True)
        mcts = [models[k] for k in order]
        x = range(len(order))
        ax1.bar(x, mcts, color="steelblue", edgecolor="white")
        ax1.set_xticks(list(x), labels=order, rotation=22, ha="right", fontsize=9)
        ax1.set_ylabel("Count")
        ax1.set_title("Model (_model)")
        ax1.grid(axis="y", alpha=0.35)
    else:
        ax1.text(0.5, 0.5, "No _model data", ha="center", va="center", transform=ax1.transAxes)

    # Ratings: 0.5★ buckets
    ax2 = fig.add_subplot(gs[1, 1])
    if rating_values:
        bucket = Counter()
        for v in rating_values:
            b = min(5.0, max(1.0, round(v * 2) / 2))
            bucket[b] += 1
        stars = sorted(bucket.keys())
        xs = range(len(stars))
        ax2.bar(xs, [bucket[s] for s in stars], color="coral", edgecolor="white")
        ax2.set_xticks(list(xs), labels=[str(s) for s in stars])
        ax2.set_xlabel("Rating (★, 0.5 steps)")
        ax2.set_ylabel("Count")
        ax2.set_title("Rating distribution (rounded to 0.5★)")
        ax2.grid(axis="y", alpha=0.35)
    else:
        ax2.text(0.5, 0.5, "No rating data", ha="center", va="center", transform=ax2.transAxes)

    # Exact rating histogram (finer view)
    ax3 = fig.add_subplot(gs[2, :])
    if rating_values:
        ax3.hist(
            rating_values,
            bins=min(40, max(10, len(set(rating_values)))),
            color="seagreen",
            edgecolor="white",
            linewidth=0.5,
        )
        ax3.set_xlabel("Rating (exact)")
        ax3.set_ylabel("Count")
        ax3.set_title("Rating histogram (raw values)")
        ax3.grid(axis="y", alpha=0.35)
    else:
        ax3.text(0.5, 0.5, "No rating data", ha="center", va="center", transform=ax3.transAxes)

    fig.suptitle("Generated review distributions", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_distribution(name: str, counter: Counter, total: int) -> None:
    print(f"\n{name} (n = {total})")
    print("-" * 56)
    w = max(len(str(k)) for k in counter) if counter else 10
    for key, count in counter.most_common():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {str(key):<{w}}  {count:6d}  {pct:6.2f}%")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to JSONL (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_CHART,
        help=f"PNG path for charts (default: {DEFAULT_CHART})",
    )
    p.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip writing the PNG (text stats only)",
    )
    args = p.parse_args()
    path = args.input.resolve()

    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = load_reviews(path)
    n = len(rows)
    if n == 0:
        print("No rows loaded.")
        sys.exit(0)

    archetypes = Counter()
    models = Counter()
    ratings = Counter()
    missing_arch = 0
    missing_model = 0
    missing_rating = 0

    rating_values: list[float] = []

    for r in rows:
        a = r.get("_archetype")
        if a is None or a == "":
            missing_arch += 1
        else:
            archetypes[str(a)] += 1

        m = r.get("_model")
        if m is None or m == "":
            missing_model += 1
        else:
            models[str(m)] += 1

        rv = r.get("rating")
        if rv is None:
            missing_rating += 1
        else:
            try:
                rf = float(rv)
            except (TypeError, ValueError):
                missing_rating += 1
            else:
                ratings[rf] += 1
                rating_values.append(rf)

    print(f"File: {path}")
    print(f"Total reviews: {n}")
    if missing_arch:
        print(f"Missing _archetype: {missing_arch}")
    if missing_model:
        print(f"Missing _model: {missing_model}")
    if missing_rating:
        print(f"Missing/invalid rating: {missing_rating}")

    print_distribution("Archetype distribution", archetypes, n - missing_arch)
    print_distribution("Model distribution (_model)", models, n - missing_model)
    if rating_values:
        print_distribution("Rating distribution (exact value)", ratings, len(rating_values))
        print("\nRating summary")
        print("-" * 56)
        print(f"  mean:   {statistics.mean(rating_values):.3f}")
        print(f"  median: {statistics.median(rating_values):.3f}")
        if len(rating_values) > 1:
            print(f"  stdev:  {statistics.stdev(rating_values):.3f}")
        # Star buckets (1–5 in 0.5 steps) for a compact view
        bucket = Counter()
        for v in rating_values:
            b = min(5.0, max(1.0, round(v * 2) / 2))
            bucket[b] += 1
        print_distribution("Rating distribution (rounded to 0.5★)", bucket, len(rating_values))

    if not args.no_chart:
        try:
            out = args.output.resolve()
            save_distribution_chart(out, archetypes, models, rating_values)
            print(f"\nChart saved: {out}")
        except ModuleNotFoundError as e:
            print(f"\nSkipping chart (install matplotlib): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
