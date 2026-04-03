"""
Regression Checker — Session 2 (Instructor Version)

Compares current eval scores against a saved baseline.
Flags any metric that drops more than the threshold.

Usage:
  python scripts/check_regression.py
  python scripts/check_regression.py --baseline scripts/baseline_scores.json
  python scripts/check_regression.py --current eval_results.json
  python scripts/check_regression.py --threshold 3.0
"""
import os
import sys
import json
import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

load_dotenv()

console = Console()

SCRIPT_DIR = os.path.dirname(__file__)

DEFAULT_BASELINE = os.path.join(SCRIPT_DIR, "baseline_scores.json")
DEFAULT_CURRENT = os.path.join(SCRIPT_DIR, "..", "eval_results.json")
DEFAULT_THRESHOLD = 5.0  # percentage points


# =========================================================================
# CORE REGRESSION LOGIC
# =========================================================================

def load_baseline(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_current(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # eval_results.json wraps scores in "summary" key
    if "summary" in data:
        return data["summary"]
    return data


def check_regression(current: dict, baseline: dict, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """
    Compare current scores against baseline.
    Returns a list of regressions (metrics that dropped more than threshold).
    """
    metrics = [
        ("retrieval_hit_rate", "Retrieval Hit Rate"),
        ("avg_faithfulness", "Answer Faithfulness"),
        ("avg_correctness", "Answer Correctness"),
    ]

    regressions = []
    for key, label in metrics:
        if key not in baseline or key not in current:
            continue
        baseline_val = baseline[key]
        current_val = current[key]
        delta = current_val - baseline_val

        regressions.append({
            "metric": label,
            "key": key,
            "baseline": baseline_val,
            "current": current_val,
            "delta": round(delta, 1),
            "is_regression": delta < -threshold,
        })

    return regressions


def check_category_regressions(
    current_results: list[dict],
    baseline: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    Compare per-category correctness against baseline category breakdown.
    Only runs if baseline has category_breakdown and current results have category field.
    """
    if "category_breakdown" not in baseline or not current_results:
        return []

    # Recompute category scores from current results
    categories = {}
    for r in current_results:
        cat = r.get("category", "general")
        if cat not in categories:
            categories[cat] = {"hits": 0, "correct": [], "count": 0}
        categories[cat]["count"] += 1
        categories[cat]["hits"] += 1 if r["retrieval_hit"] else 0
        categories[cat]["correct"].append(r["correctness_score"])

    regressions = []
    baseline_cats = baseline["category_breakdown"]

    for cat, data in categories.items():
        if cat not in baseline_cats:
            continue
        current_correctness = (sum(data["correct"]) / len(data["correct"])) / 5 * 100
        baseline_correctness = baseline_cats[cat].get("correctness", None)
        if baseline_correctness is None:
            continue
        delta = current_correctness - baseline_correctness
        regressions.append({
            "category": cat,
            "baseline_correctness": baseline_correctness,
            "current_correctness": round(current_correctness, 1),
            "delta": round(delta, 1),
            "count": data["count"],
            "is_regression": delta < -threshold,
        })

    return sorted(regressions, key=lambda x: x["delta"])


# =========================================================================
# DISPLAY
# =========================================================================

def display_results(regressions: list[dict], category_regressions: list[dict], threshold: float):

    has_regression = any(r["is_regression"] for r in regressions)

    # Headline status
    if has_regression:
        console.print()
        console.print(Panel(
            "[bold red]❌  REGRESSION DETECTED[/]\n"
            f"[dim]One or more metrics dropped more than {threshold:.0f} percentage points below baseline.[/]",
            border_style="red",
        ))
    else:
        console.print()
        console.print(Panel(
            "[bold green]✅  NO REGRESSION[/]\n"
            f"[dim]All metrics within {threshold:.0f} percentage points of baseline.[/]",
            border_style="green",
        ))

    # Per-metric table
    console.print()
    table = Table(
        title="Metric Comparison",
        box=box.ROUNDED,
        title_style="bold cyan",
    )
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Baseline", justify="right", width=10)
    table.add_column("Current", justify="right", width=10)
    table.add_column("Delta", justify="right", width=10)
    table.add_column("Status", justify="center", width=12)

    for r in regressions:
        delta_color = "red" if r["is_regression"] else "green" if r["delta"] >= 0 else "yellow"
        delta_str = f"[{delta_color}]{r['delta']:+.1f}%[/]"
        status = "[bold red]REGRESSION[/]" if r["is_regression"] else "[bold green]OK[/]"
        table.add_row(
            r["metric"],
            f"{r['baseline']:.1f}%",
            f"{r['current']:.1f}%",
            delta_str,
            status,
        )

    console.print(table)

    # Category breakdown if available
    if category_regressions:
        console.print()
        cat_table = Table(
            title="Per-Category Correctness",
            box=box.SIMPLE,
            title_style="bold magenta",
        )
        cat_table.add_column("Category", style="cyan", width=20)
        cat_table.add_column("Count", justify="center", width=7)
        cat_table.add_column("Baseline", justify="right", width=10)
        cat_table.add_column("Current", justify="right", width=10)
        cat_table.add_column("Delta", justify="right", width=10)
        cat_table.add_column("Status", justify="center", width=12)

        for r in category_regressions:
            delta_color = "red" if r["is_regression"] else "green" if r["delta"] >= 0 else "yellow"
            delta_str = f"[{delta_color}]{r['delta']:+.1f}%[/]"
            status = "[bold red]REGRESSION[/]" if r["is_regression"] else "[bold green]OK[/]"
            cat_table.add_row(
                r["category"],
                str(r["count"]),
                f"{r['baseline_correctness']:.1f}%",
                f"{r['current_correctness']:.1f}%",
                delta_str,
                status,
            )

        console.print(cat_table)

    # Actionable guidance for regressions
    regressions_found = [r for r in regressions if r["is_regression"]]
    if regressions_found:
        console.print()
        for r in regressions_found:
            if r["key"] == "retrieval_hit_rate":
                console.print(Panel(
                    f"[bold]Retrieval hit rate dropped {abs(r['delta']):.1f}%[/]\n\n"
                    "Likely causes:\n"
                    "• Chunking change broke semantic coherence\n"
                    "• Embedding model changed\n"
                    "• TOP_K was reduced\n\n"
                    "Fix: check ingest.py CHUNK_SIZE and rag.py TOP_K",
                    title="[bold red]Regression: Retrieval[/]",
                    border_style="red",
                ))
            elif r["key"] == "avg_faithfulness":
                console.print(Panel(
                    f"[bold]Faithfulness dropped {abs(r['delta']):.1f}%[/]\n\n"
                    "Likely causes:\n"
                    "• System prompt was loosened (removed 'only use context' rule)\n"
                    "• Temperature was raised\n"
                    "• Context is nosier (more irrelevant chunks retrieved)\n\n"
                    "Fix: check SYSTEM_PROMPT and temperature in rag.py",
                    title="[bold red]Regression: Faithfulness[/]",
                    border_style="red",
                ))
            elif r["key"] == "avg_correctness":
                console.print(Panel(
                    f"[bold]Correctness dropped {abs(r['delta']):.1f}%[/]\n\n"
                    "Likely causes:\n"
                    "• Smaller chunks lost cross-section context\n"
                    "• Right doc retrieved but wrong chunk (boundary issue)\n"
                    "• Prompt change altered response structure\n\n"
                    "Fix: run eval with --include-hard to see which categories dropped",
                    title="[bold red]Regression: Correctness[/]",
                    border_style="red",
                ))


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Regression checker for RAG eval")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE,
                        help=f"Path to baseline scores JSON (default: {DEFAULT_BASELINE})")
    parser.add_argument("--current", type=str, default=DEFAULT_CURRENT,
                        help=f"Path to current eval results JSON (default: {DEFAULT_CURRENT})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Regression threshold in percentage points (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    # Load files
    if not os.path.exists(args.baseline):
        console.print(f"[red]Baseline not found: {args.baseline}[/]")
        console.print("[dim]Run eval_harness.py first, then save scores as baseline_scores.json[/]")
        sys.exit(1)

    if not os.path.exists(args.current):
        console.print(f"[red]Current eval results not found: {args.current}[/]")
        console.print("[dim]Run: python scripts/eval_harness.py[/]")
        sys.exit(1)

    baseline = load_baseline(args.baseline)
    current = load_current(args.current)

    console.print(Panel(
        f"[bold]Regression Check[/]\n"
        f"[dim]Baseline: {args.baseline}[/]\n"
        f"[dim]Current:  {args.current}[/]\n"
        f"[dim]Threshold: {args.threshold:.0f} percentage points[/]",
        title="[bold cyan]Regression Checker[/]",
        border_style="cyan",
    ))

    # Compute regressions
    regressions = check_regression(current, baseline, args.threshold)

    # Category-level regressions (only if current file has full results)
    category_regressions = []
    if os.path.exists(args.current):
        with open(args.current) as f:
            full_data = json.load(f)
        if "results" in full_data:
            category_regressions = check_category_regressions(
                full_data["results"], baseline, args.threshold
            )

    display_results(regressions, category_regressions, args.threshold)

    # Exit code — useful for CI/CD: non-zero if regression
    has_regression = any(r["is_regression"] for r in regressions)
    sys.exit(1 if has_regression else 0)


if __name__ == "__main__":
    main()
