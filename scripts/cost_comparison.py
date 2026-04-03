"""
Cost Comparison Generator for Workshop 2, Layer 3.

Simulates model routing vs all-GPT-4o to produce the cost comparison
table you'll show on screen during the workshop.

Runs N queries through two modes:
  1. All queries → GPT-4o (expensive baseline)
  2. Routed: simple → GPT-4o-mini, complex → GPT-4o

Produces a formatted comparison table with per-query and daily costs.

Run: python scripts/cost_comparison.py
Optional: python scripts/cost_comparison.py --queries 50
"""
import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
console = Console()

# Pricing per 1M tokens (USD, as of 2025)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# INR conversion (approximate)
USD_TO_INR = 85

SYSTEM_PROMPT = """You are a helpful customer support assistant for Acmera, an Indian e-commerce company. 
Answer the customer's question based on the provided context from our documentation.
Be concise but thorough."""

# Sample queries with difficulty labels
SAMPLE_QUERIES = [
    # SIMPLE — these work perfectly with mini
    {"query": "What is the standard return window?", "difficulty": "simple"},
    {"query": "Does Acmera ship internationally?", "difficulty": "simple"},
    {"query": "What payment methods does Acmera accept?", "difficulty": "simple"},
    {"query": "How do I track my order?", "difficulty": "simple"},
    {"query": "What is the maximum wallet balance?", "difficulty": "simple"},
    {"query": "How much does express shipping cost?", "difficulty": "simple"},
    {"query": "What cities have same-day delivery?", "difficulty": "simple"},
    {"query": "Is there cash on delivery?", "difficulty": "simple"},
    {"query": "What warranty comes with electronics?", "difficulty": "simple"},
    {"query": "Can I change my delivery address after ordering?", "difficulty": "simple"},
    {"query": "What items cannot be returned?", "difficulty": "simple"},
    {"query": "How do I create an Acmera account?", "difficulty": "simple"},
    {"query": "What is the referral program?", "difficulty": "simple"},
    {"query": "What smart home devices does Acmera sell?", "difficulty": "simple"},
    {"query": "How much does the ProBook X15 cost?", "difficulty": "simple"},
    {"query": "What is the dead pixel policy?", "difficulty": "simple"},
    # COMPLEX — these need GPT-4o
    {"query": "I'm a Premium Silver member and bought a laptop during the Diwali sale 40 days ago. Can I return it?", "difficulty": "complex"},
    {"query": "How does the refund process differ between Gold and standard customers for promotional electronics?", "difficulty": "complex"},
    {"query": "My Gold membership is expiring and I'm short of the threshold. What return privileges do I lose if downgraded?", "difficulty": "complex"},
    {"query": "I need to return 3 laptops from a corporate order of 8 and our company is Premium Gold. What's the process?", "difficulty": "complex"},
]


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a single query."""
    pricing = PRICING[model]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def run_query(query: str, model: str, context: str = "Sample context for cost estimation.") -> dict:
    """Run a single query and return token usage + cost."""
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext: {context}"},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=500,
    )

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
        "cost_inr": cost_usd * USD_TO_INR,
    }


def run_comparison(num_queries: int = 20):
    queries = SAMPLE_QUERIES[:num_queries]
    simple_count = sum(1 for q in queries if q["difficulty"] == "simple")
    complex_count = sum(1 for q in queries if q["difficulty"] == "complex")

    console.print(Panel(
        f"[bold]Running cost comparison on {num_queries} queries[/]\n"
        f"[dim]Simple: {simple_count} | Complex: {complex_count}[/]",
        title="[bold cyan]Cost Comparison Generator[/]",
        border_style="cyan",
    ))

    # Mode 1: All GPT-4o
    console.print("\n[bold yellow]Mode 1: All queries → GPT-4o[/]")
    all_4o_costs = []
    all_4o_tokens = []
    for i, q in enumerate(queries):
        console.print(f"  [{i+1}/{num_queries}] {q['query'][:50]}...", style="dim")
        result = run_query(q["query"], "gpt-4o")
        all_4o_costs.append(result["cost_inr"])
        all_4o_tokens.append(result["total_tokens"])

    # Mode 2: Routed (simple → mini, complex → 4o)
    console.print("\n[bold yellow]Mode 2: Routed (simple → mini, complex → 4o)[/]")
    routed_costs = []
    routed_tokens = []
    routed_models = {"gpt-4o": 0, "gpt-4o-mini": 0}
    for i, q in enumerate(queries):
        model = "gpt-4o" if q["difficulty"] == "complex" else "gpt-4o-mini"
        console.print(f"  [{i+1}/{num_queries}] → {model}: {q['query'][:45]}...", style="dim")
        result = run_query(q["query"], model)
        routed_costs.append(result["cost_inr"])
        routed_tokens.append(result["total_tokens"])
        routed_models[model] += 1

    # Calculate totals
    total_4o = sum(all_4o_costs)
    total_routed = sum(routed_costs)
    avg_4o = total_4o / num_queries
    avg_routed = total_routed / num_queries
    savings_pct = (1 - total_routed / total_4o) * 100

    # Scale to daily (assume 5000 queries/day with same distribution)
    daily_scale = 5000 / num_queries
    daily_4o = total_4o * daily_scale
    daily_routed = total_routed * daily_scale
    monthly_4o = daily_4o * 30
    monthly_routed = daily_routed * 30

    # Display results
    console.print()
    results_table = Table(
        title="Cost Comparison Results",
        box=box.ROUNDED,
        title_style="bold green",
    )
    results_table.add_column("Metric", style="bold", width=30)
    results_table.add_column("All GPT-4o", justify="right", width=18)
    results_table.add_column("Routed", justify="right", width=18)
    results_table.add_column("Savings", justify="right", width=12)

    results_table.add_row(
        f"Total cost ({num_queries} queries)",
        f"₹{total_4o:.2f}",
        f"₹{total_routed:.2f}",
        f"[bold green]{savings_pct:.0f}%[/]",
    )
    results_table.add_row(
        "Avg cost per query",
        f"₹{avg_4o:.4f}",
        f"₹{avg_routed:.4f}",
        f"[bold green]{savings_pct:.0f}%[/]",
    )
    results_table.add_row("", "", "", "")
    results_table.add_row(
        "Projected daily (5,000 queries)",
        f"₹{daily_4o:.0f}",
        f"₹{daily_routed:.0f}",
        f"[bold green]₹{daily_4o - daily_routed:.0f} saved[/]",
    )
    results_table.add_row(
        "Projected monthly",
        f"₹{monthly_4o:,.0f}",
        f"₹{monthly_routed:,.0f}",
        f"[bold green]₹{monthly_4o - monthly_routed:,.0f} saved[/]",
    )
    results_table.add_row("", "", "", "")
    results_table.add_row(
        "Model distribution",
        f"GPT-4o: {num_queries} (100%)",
        f"GPT-4o: {routed_models['gpt-4o']} ({routed_models['gpt-4o']/num_queries*100:.0f}%)\nMini: {routed_models['gpt-4o-mini']} ({routed_models['gpt-4o-mini']/num_queries*100:.0f}%)",
        "",
    )

    console.print(results_table)

    # The slide-ready summary
    console.print()
    console.print(Panel(
        f"[bold]SLIDE-READY SUMMARY[/]\n\n"
        f"All queries → GPT-4o:              [red]₹{daily_4o:,.0f}/day[/]  (₹{monthly_4o:,.0f}/month)\n"
        f"Routed (simple→mini, complex→4o):  [green]₹{daily_routed:,.0f}/day[/]  (₹{monthly_routed:,.0f}/month)\n"
        f"Savings:                            [bold green]{savings_pct:.0f}%[/]\n"
        f"Simple queries using mini:          {routed_models['gpt-4o-mini']}/{num_queries} ({routed_models['gpt-4o-mini']/num_queries*100:.0f}%)",
        title="[bold yellow]For Workshop Slide[/]",
        border_style="yellow",
    ))

    # Save raw data
    output = {
        "num_queries": num_queries,
        "simple_count": simple_count,
        "complex_count": complex_count,
        "all_gpt4o": {
            "total_cost_inr": round(total_4o, 4),
            "avg_per_query_inr": round(avg_4o, 6),
            "daily_5k_inr": round(daily_4o, 0),
            "monthly_inr": round(monthly_4o, 0),
        },
        "routed": {
            "total_cost_inr": round(total_routed, 4),
            "avg_per_query_inr": round(avg_routed, 6),
            "daily_5k_inr": round(daily_routed, 0),
            "monthly_inr": round(monthly_routed, 0),
            "model_distribution": routed_models,
        },
        "savings_pct": round(savings_pct, 1),
    }

    output_path = os.path.join(os.path.dirname(__file__), "..", "cost_comparison.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"\n[dim]Raw data saved to {output_path}[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to run")
    args = parser.parse_args()
    run_comparison(num_queries=args.queries)
