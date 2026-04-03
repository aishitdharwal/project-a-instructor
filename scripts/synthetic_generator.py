"""
Synthetic Question Generator — Session 2 (Instructor Version)

Generates evaluation questions from corpus documents using GPT-4o-mini.
Supports three personas to stress-test vocabulary mismatch and edge cases.

Personas:
  standard    — Clean customer language, direct questions
  frustrated  — Emotional, urgent, non-standard phrasing
  mismatch    — Different vocabulary than the source docs

Usage:
  python scripts/synthetic_generator.py
  python scripts/synthetic_generator.py --doc 02_premium_membership.md
  python scripts/synthetic_generator.py --persona frustrated --count 5
  python scripts/synthetic_generator.py --all-docs --persona mismatch
  python scripts/synthetic_generator.py --merge   # merge output into golden_dataset.json
"""
import os
import sys
import json
import argparse
import glob

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
console = Console()

SCRIPT_DIR = os.path.dirname(__file__)
CORPUS_DIR = os.path.join(SCRIPT_DIR, "..", "corpus")

# =========================================================================
# PERSONA PROMPTS
# =========================================================================

PERSONA_PROMPTS = {
    "standard": """You are generating evaluation questions for a customer support RAG system.

Given the documentation below, generate {count} customer support questions that test retrieval and generation quality.

Rules:
- Use natural customer language — NOT the exact phrasing from the document
- Each question should be answerable from this document alone
- Mix difficulty: some straightforward lookups, some requiring careful reading
- For each question, the expected_answer should be specific and include key details (numbers, dates, conditions)
- Use category names from this list: returns, shipping, payments, warranty, membership, orders, products, account, rewards, promotions, sustainability, business

Document: {doc_name}
Content:
{doc_text}

Respond ONLY with a valid JSON array. Each object must have:
  id, query, expected_answer, expected_source, difficulty (easy/medium/hard), category""",

    "frustrated": """You are generating adversarial evaluation questions for a customer support RAG system.

These questions come from FRUSTRATED customers who are upset, under time pressure, or venting emotionally. The language is informal, sometimes unclear, and does NOT match the documentation vocabulary.

Document: {doc_name}
Content:
{doc_text}

Generate {count} questions a frustrated customer might actually ask.

Examples of frustrated phrasing:
- "why cant i return this thing already" instead of "what is the return window"
- "they charged me wrong and now i cant get my money back" instead of "refund process"
- "i spent SO much money and still dont get free shipping??" instead of "Premium shipping benefits"

Rules:
- NO documentation vocabulary in the queries
- Include emotion, urgency, or frustration in the phrasing
- Questions should still be answerable from this document
- expected_answer should be calm and correct (what a good agent would say)

Respond ONLY with a valid JSON array. Each object must have:
  id, query, expected_answer, expected_source, difficulty (easy/medium/hard), category""",

    "mismatch": """You are generating vocabulary-mismatch evaluation questions for a RAG system stress test.

These questions use completely DIFFERENT vocabulary than the source document. The semantic meaning is the same, but the words are different. This tests whether the embedding model can handle paraphrase and synonym matching.

Document: {doc_name}
Content:
{doc_text}

Generate {count} questions where:
- The meaning matches something in the document
- The vocabulary does NOT match the document's exact phrasing
- Use synonyms, paraphrases, indirect references

Examples:
- "send it back" instead of "return"
- "get my cash" instead of "refund"
- "loyalty program tiers" instead of "Premium membership levels"
- "when does my coverage kick in" instead of "warranty period"

Rules:
- Avoid any words that appear verbatim in the document
- expected_answer should correctly address the question
- difficulty should reflect how hard the vocabulary mismatch is

Respond ONLY with a valid JSON array. Each object must have:
  id, query, expected_answer, expected_source, difficulty (easy/medium/hard), category"""
}


# =========================================================================
# CORE GENERATION FUNCTION
# =========================================================================

def generate_questions(doc_name: str, doc_text: str, persona: str = "standard", count: int = 5) -> list[dict]:
    """Generate synthetic questions for a single document using the specified persona."""

    prompt = PERSONA_PROMPTS[persona].format(
        doc_name=doc_name,
        doc_text=doc_text[:3000],  # Trim to avoid token limit
        count=count,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,  # Higher temp = more varied questions
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        questions = json.loads(text)
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON parse error for {doc_name}: {e}[/]")
        return []

    # Normalise: ensure expected_source is set to doc_name
    for q in questions:
        q["expected_source"] = doc_name
        q["persona"] = persona

    return questions


def assign_ids(questions: list[dict], existing_dataset: list[dict], prefix: str = "s") -> list[dict]:
    """Assign sequential IDs that don't conflict with existing dataset."""
    existing_ids = {q["id"] for q in existing_dataset}
    counter = 1
    for q in questions:
        while f"{prefix}{counter:03d}" in existing_ids:
            counter += 1
        q["id"] = f"{prefix}{counter:03d}"
        existing_ids.add(q["id"])
        counter += 1
    return questions


def load_golden_dataset() -> list[dict]:
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_golden_dataset(dataset: list[dict]):
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Saved {len(dataset)} entries to golden_dataset.json[/]")


# =========================================================================
# DISPLAY
# =========================================================================

def display_questions(questions: list[dict], doc_name: str, persona: str):
    console.print()
    console.print(Panel(
        f"[bold]Generated {len(questions)} questions[/]\n"
        f"[dim]Document: {doc_name} | Persona: {persona}[/]",
        border_style="cyan",
    ))

    table = Table(box=box.SIMPLE, show_lines=True)
    table.add_column("ID", style="dim", width=6)
    table.add_column("Query", width=45)
    table.add_column("Difficulty", width=8, justify="center")
    table.add_column("Category", width=14)
    table.add_column("Expected Answer (preview)", width=40)

    diff_colors = {"easy": "green", "medium": "yellow", "hard": "red"}

    for q in questions:
        diff = q.get("difficulty", "easy")
        color = diff_colors.get(diff, "white")
        table.add_row(
            q.get("id", "?"),
            q["query"][:43] + "..." if len(q["query"]) > 43 else q["query"],
            f"[{color}]{diff}[/]",
            q.get("category", "general"),
            q["expected_answer"][:38] + "..." if len(q["expected_answer"]) > 38 else q["expected_answer"],
        )

    console.print(table)


# =========================================================================
# CRITIQUE HELPER (used live during session)
# =========================================================================

def critique_questions(questions: list[dict]) -> list[dict]:
    """
    Use GPT-4o-mini to critique each generated question.
    Flags questions that are too clean, too vague, or use doc vocabulary.

    Returns questions with a 'critique' field added.
    """
    critiques = []
    for q in questions:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Rate this synthetic RAG evaluation question on 2 dimensions:

1. Realism (1-5): Does it sound like a real customer wrote it?
   - 5 = Completely natural, could be from a real support chat
   - 3 = Mostly natural but slightly formal
   - 1 = Obviously AI-generated, uses documentation vocabulary

2. Difficulty (1-5): How hard is it for the RAG system?
   - 5 = Requires cross-document reasoning or vocabulary bridging
   - 3 = Moderate — requires careful reading
   - 1 = Trivial keyword match

Question: {q['query']}
Expected answer: {q['expected_answer']}

Respond ONLY with JSON: {{"realism": N, "difficulty_actual": N, "flag": "keep/rewrite/drop", "note": "one line"}}"""
            }]
        )
        try:
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            critique = json.loads(text)
            q["critique"] = critique
        except Exception:
            q["critique"] = {"realism": 3, "difficulty_actual": 3, "flag": "keep", "note": "parse error"}
        critiques.append(q)

    return critiques


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Synthetic question generator for RAG eval")
    parser.add_argument("--doc", type=str, help="Specific corpus doc to generate from (e.g. 02_premium_membership.md)")
    parser.add_argument("--all-docs", action="store_true", help="Generate from all corpus documents")
    parser.add_argument("--persona", type=str, default="standard", choices=["standard", "frustrated", "mismatch"],
                        help="Question persona style")
    parser.add_argument("--count", type=int, default=5, help="Questions per document")
    parser.add_argument("--critique", action="store_true", help="Run auto-critique on generated questions")
    parser.add_argument("--merge", action="store_true", help="Merge generated questions into golden_dataset.json")
    parser.add_argument("--save", type=str, help="Save output to a JSON file (e.g. synthetic_output.json)")
    args = parser.parse_args()

    all_generated = []

    if args.all_docs:
        doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
        console.print(f"\n[bold]Generating from {len(doc_files)} documents with persona: [cyan]{args.persona}[/][/]\n")
        for filepath in doc_files:
            doc_name = os.path.basename(filepath)
            with open(filepath) as f:
                doc_text = f.read()
            console.print(f"  Generating from {doc_name}...", style="dim")
            questions = generate_questions(doc_name, doc_text, args.persona, args.count)
            all_generated.extend(questions)

    elif args.doc:
        filepath = os.path.join(CORPUS_DIR, args.doc)
        if not os.path.exists(filepath):
            console.print(f"[red]File not found: {filepath}[/]")
            return
        with open(filepath) as f:
            doc_text = f.read()
        questions = generate_questions(args.doc, doc_text, args.persona, args.count)
        all_generated.extend(questions)

    else:
        # Default: generate from the 3 most interesting docs for live demo
        demo_docs = [
            "02_premium_membership.md",
            "07_promotional_events.md",
            "11_internal_pricing.md",
        ]
        console.print(f"\n[bold]Demo mode: generating from 3 docs with persona: [cyan]{args.persona}[/][/]\n")
        for doc_name in demo_docs:
            filepath = os.path.join(CORPUS_DIR, doc_name)
            with open(filepath) as f:
                doc_text = f.read()
            console.print(f"  Generating from {doc_name}...", style="dim")
            questions = generate_questions(doc_name, doc_text, args.persona, args.count)
            all_generated.extend(questions)

    if not all_generated:
        console.print("[red]No questions generated.[/]")
        return

    # Assign IDs
    existing = load_golden_dataset()
    all_generated = assign_ids(all_generated, existing, prefix="s")

    # Display
    for doc_name in {q["expected_source"] for q in all_generated}:
        doc_qs = [q for q in all_generated if q["expected_source"] == doc_name]
        display_questions(doc_qs, doc_name, args.persona)

    # Optional: critique
    if args.critique:
        console.print("\n[bold yellow]Running auto-critique...[/]")
        all_generated = critique_questions(all_generated)

        crit_table = Table(title="Critique Results", box=box.SIMPLE)
        crit_table.add_column("ID", width=6)
        crit_table.add_column("Query", width=45)
        crit_table.add_column("Realism", justify="center", width=8)
        crit_table.add_column("Difficulty", justify="center", width=9)
        crit_table.add_column("Flag", justify="center", width=8)
        crit_table.add_column("Note", width=30)

        for q in all_generated:
            c = q.get("critique", {})
            flag = c.get("flag", "keep")
            flag_color = "green" if flag == "keep" else "yellow" if flag == "rewrite" else "red"
            crit_table.add_row(
                q["id"],
                q["query"][:43] + "...",
                str(c.get("realism", "?")),
                str(c.get("difficulty_actual", "?")),
                f"[{flag_color}]{flag}[/]",
                c.get("note", "")[:28],
            )
        console.print(crit_table)

    # Save to file
    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_generated, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Saved to {args.save}[/]")

    # Merge into golden dataset
    if args.merge:
        existing = load_golden_dataset()
        merged = existing + all_generated
        save_golden_dataset(merged)
        console.print(f"\n[bold green]Merged {len(all_generated)} new questions. Total: {len(merged)}[/]")
    else:
        console.print(f"\n[dim]Generated {len(all_generated)} questions. Use --merge to add to golden_dataset.json[/]")


if __name__ == "__main__":
    main()
