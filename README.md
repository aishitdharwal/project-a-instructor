# Workshop Demo: "Why your AI demo will break in production"

A deliberately broken RAG system for live debugging during the workshop.
The system looks good on the surface (88%+ eval scores) but fails on real-world queries.

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Embed Query │────▶│   Retrieve    │────▶│ Assemble Context │
│  (OpenAI)    │     │  (pgvector)   │     │  (naive concat)  │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │   Generate    │
                                         │   (GPT-4o)    │
                                         └──────────────┘
                                                 │
                    All stages traced in LangFuse ▼
```

## Deliberate Failure Traps

1. **Evaluation blindness**: Golden dataset is biased toward easy questions → inflated scores
2. **Chunking + ranking**: Premium return extensions split across chunk boundaries
3. **Vocabulary mismatch**: "refund procedure" vs "return policy" vs "get money back"
4. **PII in context**: Support tickets contain customer emails, phone numbers, card info
5. **Internal doc leak**: Internal pricing doc with cost prices and discount authority

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker (for pgvector)
- OpenAI API key
- LangFuse account (cloud.langfuse.com)

### 2. Setup

```bash
# Clone/navigate to this directory
cd workshop-demo

# Copy env file and fill in your keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, LANGFUSE keys

# Start pgvector
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Set up the database
python scripts/setup_db.py

# Ingest documents (embeds and stores in pgvector)
python scripts/ingest.py
```

### 3. Run the Demo CLI

```bash
python scripts/demo.py
```

This opens an interactive CLI with pre-loaded demo queries:
- **Query 1**: Clean query that works well (establishes baseline)
- **Query 2**: THE BREAKING QUERY (Premium Silver + Diwali sale + laptop + 40 days)
- **Query 3**: Multi-hop reasoning required
- **Query 4**: PII leak risk
- **Query 5**: Internal data leak risk
- **Query 6**: Vocabulary mismatch test

### 4. Run the Eval Harness

```bash
# Run with ONLY the biased golden dataset (produces the "88%" lie)
python scripts/eval_harness.py

# Run WITH hard queries included (reveals the true performance)
python scripts/eval_harness.py --include-hard
```

## Workshop Demo Flow

### Phase 1: "The Good Eval" (3 min)
```bash
python scripts/eval_harness.py
```
Show the 88%+ scores. "By most standards, this is a good system."

### Phase 2: "The Breaking Query" (5 min)
```bash
python scripts/demo.py
# Choose option 1 first (clean query - it works)
# Then choose option 2 (THE BREAKING QUERY - it fails)
```

### Phase 3: "Pull Up the Trace" (5 min)
Click the LangFuse trace URL from the demo output.
Walk through: retrieval results → ranking → context assembly → generation.

### Phase 4: "The Eval Was Lying" (5 min)
```bash
python scripts/eval_harness.py --include-hard
```
Show how scores drop when hard queries are included.
"That 88% was a vanity metric."

## File Structure

```
workshop-demo/
├── corpus/                    # 19 Acmera company documents
│   ├── 01_return_policy.md
│   ├── 02_premium_membership.md   ← Main failure trap source
│   ├── 03_shipping_policy.md
│   ├── ...
│   ├── 15_slack_support_chat.md   ← PII source
│   └── 19_acmera_business.md
├── scripts/
│   ├── setup_db.py               # Create pgvector table
│   ├── ingest.py                 # Naive chunking + embedding
│   ├── rag.py                    # Core RAG pipeline + LangFuse
│   ├── demo.py                   # Interactive workshop CLI
│   ├── eval_harness.py           # Evaluation with LLM-as-judge
│   └── golden_dataset.json       # Biased eval dataset (40 easy Qs)
├── docker-compose.yml            # pgvector container
├── .env.example                  # Config template
├── requirements.txt
└── README.md
```

## Pre-Workshop Checklist

- [ ] `.env` filled with real API keys
- [ ] `docker-compose up -d` running (pgvector healthy)
- [ ] `python scripts/setup_db.py` — no errors
- [ ] `python scripts/ingest.py` — all 19 docs ingested
- [ ] `python scripts/eval_harness.py` — produces 85%+ scores
- [ ] `python scripts/demo.py` — Query 1 works well, Query 2 breaks
- [ ] LangFuse traces are appearing in your dashboard
- [ ] Terminal font is large enough for screen sharing
- [ ] LangFuse dashboard zoomed in for screen sharing
- [ ] Run through the full demo flow at least 3 times
- [ ] Backup: screenshots of key LangFuse traces saved locally
