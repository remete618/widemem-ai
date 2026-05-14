# Mini-LoCoMo Regression Gate

A cheap, fast regression gate over a 50-question stratified subset of LoCoMo. Used to catch retrieval regressions during feature development without paying for the full 1,540-question / 13-hour / ~$5 run.

## What it measures

| Metric | What | Why |
|---|---|---|
| Overall J | LLM-as-judge correctness across all 50 questions (averaged over 3 judge runs each) | Headline regression number |
| Per-category J | Same metric, sliced by `single-hop`, `multi-hop`, `open-domain`, `temporal` | Catches category-specific regressions |
| Avg memory tokens | Tokens retrieved per query, averaged | Token-efficiency regression |
| Avg search latency | Local FAISS search time, p50-ish (mean) | Latency regression |

## What it does NOT measure

- **Extraction quality.** Memory stores are reused from the v1 ingestion run (`benchmark/results/widemem_stores/`), so the extraction prompt is v1.3 era. The gate measures retrieval and scoring code paths only.
- **Statistical certainty.** n=50 is small. A ±2 point fluctuation between runs of the same code is normal noise.
- **Absolute leaderboard standing.** For a fair LoCoMo number to publish, run the full `run_locomo.py` with fresh ingestion.

For extraction quality, use the full LoCoMo run.

## Sample design

- **Stratified**: 12-13 questions per category, total 50
- **Deterministic**: seeded shuffle (`SAMPLE_SEED = 42`). Same 50 questions on every run.
- **Stable ordering**: each category sorted by `(sample_id, question)` before the seeded shuffle, so determinism survives changes to the source data file order.

| Category | Sample | LoCoMo total |
|---|---|---|
| single-hop | 12 | 282 |
| multi-hop | 13 | 841 |
| open-domain | 12 | 96 |
| temporal | 13 | 321 |
| **total** | **50** | **1,540** |

## Cost and time per run

- ~300 API calls (50 answer-gen + 250 judge runs at 5 runs per question)
- ~$0.07 per run on gpt-4o-mini
- ~3-4 minutes elapsed

Cheap enough to run on every feature branch.

## Gate-pass criteria

A PR fails the gate if **either**:

| Check | Threshold | Why |
|---|---|---|
| Overall J drop vs baseline | > 3.0 points | Headline regression detector |
| Multi-hop J drop vs baseline | > 5.0 points | Multi-hop is widemem's current strength; tighter tolerance |

Other categories (`single-hop`, `open-domain`, `temporal`) are reported as informational; they can fluctuate freely. They are currently weak categories, and the v1.5 work is specifically targeting them, so a real improvement there should not be gated against by the previous-baseline.

## Judge-run averaging

Each question is judged 5 times by GPT-4o-mini, and the score is the fraction of CORRECT verdicts. 3 runs was the initial setting; raised to 5 after observing that a single judge flip on a single question could push a category by ~8 points (1 of 13 × 100 / 3). The Mem0 paper uses 10 runs; we use 5 as the cost/stability balance for a gate that runs on every PR.

If gate flapping returns at this setting, the next escalation is 10 judge runs (~$0.10, ~5 minutes) rather than widening the tolerance, because the tolerance is the load-bearing safety check.

## Usage

### Run the gate

```bash
cd /Users/radu/widemem-ai
set -a; source .env.local; set +a    # load OPENAI_API_KEY
.venv/bin/python3 benchmark/mini_locomo.py
```

If a baseline exists at `benchmark/results/mini_locomo_baseline.json`, the gate auto-compares against it and exits non-zero on FAIL.

### Save the current run as the new baseline

```bash
.venv/bin/python3 benchmark/mini_locomo.py --save-as-baseline
```

Use this when:
- You ship an intentional improvement that you want to lock in as the new floor
- The previous baseline was from a stale codebase

### Compare against an explicit baseline file

```bash
.venv/bin/python3 benchmark/mini_locomo.py --baseline path/to/old_baseline.json
```

Useful for diffing two specific versions, e.g., before and after a refactor.

## CI integration

Not yet wired into CI. Suggested when v1.5 work starts:

```yaml
# .github/workflows/mini-locomo.yml (sketch)
on:
  pull_request:
    branches: [main]
    paths:
      - 'widemem/retrieval/**'
      - 'widemem/scoring/**'
      - 'widemem/core/**'

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev,faiss]"
      - run: python benchmark/mini_locomo.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The store directory (`benchmark/results/widemem_stores/`) is gitignored, so CI would need to either restore it from a release artifact or run ingestion once.

For now: run the gate locally before every PR merge.

## Files

- `benchmark/mini_locomo.py` — the gate runner
- `benchmark/results/mini_locomo_baseline.json` — the line every PR has to clear
- `benchmark/results/mini_locomo_<sha>_<timestamp>.json` — per-run results

## When to regenerate the baseline

- After a major intentional improvement (e.g., v1.5 hybrid search ships with new defaults)
- After widening / narrowing the gate criteria
- Never silently. Always in an explicit commit with rationale in the message.
