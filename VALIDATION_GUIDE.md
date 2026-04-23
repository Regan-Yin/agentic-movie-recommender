# Independent Validation Framework — Guide

This folder ships with a **second validation methodology** that runs
alongside the DSPy/GEPA loop (`dspy_gepa_benchmark.py`). Where GEPA optimizes
our agent's prompt against a bespoke, preference-first metric, the validator
here compares the tuned agent against a completely independent, **TF-IDF
baseline recommender** that does not use an LLM at all.

| Layer | Script | What it optimizes / reports |
|-------|--------|------------------------------|
| Prompt tuning | `dspy_gepa_benchmark.py` | Finds the best instruction style for `llm.py` |
| **Independent validation** | `validation.py` / `run_validation_example.py` / `streamlit_app.py` | Sanity-checks the tuned agent against a data-driven baseline |
| Contract tests | `test.py` | Professor-supplied pass/fail checks (timeout, valid id, no repeats) |

---

## Why have two validation layers?

- **GEPA** is a task-specific optimizer — it can "teach" the metric it is
  graded on. That is fine (and desirable) for tuning, but it does not prove
  the agent is broadly sensible.
- **This framework** grades the agent with a *different* methodology it has
  never seen: classical retrieval. If the two methodologies largely agree,
  we have strong evidence the agent is doing the right thing. When they
  disagree, the specific case is worth inspecting.

The baseline is restricted to the **same candidate pool the agent uses**
(`llm.TOP_MOVIES`, 350 movies). That keeps the comparison fair — the
baseline never recommends movies the agent couldn't.

---

## What "accuracy" means here

Multiple valid recommendations can exist for the same preferences, so exact
match is too strict. We combine several metrics:

- **Rank Score** *(primary)* — `101 − rank(agent_pick)` in the baseline's
  hidden top-100. Rank 1 → 100, rank 100 → 1, outside top-100 → 0.
- **Soft Match Rate** *(secondary)* — % of cases where the agent's pick has
  hybrid similarity ≥ 0.60 to the baseline's top-1.
- **Top-K Hit Rate** *(tertiary)* — % of cases where the agent's pick is in
  the baseline's top-5 / top-10 / top-100.
- **Similarity breakdown** — genre Jaccard, keyword Jaccard, overview
  TF-IDF cosine, and a hybrid composite.
- **Preference alignment** *(sanity)* — TF-IDF similarity between the user's
  preference text and each recommendation's overview.
- **Contract checks** — description length ≤ 500 chars, agent never
  re-recommends a movie from history, agent latency ≤ 20 s.

---

## Files in this validation setup

| File | Purpose |
|------|---------|
| `validation.py` | Core library: `BaselineRecommender`, `SimilarityMetrics`, `ValidationFramework`. Can be imported or run as a CLI. |
| `run_validation_example.py` | End-to-end script. Loads the GEPA eval cases, calls the live agent for each case, prints and saves the report. |
| `streamlit_app.py` | Interactive web UI. Live Agent mode calls `llm.get_recommendation()`; Manual mode accepts a TMDB ID. |
| `VALIDATION_GUIDE.md` | This document. |

Artifacts the tools write (already in `.gitignore`-style convention, safe to delete):

- `validation_results.csv` — one row per case with every metric
- `validation_summary.json` — aggregate report

---

## Installation

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn` and `streamlit` are added to `requirements.txt` for the
validation tools; they are not imported by `llm.py`, so `test.py`'s
dependency check still passes.

### Optional: enable the live agent

The validator can drive `llm.get_recommendation()` directly. For it to hit
Ollama rather than the deterministic fallback, export your key:

```bash
export OLLAMA_API_KEY="..."
```

Without the key the validator still runs — the agent just uses its offline
fallback path, which is fine for sanity-checking the framework.

---

## Quick start — 3 ways to use it

### A) Run the full benchmark

```bash
python run_validation_example.py
```

What it does:
1. Loads `tmdb_top1000_movies.csv`.
2. Loads `dspy_gepa_eval_cases.json` (40 cases).
3. Calls `llm.get_recommendation()` once per case.
4. Scores each pick against the baseline.
5. Prints the report and saves `validation_results.csv` + `validation_summary.json`.

Helpful flags:

```bash
# Smoke test on 5 random cases (cheap, verifies wiring)
python run_validation_example.py --sample 5

# First 3 cases, no LLM at all (uses baseline-top-1 as the "agent")
python run_validation_example.py --limit 3 --no-live

# Let the baseline search the entire CSV, not just llm.TOP_MOVIES
python run_validation_example.py --full-pool

# Use your own cases file
python run_validation_example.py --cases my_cases.json
```

### B) Interactive UI with Streamlit

```bash
streamlit run streamlit_app.py
```

Two modes in the sidebar:

1. **Live Agent** — type preferences, pick watched movies, click *Run the agent*.
   The app calls `llm.get_recommendation()`, shows the agent's pick + its
   description, and scores it against the baseline's hidden top-100.
2. **Manual TMDB ID** — paste the `tmdb_id` your agent produced elsewhere
   (e.g. from a prior run or `test.py`) and see where it ranks.

Both modes show:
- Baseline top-5 preview
- Agent pick with title, year, genres, description, latency
- Accuracy score (rank-based)
- Similarity breakdown (genre / keyword / overview / hybrid)
- Contract checks (≤ 500 chars, ≤ 20 s)

### C) Use the library directly

```python
import pandas as pd
from validation import build_validator

movies_df = pd.read_csv("tmdb_top1000_movies.csv")
validator = build_validator(movies_df=movies_df)  # restricts to llm.TOP_MOVIES

# Option 1: validate with a pre-computed agent pick
result = validator.validate_case(
    case_id="manual_001",
    preferences="I want a pure thriller with real tension and no romance subplot.",
    watch_history=["Crime 101"],
    watch_history_ids=[1171145],
    agent_tmdb_id=210577,                # whatever your agent recommended
    agent_description="Gone Girl ...",   # optional — enables contract checks
)

# Option 2: let the framework call llm.get_recommendation() for you
result = validator.validate_with_agent(
    case_id="manual_002",
    preferences="Give me a light feel-good comedy, not a dark crime piece.",
    watch_history=["The Shadow's Edge"],
    watch_history_ids=[1419406],
)

validator.print_report()
validator.save_results_csv("validation_results.csv")
validator.save_summary_json("validation_summary.json")
```

---

## How the baseline decides

`BaselineRecommender.recommend()` scores every candidate with:

| Component | Weight | What it measures |
|-----------|-------:|------------------|
| Overview / keywords / genres TF-IDF similarity to preference text | 0.30 | Free-text semantic match |
| Explicit preference-term keyword matches | 0.22 | Does the movie metadata actually contain the user's words? |
| Genre alignment | 0.18 | Jaccard overlap vs. inferred requested genres |
| Tone/mood boost | 0.12 | *dark*, *slow-burn*, *atmospheric*, *feel-good* ... |
| Watch-history similarity | 0.10 | Light signal — never dominates |
| Quality (vote avg + vote count) | 0.08 | Tie-breaker |
| Blocked-genre penalty (*no horror*, *not romance*) | **−0.35** | Hard exclusion for negations |
| Low-quality penalty | up to −0.16 | Discourages fringe picks |

Plus three tiny bonuses for strong direct matches. All component scores are
normalized to `[0, 1]` before blending.

This philosophy intentionally mirrors the intent-aware scoring in our
agent's RAG layer (see `llm._rank_candidates`), so when they disagree it
tends to be on hard judgment calls rather than obvious errors.

---

## Interpreting the report

```
===========================================================================
VALIDATION REPORT - AGENT VS BASELINE RECOMMENDER
===========================================================================

Test Coverage:
  Total Cases:       40
  Successful:        40
  Failed:            0

---------------------------------------------------------------------------
PRIMARY ACCURACY METRICS
---------------------------------------------------------------------------
  Avg Rank Score:     85.40
  Exact Match Rate:   22.5%
  Soft Match Rate:    60.0%

---------------------------------------------------------------------------
TOP-K HIT RATES
---------------------------------------------------------------------------
  Top-5  Hit Rate:    55.0%
  Top-10 Hit Rate:    75.0%
  Top-100 Hit Rate:   95.0%

---------------------------------------------------------------------------
AGENT CONTRACT CHECKS
---------------------------------------------------------------------------
  Description ≤ 500 chars:  100.0%
  History Respected:        100.0%
  Avg Agent Latency:        6.12s
  Max Agent Latency:        14.80s (20s DQ limit)
```

Good bands to aim for (empirical):

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Avg Rank Score | < 40 | 40–60 | 60–80 | > 80 |
| Soft Match Rate | < 25% | 25–40% | 40–55% | > 55% |
| Top-10 Hit Rate | < 40% | 40–60% | 60–80% | > 80% |
| Avg Hybrid Sim | < 0.30 | 0.30–0.40 | 0.40–0.55 | > 0.55 |
| Description ≤ 500 | < 95% | 95–99% | 99–100% | 100% |
| History Respected | < 98% | 98–99% | 99–100% | 100% |
| Max Agent Latency | > 20s (DQ) | 15–20s | 10–15s | < 10s |

**Red flags:**

- *Description ≤ 500 chars* below 100%: check `_sanitize_description` in
  `llm.py` — smart truncation should be catching this.
- *History Respected* below 100%: the agent picked a movie it was told
  the user already watched. The retry path in `get_recommendation` should
  prevent this; if it slips, inspect the offending case.
- *Max Agent Latency* > 20 s: `test.py` will fail. Lower
  `LLM_PRIMARY_TIMEOUT` in `llm.py` or shorten the prompt.
- *Soft Match Rate* < 25% while *Agent Pref Alignment* is still high:
  means the agent is matching user intent but via very different movies
  than the baseline. Not necessarily bad — inspect a few cases to decide.

---

## Per-case output (`validation_results.csv`)

Each row contains:

- `case_id`, `preferences_snippet`, `watch_history_count`
- `baseline_tmdb_id`, `baseline_title`, `baseline_score`, `baseline_rank`
- `agent_tmdb_id`, `agent_title`, `agent_description_length`,
  `agent_description_within_limit`, `agent_respected_history`,
  `agent_latency_s`
- `exact_match`, `soft_match`, `top5_hit`, `top10_hit`, `top100_hit`,
  `rank_score`
- `genre_similarity`, `keyword_similarity`, `overview_similarity`,
  `hybrid_similarity`
- `agent_pref_alignment`, `baseline_pref_alignment`

Open the CSV in your spreadsheet of choice to drill into any case.

---

## Customization

### Stricter or looser "soft match"

```python
validator = build_validator()
validator.SOFT_MATCH_THRESHOLD = 0.70   # default 0.60
```

### Use a different baseline pool

```python
# Baseline searches the entire CSV, not just llm.TOP_MOVIES
validator = build_validator(use_agent_pool=False)
```

### Different year filter (only matters when `use_agent_pool=False`)

```python
validator = build_validator(use_agent_pool=False, min_year=2015)
```

### Tune the baseline weights

Edit the final blend in `validation.py::BaselineRecommender.recommend`:

```python
final = (
    0.22 * keyword_match
    + 0.30 * text_sim
    + 0.18 * genre_align
    + 0.12 * tone_boost
    + 0.10 * history_sim
    + 0.08 * quality
)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'sklearn'`**
→ Re-run `pip install -r requirements.txt` in the active venv.

**`ValueError: empty vocabulary`** during TF-IDF fit
→ The baseline pool is empty. Either every movie is in watch history, or
`use_agent_pool=True` but `llm.TOP_MOVIES` hasn't been populated yet. Run
the agent once (e.g. `python -c "from llm import TOP_MOVIES"`) to verify.

**Agent always returns rank 1**
→ Likely using `--no-live` (the stub) or the baseline's candidate pool has
collapsed to a single movie. Check `validator.baseline.movies`.

**`OLLAMA_API_KEY` warning in Streamlit**
→ Expected if you haven't exported the key. The agent will use its
deterministic fallback path, which still returns a valid recommendation —
just templated.

**`Movie with TMDB ID X not found`**
→ The id you pasted is not in `tmdb_top1000_movies.csv`. Double-check the
id or widen the dataset.

---

## Custom eval case file format

Same shape as `dspy_gepa_eval_cases.json` — a JSON array of objects:

```json
[
  {
    "case_id": "user_001",
    "preferences": "I love sci-fi with great visuals",
    "history": ["The Martian", "Interstellar"],
    "history_ids": [286217, 157336]
  }
]
```

`case_id` is optional; if omitted, the runner auto-generates
`case_001`, `case_002`, ...

Pass the path to the runner:

```bash
python run_validation_example.py --cases my_cases.json
```

---

## Questions

- Inspect `validation.py` docstrings for class- and method-level details.
- See `BaselineRecommender` for scoring internals.
- See `SimilarityMetrics` for the individual similarity definitions.
- See `ValidationFramework.validate_with_agent` for the live-agent flow.
