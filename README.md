# Agentic Movie Recommender

Our team's submission for the BAMS 521 Agentic AI project. We built a **RAG + DSPy/GEPA‑tuned, preference‑first** movie recommender that runs on top of the TMDB top‑1000 CSV, with a self‑correcting LLM retry and a deterministic fallback so every call returns a valid, persuasive pick within the 20‑second budget.

The model is **`gemma4:31b-cloud`** (required by the course), served via Ollama Cloud. Descriptions are hard‑capped at **500 characters**, cleaned, and truncated at a sentence boundary.

---

## Project requirements we enforce

The grader will DQ a submission that violates any of these, so everything in this repo is designed around them:

| Requirement | Where it's enforced |
|---|---|
| `get_recommendation(preferences, history, history_ids) -> {"tmdb_id": int, "description": str}` | `llm.py::get_recommendation` + `_enforce_output_spec` |
| `tmdb_id` must be in the candidate pool (`TOP_MOVIES`, 350 rows) | Validated after the LLM call; retried if violated; fallback otherwise |
| Never recommend a movie in the user's watch history | De‑duped at retrieval, blocked in the prompt, re‑checked before return |
| Response within **20 s** | LLM primary 13 s + corrective retry 4 s + instant fallback |
| Description ≤ **500 chars** | `DESCRIPTION_MAX_CHARS = 500`, smart sentence‑boundary truncation |
| Model is `gemma4:31b-cloud` | Hard‑coded as `MODEL`; do not change |
| **Do not hard‑code API keys** | Read via `os.getenv("OLLAMA_API_KEY")`; optional TMDB keys for the offline tuning loop only |

---

## Quick start — step by step

### 1. Clone and enter the repo

```bash
git clone <repo-url> agentic-movie-recommender
cd agentic-movie-recommender
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pulls in `pandas`, `ollama`, `dspy`, `gepa`, and the FastAPI stack.

### 3. Get a free Ollama Cloud API key

Sign up at [ollama.com/settings/keys](https://ollama.com/settings/keys) and copy the key. Export it (or prefix each command):

```bash
export OLLAMA_API_KEY=your_key_here
```

### 4. Run a one‑off recommendation (interactive CLI)

```bash
python llm.py
# Preferences: I love sci-fi thrillers with smart twists
# Watch history (optional): Inception, Tenet
```

Or pass the prompt via flags — handy for scripted tests:

```bash
python llm.py \
  --preferences "Funny, light, action-packed movie." \
  --history "The Dark Knight Rises"
```

Sample output:

```
Served in 3.42s
{'tmdb_id': 1022789, 'description': "Moving away from the brooding tone of The Dark Knight Rises, Inside Out 2 delivers funny, feel-good animation ..."}
```

### 5. Run the grader's test suite

The course ships `test.py`; we run it on every change:

```bash
python test.py
```

It checks that `get_recommendation()` returns the right shape, picks a valid `tmdb_id`, never repeats history, and stays under 20 s. All tests must pass before we submit.

### 6. (Optional) Run the DSPy + GEPA tuning loop

This regenerates `dspy_gepa_best_config.json`, which `llm.py` loads automatically at import:

```bash
# minimal (local-only eval cases)
python dspy_gepa_benchmark.py --num-cases 12 --min-cases 30 --auto light

# with live TMDB augmentation (more diverse eval cases)
export TMDB_API_KEY=your_tmdb_v3_key        # or: TMDB_READ_ACCESS_TOKEN=...
python dspy_gepa_benchmark.py --num-cases 20 --min-cases 40 --auto light
```

Useful flags:

- `--auto {light,medium,heavy}` — GEPA reflection budget
- `--max-metric-calls N` / `--max-full-evals N` — explicit budget override
- `--prepare-only` — dump `dspy_gepa_eval_cases.json` without running GEPA
- `TMDB_SSL_NO_VERIFY=1` — only if your network has SSL cert‑chain issues

### 7. Submit

Zip everything except the usual junk, keep it under 10 MB:

```bash
zip -r submission.zip \
  llm.py requirements.txt README.md \
  tmdb_top1000_movies.csv \
  dspy_gepa_benchmark.py dspy_gepa_best_config.json dspy_gepa_eval_cases.json \
  -x "*.venv*" "*__pycache__*" "*.env*"
```

---

## Our approach

The big idea: treat this as a **retrieval‑augmented, preference‑first re‑ranker**. All deterministic work (filtering, scoring, validation, formatting) happens in Python; the LLM is only responsible for (a) picking one `tmdb_id` from a tight, high‑signal shortlist and (b) writing a persuasive ≤ 500‑char pitch.

### Pipeline

```
user input
   │
   ▼
┌──────────────────────────┐
│ 1. Input validation       │  strip, dedupe, type‑coerce, backfill
└──────────────────────────┘
   │
   ▼
┌──────────────────────────┐
│ 2. Preference analysis    │  genre weights, blocked genres,
│                           │  tone/mood token expansion
└──────────────────────────┘
   │
   ▼
┌──────────────────────────┐
│ 3. RAG retrieval (top 100)│  lexical + keyword + overview +
│                           │  tagline + genre + quality, with a
│                           │  conflict‑aware penalty
└──────────────────────────┘
   │
   ▼
┌──────────────────────────┐
│ 4. Hybrid rerank (top 14) │  preference > history, tone bonus,
│                           │  title match, blocked/conflict penalty
└──────────────────────────┘
   │
   ▼
┌──────────────────────────┐
│ 5. LLM selection (gemma4) │  strict JSON, banned‑phrases list,
│     13 s primary timeout  │  pivot‑clause guidance when conflict
└──────────────────────────┘
   │ invalid id / watched / empty?
   ▼
┌──────────────────────────┐
│ 6. Corrective retry (4 s) │  re‑prompt with the bad id banned
└──────────────────────────┘
   │ still bad?
   ▼
┌──────────────────────────┐
│ 7. Deterministic fallback │  5 templates × 5 hooks, hash‑selected
└──────────────────────────┘
   │
   ▼
┌──────────────────────────┐
│ 8. Sanitize + 500 char cap│  strip markdown/labels/banned phrases,
│                           │  smart sentence‑boundary truncate
└──────────────────────────┘
   │
   ▼
 dict(tmdb_id, description)
```

### Key design choices

1. **Preference outranks history.** If the user asks for *"a pure thriller"* but their history is *"a romantic thriller,"* the ranker computes `conflict_genres = history_genres − preference_genres = {Romance}` and subtracts a penalty from romance‑heavy candidates. The prompt also tells the LLM to pivot toward the preference and acknowledge the shift in one short clause of the description.
2. **Tone vocabulary.** `TONE_ALIASES` expands words like *atmospheric, slow‑burn, psychological, twist, gritty, feel‑good* into related terms matched against each candidate's keywords and overview. Lifts intent‑specific matches that pure genre‑matching would miss.
3. **Negative constraints.** Phrases like *no horror, not romance, avoid gore* are parsed into a `blocked_genres` set — hard penalty in scoring + explicit rule in the prompt.
4. **Strict, opinionated prompt.** Tells the LLM (a) the priority order, (b) the 500‑char hard cap (aim 120–380), (c) a list of banned marketing phrases (*masterpiece, breathtaking, must‑watch, tour de force, edge of your seat,* …), and (d) to emit two‑key JSON only — no reasoning, no labels, no markdown.
5. **Self‑correcting retry.** If the LLM picks an id in the watch history or outside the candidate pool, we re‑prompt once with that id added to an explicit `Do NOT pick` list and a tightened shortlist. Worst‑case LLM time: 13 s + 4 s = 17 s, inside the 20 s DQ limit.
6. **Guaranteed valid output.** A deterministic fallback composes a fluent description from the top ranked candidate using 5 opener templates × 5 mood hooks, selected by a stable `blake2b` hash of `(preferences, history_ids, tmdb_id)`. Same input → same output; different inputs → visibly different text. Also inserts the pivot clause under conflict, so even the no‑LLM path honors preference‑over‑history.
7. **Sanitizer as last line of defense.** Strips markdown, label prefixes (`Description:`, `Recommendation:`), banned phrases, orphan articles, code fences, duplicate punctuation; truncates at the nearest sentence boundary ≥ 55 % of 500 chars (word boundary / hard cut as fallbacks); always ends in `.`, `!`, or `?`.

---

## How we evaluated it

Quality was tuned against an **automated, multi‑component metric** and a **conflict‑heavy evaluation set** that mirrors the peer‑review rubric. See `dspy_gepa_benchmark.py`.

### Scoring metric

Each `(preferences, history, prediction)` triple gets a weighted score in `[0, 1]`:

| Component | Weight | What it measures |
|---|---|---|
| `genre_score` | **0.36** | Weighted genre alignment between inferred preference intent and the chosen movie. Preference‑first. |
| `quality_score` | 0.22 | Normalized blend of TMDB `vote_average`, `log(vote_count)`, `popularity`, `year`. |
| `desc_len_score` | 0.16 | Peak 1.0 at 90–380 chars; **0 above the 500‑char cap**. |
| `specificity_score` | 0.18 | Preference‑token coverage minus density of generic filler words. |
| `history_ack_bonus` | 0.08 | Rewards descriptions that explicitly pivot when history genres conflict with preferences. |
| `banned_penalty` | **−0.20** | Subtracts per banned marketing phrase the LLM emits. |

Plus two hard gates: `tmdb_id ∉ candidates → 0.0`, `tmdb_id ∈ watch_history → 0.0`.

Weights chosen to (a) make preference alignment dominate and (b) punish the two failure modes we saw most: bloated marketing prose, and ignoring user constraints in favor of the most popular candidate.

### Eval set

- **8 hand‑designed base cases** covering the main genres, a blocked‑genre case, and two history‑aware cases.
- **4 conflict cases** that stress the preference‑vs‑history logic (e.g. *"pure thriller, no romance subplot"* with a romantic‑thriller in history; *"atmospheric horror, not gore"* with a gore‑heavy history).
- **TMDB API augmentation**: when `TMDB_API_KEY` / `TMDB_READ_ACCESS_TOKEN` is set, we fetch additional popular titles and generate genre‑templated prefs + randomized watch histories for up to `--num-cases` rows. Falls back to local CSV sampling if TMDB auth is absent.

### DSPy + GEPA optimization loop

1. **Style sweep.** Evaluate four hand‑authored prompt styles (*balanced, cinematic, precision, novelty*) against the metric, using the real `get_recommendation()` pipeline; pick the best.
2. **GEPA reflection.** A DSPy `ChainOfThought` module wraps the candidate‑block → (tmdb_id, description) mapping. `dspy.GEPA` uses a high‑temperature reflection LM to iteratively rewrite the signature's instructions using the metric's structured feedback strings (*"genre=0.87, specificity=0.41, length=0.75, history_ack=0.00…"*), so it can target fixes rather than random rewrites.
3. **Persist.** The winning style + optimized instruction are written to `dspy_gepa_best_config.json`; `llm.py` loads them at import time and injects into the live prompt.

Why this works for this project:

- **Reproducible.** No hand‑graded runs; any change gets re‑scored in minutes.
- **Targets peer‑review failure modes.** Specificity up, marketing fluff down — exactly what human judges reward.
- **Self‑improving.** GEPA optimizes the instruction using the metric's feedback, so improving the metric improves the agent.

---

## Code guide

### `llm.py` (~1 k lines — the only file the grader calls)

| Section | Key symbols | Role |
|---|---|---|
| Config | `MODEL`, `LLM_PRIMARY_TIMEOUT`, `LLM_RETRY_TIMEOUT`, `DESCRIPTION_MAX_CHARS=500`, `CANDIDATES_IN_PROMPT=14`, `PROMPT_STYLES`, `BANNED_PHRASES`, `TONE_ALIASES`, `GENRE_ALIASES` | All tunables; timeouts sum to < 20 s. |
| Config loading | `_load_prompt_style`, `_load_tuned_instruction` | Reads `dspy_gepa_best_config.json`; honors env overrides `RECOMMENDER_PROMPT_STYLE`, `RECOMMENDER_TUNED_INSTRUCTION`. |
| Preference analysis | `_infer_genre_weights`, `_infer_blocked_genres`, `_infer_tone_tokens` | Free‑text preferences → structured signals. |
| Data prep | `_prepare_movie_table`, `_MOVIES`, `TOP_MOVIES` | One‑time CSV load, normalization, quality score, top‑350 pool. |
| Retrieval | `_rag_retrieve` | Top‑100 by weighted score over lexical + keyword + overview + tagline + genre + quality + conflict + blocked. |
| Reranking | `_rank_candidates` | Shortlist with preference‑first reweighting + conflict penalty. |
| Prompt | `_build_prompt`, `_candidate_block`, `_history_pairs` | Strict JSON prompt with conflict note, banned phrases, optional `banned_ids` for retries. |
| LLM | `_call_llm`, `_extract_json_payload`, `_parse_llm_output` | Ollama Cloud call with JSON mode; robust JSON extraction; post‑validation. |
| Sanitization | `_sanitize_description`, `_smart_truncate`, `_strip_banned_phrases` | Markdown/label/banned‑phrase scrubbing; sentence‑boundary truncation; ≤ 500 char cap. |
| Fallback | `_fallback_result`, `_summarize_preference`, `_FALLBACK_OPENERS`, `_FALLBACK_HOOKS` | Deterministic, varied, no‑LLM description. |
| Validation | `_validate_inputs` | Type coercion, dedupe, empty/invalid filtering, id/title alignment. |
| Entry | `get_recommendation` | Pipeline orchestrator; per‑process cache keyed on normalized inputs + style + instruction. |
| CLI | `__main__` | Local smoke test with `--preferences` / `--history`. |

### `dspy_gepa_benchmark.py`

- `BASE_DEV_CASES` — 8 base + 4 conflict eval cases.
- `_build_eval_cases` — hybrid of local CSV sampling + optional TMDB API augmentation.
- `RecommendFromRAG`, `MovieProgram` — DSPy signature + `ChainOfThought` module.
- `_metric_from_output`, `_metric` — the weighted metric used for both native‑style scoring and GEPA optimization.
- `_banned_phrase_penalty`, `_history_acknowledgement_bonus`, `_specificity_score`, `_genre_alignment_score` — metric components.
- `run()` — style sweep → GEPA compile → persist `dspy_gepa_best_config.json`.

### `test.py` (course‑provided)

- Validates shape, candidate membership, watch‑history exclusion, 20 s budget.

### `dspy_gepa_best_config.json`

- `best_style`, `best_instruction` — loaded by `llm.py` at import time and injected into the live prompt.
- `native_style_scores`, `dspy_gepa_score`, `dspy_baseline_score` — evidence of improvement over baseline.

### `tmdb_top1000_movies.csv`

- Offline TMDB dump (~1 000 movies). Columns used: `tmdb_id, title, year, genres, overview, tagline, keywords, director, top_cast, vote_average, vote_count, popularity`.

---

## Trade‑offs we made (and why)

- **No live TMDB calls at inference.** The 20 s DQ and the cold‑start cost of TMDB + LLM together made live retrieval risky. TMDB is used only in the offline GEPA loop to diversify training cases.
- **Results are cached** per process on normalized inputs → 0.1 ms repeats and deterministic outputs across calls.
- **Shortlist of 14** into the LLM. Larger shortlists hurt latency more than they helped selection quality under our metric.
- **Banned phrases.** Peer judging rewards specificity; generic marketing copy is how LLMs dilute otherwise good picks, so we strip it both in the prompt and post‑hoc.
- **Future work:** (a) plug an LLM‑as‑judge into the metric for a pairwise preference score (closer to the competition rubric); (b) learn the scoring weights from a held‑out preference‑pair dataset; (c) pre‑embed candidates with a local embedder to replace the hand‑tuned lexical RAG.
