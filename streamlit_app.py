"""
Streamlit app for interactively testing our movie recommendation agent
against the independent TF-IDF baseline.

Two modes:
  1) LIVE AGENT  — you enter preferences + history; we call
     `llm.get_recommendation()` (requires OLLAMA_API_KEY to be set, otherwise
     the agent falls back to its deterministic template path).
  2) MANUAL ID   — you paste the TMDB ID your agent recommended; we rank it
     against the baseline's hidden top-100 candidates.

Scoring:
  Rank 1  -> 100%    Rank 10 -> 91%     ...    Rank 100 -> 1%
  Outside top-100 -> 0%

The hidden top-100 is drawn from the SAME candidate pool the agent uses
(`llm.TOP_MOVIES`) so the comparison is fair.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from validation import (
    BaselineRecommender,
    MovieDataset,
    SimilarityMetrics,
    ValidationFramework,
    default_candidate_ids,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation Agent Tester",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
    .score-excellent { color: #05ab07; font-size: 2em; font-weight: bold; }
    .score-good      { color: #1f77b4; font-size: 2em; font-weight: bold; }
    .score-fair      { color: #ff7f0e; font-size: 2em; font-weight: bold; }
    .score-poor      { color: #d62728; font-size: 2em; font-weight: bold; }
    .pill { display:inline-block; padding:2px 10px; margin-right:6px; border-radius:999px;
            background:#eef; color:#335; font-size:0.85em; }
    .desc-box { background:#fafafa; border-left:4px solid #1f77b4; padding:12px 16px;
                border-radius:6px; font-size:1.02em; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_resources():
    """Load CSV, build baseline restricted to the agent's candidate pool."""
    data_path = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
    if not os.path.exists(data_path):
        st.error(f"❌ Dataset not found at {data_path}")
        st.stop()
    df = load_dataset(data_path)
    cand_ids = default_candidate_ids()
    baseline = BaselineRecommender(df, candidate_ids=cand_ids or None, min_year=2010)
    dataset = MovieDataset.__new__(MovieDataset)
    dataset.df = df.copy()
    dataset._preprocess()
    return df, baseline, dataset, cand_ids


@st.cache_resource(show_spinner=False)
def _get_agent_callable():
    """Import the live agent once. We keep it cached so the CSV load is paid once."""
    from llm import get_recommendation  # heavy import: pandas + ollama
    return get_recommendation


movies_df, baseline, dataset, candidate_ids = _load_resources()

# Title -> tmdb_id lookup used by the watch-history picker
_title_to_id = dict(zip(movies_df["title"].astype(str), movies_df["tmdb_id"].astype(int)))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎬 Movie Recommendation Agent Tester")
st.markdown(
    """
Compare our LLM-based recommendation agent against an independent TF-IDF
baseline. The baseline is **intent-aware** — it weights the explicit terms
in your prompt (genre, mood, tone) heavily, so it's a tough opponent.

**Two workflows:**
1. **Live Agent** — enter your preferences, we call `llm.get_recommendation()`
   with your inputs and score the result automatically.
2. **Manual** — paste a TMDB ID (e.g. from another run of the agent) and see
   where it ranks against the baseline's hidden top-100.
"""
)

st.markdown(
    f"<span class='pill'>Baseline pool: {len(baseline.movies)} movies</span>"
    f"<span class='pill'>Agent pool: {len(candidate_ids) if candidate_ids else 'llm.TOP_MOVIES'}</span>"
    f"<span class='pill'>Agent timeout: 20s</span>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📋 Your Inputs")

    mode = st.radio(
        "Mode",
        options=["Live Agent", "Manual TMDB ID"],
        index=0,
        help="Live calls llm.get_recommendation(). Manual accepts a pre-computed tmdb_id.",
    )

    preferences = st.text_area(
        "What kind of movie are you looking for?",
        placeholder="e.g., 'I want a pure thriller with real tension and no romance subplot'",
        height=110,
        value=st.session_state.get("preferences_default", ""),
    )

    all_titles = sorted({t for t in movies_df["title"].astype(str).tolist() if t and t != "Unknown"})
    watched_titles: List[str] = st.multiselect(
        "Movies you've already watched:",
        options=all_titles,
        default=[],
        placeholder="Select movies (optional)",
    )

    watch_history_ids = [int(_title_to_id[t]) for t in watched_titles if t in _title_to_id]

    st.divider()
    has_api_key = bool(os.environ.get("OLLAMA_API_KEY"))
    if mode == "Live Agent":
        if has_api_key:
            st.success("OLLAMA_API_KEY is set — agent will use Ollama.")
        else:
            st.warning(
                "OLLAMA_API_KEY is NOT set. The agent will fall through to its deterministic "
                "fallback. You can still compare against the baseline, but descriptions will "
                "look templated."
            )


# ---------------------------------------------------------------------------
# Baseline top-5 preview
# ---------------------------------------------------------------------------

if not preferences.strip():
    st.info("👈 Enter your preferences in the sidebar to start.")
    st.stop()

st.header("📊 Baseline Top-5 Recommendations")
st.caption("Intent-aware TF-IDF baseline, restricted to the agent's candidate pool for fairness.")

baseline_full = baseline.recommend(
    preferences=preferences,
    watch_history=watched_titles,
    watch_history_ids=watch_history_ids,
    top_k=100,
)
baseline_top5 = baseline_full[:5]
if not baseline_full:
    st.error("❌ No candidates available — check your inputs (all movies may be in watch history).")
    st.stop()

recs_rows = []
for rank, (tid, title, score) in enumerate(baseline_top5, 1):
    info = movies_df[movies_df["tmdb_id"] == tid].iloc[0]
    recs_rows.append(
        {
            "Rank": rank,
            "Movie": title,
            "TMDB ID": int(tid),
            "Year": int(info["year"]) if pd.notna(info["year"]) else "N/A",
            "Genres": str(info["genres"]),
            "Score": f"{score:.3f}",
        }
    )
st.dataframe(pd.DataFrame(recs_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Agent recommendation (live or manual)
# ---------------------------------------------------------------------------

st.header("🤖 Agent Recommendation")

agent_tmdb_id: Optional[int] = None
agent_description: str = ""
agent_latency_s: Optional[float] = None
agent_error: Optional[str] = None

if mode == "Live Agent":
    run_clicked = st.button("Run the agent on my inputs", type="primary", use_container_width=True)
    if run_clicked:
        try:
            agent_callable = _get_agent_callable()
            with st.spinner("Calling llm.get_recommendation() ..."):
                start = time.perf_counter()
                out = agent_callable(preferences, list(watched_titles), list(watch_history_ids))
                agent_latency_s = time.perf_counter() - start
            try:
                agent_tmdb_id = int(out.get("tmdb_id"))
                agent_description = str(out.get("description", ""))
            except Exception as exc:
                agent_error = f"Agent returned invalid shape: {exc!r} (raw={out!r})"
        except Exception:
            agent_error = "Agent raised an exception:\n\n" + traceback.format_exc()
else:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        raw_id = st.text_input("TMDB ID from your agent run:", placeholder="e.g., 945961")
    with col_b:
        agent_description = st.text_area(
            "Optional: paste the agent's description (to check ≤ 500 chars)",
            placeholder="(optional)",
            height=80,
        )
    if raw_id.strip():
        try:
            agent_tmdb_id = int(raw_id.strip())
        except ValueError:
            agent_error = "TMDB ID must be a number."

# ---------------------------------------------------------------------------
# Render agent result + metrics
# ---------------------------------------------------------------------------

if agent_error:
    st.error(agent_error)
    st.stop()

if agent_tmdb_id is None:
    st.info("Run the agent (or enter a TMDB ID) to see the comparison.")
    st.stop()

agent_rows = movies_df[movies_df["tmdb_id"] == agent_tmdb_id]
if agent_rows.empty:
    st.error(f"❌ Movie with TMDB ID {agent_tmdb_id} not found in dataset.")
    st.stop()
agent_row = agent_rows.iloc[0]

st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.write("**Agent picked:**")
    st.write(f"🎬 **{agent_row['title']}**  ·  TMDB `{int(agent_row['tmdb_id'])}`")
with col2:
    st.write("**Year:**")
    st.write(int(agent_row["year"]) if pd.notna(agent_row["year"]) else "N/A")
with col3:
    st.write("**Genres:**")
    st.write(str(agent_row["genres"]))

if agent_description:
    st.markdown(
        f"<div class='desc-box'>{agent_description}</div>",
        unsafe_allow_html=True,
    )
    desc_len = len(agent_description)
    within = desc_len <= 500
    st.caption(f"Description length: {desc_len} chars {'✅ (≤ 500)' if within else '❌ (> 500)'}")
if agent_latency_s is not None:
    within_budget = agent_latency_s <= 20.0
    st.caption(
        f"Agent latency: {agent_latency_s:.2f}s "
        + ("✅ (≤ 20s)" if within_budget else "❌ (> 20s test limit)")
    )

# -- Rank-based accuracy score against hidden top-100 ---------------------

st.markdown("---")
st.header("📈 Accuracy Score (Baseline Rank)")

agent_rank = next((i + 1 for i, r in enumerate(baseline_full) if r[0] == agent_tmdb_id), None)

if agent_rank is not None:
    accuracy_score = max(0, 101 - agent_rank)
    if accuracy_score >= 90:
        score_class, emoji = "score-excellent", "🎯"
    elif accuracy_score >= 80:
        score_class, emoji = "score-good", "✓"
    elif accuracy_score >= 70:
        score_class, emoji = "score-fair", "~"
    else:
        score_class, emoji = "score-poor", "⚠"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Baseline Rank",
            f"#{agent_rank} of 100",
            delta="Top-1!" if agent_rank == 1 else f"-{agent_rank - 1} positions",
        )
    with m2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;">
              <div style="font-size: 0.9em; color: #666;">Accuracy Score</div>
              <div class="{score_class}">{accuracy_score:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.metric("Status", emoji, delta=f"rank #{agent_rank} in hidden top-100")
else:
    st.error(
        f"❌ Agent's pick **{agent_row['title']}** is NOT in the baseline's hidden top-100. "
        "Accuracy Score: 0%."
    )
    st.caption(
        "This can mean the agent is weighting signals very differently from the baseline, "
        "or that the pick is outside the `llm.TOP_MOVIES` pool (unusual)."
    )


# -- Detailed similarity breakdown vs. baseline top-1 ----------------------

st.markdown("---")
st.header("🧪 Similarity Breakdown vs. Baseline Top-1")

b_tid, b_title, b_score = baseline_top5[0]
b_info = movies_df[movies_df["tmdb_id"] == b_tid].iloc[0]

genre_sim = SimilarityMetrics.genre_similarity(str(b_info["genres"]), str(agent_row["genres"]))
keyword_sim = SimilarityMetrics.keyword_similarity(str(b_info.get("keywords", "")), str(agent_row.get("keywords", "")))
overview_sim = SimilarityMetrics.overview_similarity(str(b_info["overview"]), str(agent_row["overview"]))
hybrid = (
    0.40 * overview_sim
    + 0.25 * genre_sim
    + 0.25 * keyword_sim
    + 0.10 * (1.0 if (genre_sim > 0.3 or keyword_sim > 0.3) else 0.0)
)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Genre Overlap",   f"{genre_sim:.2f}")
s2.metric("Keyword Overlap", f"{keyword_sim:.2f}")
s3.metric("Overview Sim",    f"{overview_sim:.2f}")
s4.metric("Hybrid",          f"{hybrid:.2f}", delta="soft match" if hybrid >= 0.6 else "divergent")

cb, ca = st.columns(2)
with cb:
    st.write("**Baseline Top-1:**")
    st.write(f"🏆 {b_title}")
    st.caption(f"Score: {b_score:.3f}  ·  Genres: {b_info['genres']}")
with ca:
    st.write("**Agent's Pick:**")
    st.write(f"🎯 {agent_row['title']}" if agent_rank == 1 else f"#{agent_rank} {agent_row['title']}")
    st.caption(f"Genres: {agent_row['genres']}")


# -- Full top-5 ranked list with agent highlight ---------------------------

st.markdown("---")
st.subheader("All Top-5 Options Ranked")
lines = []
for rank, (tid, title, score) in enumerate(baseline_top5, 1):
    prefix = "🎯" if rank == agent_rank else f"#{rank}"
    tail = " ← Agent picked this" if rank == agent_rank else ""
    lines.append(f"{prefix} **{title}** (Score: {score:.3f}){tail}")
st.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Footer docs
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
### How the score works

| Rank | Score | Interpretation |
|------|-------|----------------|
| #1   | 100%  | Perfect match against baseline's top pick |
| #2   | 99%   | Essentially optimal |
| #10  | 91%   | Still in the front of the pack |
| #100 | 1%    | Edge of the hidden top-100 |
| >100 | 0%    | Diverging from baseline's ranking |

### How the baseline decides

It blends five signals, all restricted to the **same candidate pool the agent uses** (`llm.TOP_MOVIES`):

- **30%** overview/keywords/genres TF-IDF similarity to your preference text
- **22%** explicit term matches (genre keywords, intent vocabulary)
- **18%** genre alignment
- **12%** tone/mood matching (*dark*, *slow-burn*, *atmospheric*, ...)
- **10%** watch-history similarity (light signal only)
- **8%**  quality (vote average + vote count)
- `-35%` penalty if a blocked genre shows up (*"no horror"*, *"not romance"* ...)

### Why run this in addition to DSPy/GEPA?

- **GEPA** tunes our agent's prompt via programmatic feedback on a task-specific metric.
- **This validator** compares the tuned agent against a totally different methodology.
  High agreement = the agent is behaving reasonably in information-retrieval terms.
  Disagreement = an interesting case worth inspecting individually.
"""
)
