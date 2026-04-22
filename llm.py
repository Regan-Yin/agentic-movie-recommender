"""
TODO: This is the file you should edit.

get_recommendation() is called once per request with the user's input.
It should return a dict with keys "tmdb_id" and "description".

build_prompt() and call_llm() are broken out as separate functions so they are
easy to swap or extend individually, but you are free to restructure this file
however you like.

IMPORTANT: Do NOT hard-code your API key in this file. The grader will supply
its own OLLAMA_API_KEY environment variable when running your submission. Your
code must read it from the environment (os.environ or os.getenv), not from a
string literal in the source.
"""

import json
import os
import re
import time
import argparse
import importlib
import math

ollama = importlib.import_module("ollama")
pd = importlib.import_module("pandas")

# ---------------------------------------------------------------------------
# TODO: Edit these to improve your recommendations
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TUNED_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "dspy_gepa_best_config.json")
_ALL_MOVIES = pd.read_csv(DATA_PATH)

PROMPT_STYLES = {
    "balanced": "Be concise, specific, and grounded in concrete movie attributes.",
    "cinematic": "Use vivid cinematic language while staying factual and avoiding spoilers.",
    "precision": "Prioritize tight preference matching and explicit evidence from the candidate context.",
    "novelty": "Favor fresh yet relevant options and explain what feels new versus watch history.",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9']+", _normalize_text(text)) if len(tok) >= 3}


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_prompt_style() -> str:
    style = "balanced"
    try:
        if os.path.exists(TUNED_CONFIG_PATH):
            with open(TUNED_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            maybe_style = str(data.get("best_style", "")).strip().lower()
            if maybe_style in PROMPT_STYLES:
                style = maybe_style
    except Exception:
        style = "balanced"
    env_style = str(os.getenv("RECOMMENDER_PROMPT_STYLE", "")).strip().lower()
    if env_style in PROMPT_STYLES:
        style = env_style
    return style


def _load_tuned_instruction() -> str:
    instruction = ""
    try:
        if os.path.exists(TUNED_CONFIG_PATH):
            with open(TUNED_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            maybe = str(data.get("best_instruction", "")).strip()
            if maybe:
                instruction = maybe
    except Exception:
        instruction = ""

    env_instruction = str(os.getenv("RECOMMENDER_TUNED_INSTRUCTION", "")).strip()
    if env_instruction:
        instruction = env_instruction

    return instruction[:3000]


def _infer_genre_weights(preferences: str) -> dict[str, float]:
    pref = _normalize_text(preferences)
    genre_weights: dict[str, float] = {}

    def add_weight(genres: list[str], weight: float) -> None:
        for g in genres:
            genre_weights[g] = genre_weights.get(g, 0.0) + weight

    if any(w in pref for w in ["action", "fight", "adventure", "explosive"]):
        add_weight(["Action", "Adventure", "Thriller"], 1.0)
    if any(w in pref for w in ["superhero", "comic", "marvel", "dc"]):
        add_weight(["Action", "Adventure", "Science Fiction", "Fantasy"], 1.1)
    if any(w in pref for w in ["funny", "comedy", "laugh", "light", "feel good", "feel-good"]):
        add_weight(["Comedy", "Family", "Adventure"], 1.0)
    if any(w in pref for w in ["romance", "romantic", "love"]):
        add_weight(["Romance", "Drama"], 1.0)
    if any(w in pref for w in ["horror", "scary", "creepy", "slasher"]):
        add_weight(["Horror", "Mystery", "Thriller"], 1.0)
    if any(w in pref for w in ["mystery", "detective", "whodunit", "investigation"]):
        add_weight(["Mystery", "Crime", "Thriller"], 1.0)
    if any(w in pref for w in ["crime", "heist", "gangster", "underworld"]):
        add_weight(["Crime", "Thriller", "Action"], 1.0)
    if any(w in pref for w in ["sci fi", "sci-fi", "science fiction", "space", "alien", "future", "ai"]):
        add_weight(["Science Fiction", "Adventure", "Thriller"], 1.0)
    if any(w in pref for w in ["animation", "animated", "kids", "family"]):
        add_weight(["Animation", "Family", "Comedy", "Adventure"], 1.0)
    if any(w in pref for w in ["drama", "emotional", "character-driven"]):
        add_weight(["Drama"], 0.8)

    return genre_weights


def _infer_blocked_genres(preferences: str) -> set[str]:
    pref = _normalize_text(preferences)
    blocked = set()
    genre_aliases = {
        "Action": ["action"],
        "Adventure": ["adventure"],
        "Comedy": ["comedy", "funny"],
        "Crime": ["crime"],
        "Drama": ["drama"],
        "Family": ["family", "kids"],
        "Fantasy": ["fantasy"],
        "Horror": ["horror", "scary"],
        "Mystery": ["mystery"],
        "Romance": ["romance", "romantic"],
        "Science Fiction": ["science fiction", "sci-fi", "sci fi"],
        "Thriller": ["thriller"],
        "Animation": ["animation", "animated"],
    }
    prefixes = ["no ", "not ", "without ", "avoid "]
    for genre, aliases in genre_aliases.items():
        if any(any((p + a) in pref for p in prefixes) for a in aliases):
            blocked.add(genre)
    return blocked


def _prepare_movie_table(df: pd.DataFrame) -> pd.DataFrame:
    movies = df.copy()
    for col in ["title", "genres", "keywords", "overview", "tagline", "director", "top_cast"]:
        movies[col] = movies[col].fillna("")

    for col in ["vote_average", "vote_count", "popularity", "year"]:
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0)

    movies["tmdb_id"] = pd.to_numeric(movies["tmdb_id"], errors="coerce").fillna(0).astype(int)
    movies = movies[movies["tmdb_id"] > 0].copy()

    movies["title_norm"] = movies["title"].map(_normalize_text)
    movies["genres_set"] = movies["genres"].map(lambda s: {g.strip() for g in str(s).split(",") if g.strip()})
    movies["keywords_set"] = movies["keywords"].map(_tokenize)
    movies["token_set"] = (
        movies["title"]
        + " "
        + movies["genres"]
        + " "
        + movies["keywords"]
        + " "
        + movies["overview"]
        + " "
        + movies["tagline"]
        + " "
        + movies["director"]
        + " "
        + movies["top_cast"]
    ).map(_tokenize)

    vote_avg_n = (movies["vote_average"] - movies["vote_average"].min()) / (
        (movies["vote_average"].max() - movies["vote_average"].min()) or 1.0
    )
    vote_count_n = movies["vote_count"].map(lambda x: math.log1p(max(float(x), 1.0)))
    vote_count_n = vote_count_n / (vote_count_n.max() or 1.0)
    pop_n = (movies["popularity"] - movies["popularity"].min()) / (
        (movies["popularity"].max() - movies["popularity"].min()) or 1.0
    )
    year_n = (movies["year"] - movies["year"].min()) / ((movies["year"].max() - movies["year"].min()) or 1.0)

    movies["quality_score"] = 0.42 * vote_avg_n + 0.33 * vote_count_n + 0.17 * pop_n + 0.08 * year_n
    return movies


_MOVIES = _prepare_movie_table(_ALL_MOVIES)

# Keep a broad but quality-leaning candidate pool for better recommendation quality.
TOP_MOVIES = _MOVIES.nlargest(350, ["vote_count", "popularity", "vote_average"]).copy()

_CACHE: dict[tuple[str, tuple[int, ...], tuple[str, ...]], dict] = {}
_ACTIVE_PROMPT_STYLE = _load_prompt_style()
_ACTIVE_TUNED_INSTRUCTION = _load_tuned_instruction()


def _sanitize_description(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[:500]


def _enforce_output_spec(tmdb_id: int, description: str) -> dict:
    # Return exactly two keys as required by llm.py output specification.
    return {
        "tmdb_id": int(tmdb_id),
        "description": _sanitize_description(description),
    }


def _rag_retrieve(preferences: str, history: list[str], history_ids: list[int], top_k: int = 90) -> pd.DataFrame:
    pref_text = _normalize_text(preferences)
    pref_tokens = _tokenize(pref_text)
    pref_genre_weights = _infer_genre_weights(preferences)

    history_id_set = {_safe_int(i) for i in history_ids if _safe_int(i) > 0}
    history_titles_norm = {_normalize_text(t) for t in history if _normalize_text(t)}

    candidates = TOP_MOVIES[
        (~TOP_MOVIES["tmdb_id"].isin(history_id_set))
        & (~TOP_MOVIES["title_norm"].isin(history_titles_norm))
    ].copy()
    if candidates.empty:
        candidates = TOP_MOVIES.copy()

    history_rows = _MOVIES[_MOVIES["tmdb_id"].isin(history_id_set)]
    history_tokens: set[str] = set()
    for row in history_rows.itertuples():
        history_tokens |= row.token_set

    expanded_tokens = set(pref_tokens)
    for g in pref_genre_weights:
        expanded_tokens |= _tokenize(g)

    retrieve_scores = []
    norm = max(6, len(expanded_tokens))
    for row in candidates.itertuples():
        lexical = len(expanded_tokens & row.token_set) / norm
        keyword = len(expanded_tokens & row.keywords_set) / norm

        genre_bonus = 0.0
        for genre, weight in pref_genre_weights.items():
            if genre in row.genres_set:
                genre_bonus += weight

        history_penalty = 0.0
        if history_tokens:
            history_penalty = len(history_tokens & row.token_set) / max(12, len(row.token_set))

        score = 0.72 * lexical + 0.45 * keyword + 0.32 * genre_bonus + 0.38 * float(row.quality_score) - 0.14 * history_penalty
        retrieve_scores.append(score)

    candidates["rag_score"] = retrieve_scores
    return candidates.sort_values("rag_score", ascending=False).head(top_k).copy()


def _rank_candidates(preferences: str, history: list[str], history_ids: list[int]) -> pd.DataFrame:
    pref_text = _normalize_text(preferences)
    pref_tokens = _tokenize(pref_text)
    pref_genre_weights = _infer_genre_weights(preferences)
    blocked_genres = _infer_blocked_genres(preferences)

    candidates = _rag_retrieve(preferences, history, history_ids, top_k=90)
    if candidates.empty:
        candidates = TOP_MOVIES.copy()

    history_id_set = {_safe_int(i) for i in history_ids if _safe_int(i) > 0}

    history_rows = _MOVIES[_MOVIES["tmdb_id"].isin(history_id_set)]
    history_genres: set[str] = set()
    for row in history_rows.itertuples():
        history_genres.update(row.genres_set)

    scored = []
    pref_token_norm = max(6, len(pref_tokens))
    for row in candidates.itertuples():
        token_overlap = len(pref_tokens & row.token_set) / pref_token_norm

        genre_match = 0.0
        for g, w in pref_genre_weights.items():
            if g in row.genres_set:
                genre_match += w

        blocked_penalty = 1.0 if any(g in row.genres_set for g in blocked_genres) else 0.0

        history_overlap = 0.0
        if history_genres:
            history_overlap = len(row.genres_set & history_genres) / max(1, len(row.genres_set | history_genres))

        title_bonus = 0.12 if any(tok in row.title_norm for tok in pref_tokens) else 0.0
        keyword_bonus = len(pref_tokens & row.keywords_set) / pref_token_norm

        score = (
            0.58 * float(row.quality_score)
            + 0.34 * float(getattr(row, "rag_score", 0.0))
            + 1.05 * token_overlap
            + 0.26 * genre_match
            + 0.30 * keyword_bonus
            + title_bonus
            - 0.28 * blocked_penalty
            - 0.08 * history_overlap
        )
        scored.append(score)

    candidates["hybrid_score"] = scored
    candidates = candidates.sort_values("hybrid_score", ascending=False)
    return candidates


def _build_prompt(
    preferences: str,
    history: list[str],
    history_ids: list[int],
    ranked: pd.DataFrame,
    style_name: str,
    tuned_instruction: str,
) -> str:
    short_list = ranked.head(12)
    style = PROMPT_STYLES.get(style_name, PROMPT_STYLES["balanced"])
    history_pairs = []
    for idx, title in enumerate(history):
        hid = history_ids[idx] if idx < len(history_ids) else None
        if hid:
            history_pairs.append(f'"{title}" (tmdb_id={hid})')
        else:
            history_pairs.append(f'"{title}"')
    history_text = ", ".join(history_pairs) if history_pairs else "none"

    rows = []
    for row in short_list.itertuples():
        overview = str(row.overview)[:180].replace("\n", " ")
        rows.append(
            "- "
            f"tmdb_id={row.tmdb_id} | title=\"{row.title}\" | year={_safe_int(row.year)} | "
            f"genres={row.genres} | rating={float(row.vote_average):.2f} | "
            f"director={row.director} | cast={str(row.top_cast)[:90]} | "
            f"rag_score={float(getattr(row, 'rag_score', 0.0)):.3f} | overview={overview}"
        )

    candidate_text = "\n".join(rows)
    tuning_block = ""
    if tuned_instruction:
        tuning_block = f"\nOptimization instruction (from DSPy GEPA):\n{tuned_instruction}\n"

    return f"""You are an expert movie recommender.

Task:
Choose exactly ONE movie from the candidate list and return strict JSON only.
You are using RAG context: higher rag_score means stronger retrieval relevance.

User preferences:
{preferences}

Watch history (never recommend these):
{history_text}

Candidate movies:
{candidate_text}
{tuning_block}

Hard rules:
1) Return ONLY valid JSON with keys: tmdb_id (int), description (string).
2) tmdb_id must be one from the candidate list and must not be in watch history.
3) description must be <= 500 characters, vivid and specific.
4) Style guide: {style}
5) Output must contain EXACTLY two keys: tmdb_id and description. Do not include reasoning or any extra key.

Output format:
{{
  "tmdb_id": 123,
  "description": "..."
}}"""


def _extract_json_payload(raw_text: str) -> dict:
    text = str(raw_text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty", "", 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_result(preferences: str, ranked: pd.DataFrame) -> dict:
    best = ranked.iloc[0]
    desc = (
        f"Try {best.title} ({_safe_int(best.year)}): it blends {best.genres.lower()} with a "
        f"strong audience track record, and matches your request for {preferences.strip()[:120]}"
    )
    return _enforce_output_spec(int(best.tmdb_id), desc)


def _call_llm(prompt: str) -> dict:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set")

    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=7,
    )
    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You output strict JSON only. No markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.3, "top_p": 0.9, "num_predict": 120},
        format="json",
    )

    content = ""
    if isinstance(response, dict):
        content = str(response.get("message", {}).get("content", ""))
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
    return _extract_json_payload(content)


def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    clean_pref = str(preferences or "").strip()
    clean_history = [str(h).strip() for h in (history or []) if str(h).strip()]
    clean_ids = [_safe_int(i) for i in (history_ids or []) if _safe_int(i) > 0]

    env_style = str(os.getenv("RECOMMENDER_PROMPT_STYLE", "")).strip().lower()
    style_name = env_style if env_style in PROMPT_STYLES else _ACTIVE_PROMPT_STYLE
    tuned_instruction = str(os.getenv("RECOMMENDER_TUNED_INSTRUCTION", "")).strip() or _ACTIVE_TUNED_INSTRUCTION

    cache_key = (
        _normalize_text(clean_pref),
        tuple(sorted(clean_ids)),
        tuple(sorted(_normalize_text(h) for h in clean_history)),
        style_name,
        _normalize_text(tuned_instruction),
    )
    if cache_key in _CACHE:
        return dict(_CACHE[cache_key])

    ranked = _rank_candidates(clean_pref, clean_history, clean_ids)
    if ranked.empty:
        raise RuntimeError("No candidates available for recommendation")

    valid_ids = set(ranked["tmdb_id"].astype(int).tolist())
    watched_ids = set(clean_ids)

    prompt = _build_prompt(clean_pref, clean_history, clean_ids, ranked, style_name, tuned_instruction)

    try:
        llm_result = _call_llm(prompt)
        tmdb_id = _safe_int(llm_result.get("tmdb_id"), default=-1)
        description = str(llm_result.get("description", "")).strip()

        if tmdb_id not in valid_ids or tmdb_id in watched_ids:
            raise ValueError("LLM selected an invalid or watched tmdb_id")

        if not description:
            raise ValueError("LLM returned empty description")

        result = _enforce_output_spec(tmdb_id, description)
    except Exception:
        result = _fallback_result(clean_pref, ranked)

    _CACHE[cache_key] = dict(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local movie recommendation test."
    )
    parser.add_argument(
        "--preferences",
        type=str,
        help="User preferences text. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--history",
        type=str,
        help='Comma-separated watch history titles. Example: "The Avengers, Up"',
    )
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")
    print(
        "For watch history, enter comma-separated movie titles (or leave blank)."
    )

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = (
        [t.strip() for t in history_raw.split(",") if t.strip()]
        if history_raw
        else []
    )

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    print(result)
    elapsed = time.perf_counter() - start

    print(f"\nServed in {elapsed:.2f}s")

