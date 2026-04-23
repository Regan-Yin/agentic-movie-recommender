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

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import time

ollama = importlib.import_module("ollama")
pd = importlib.import_module("pandas")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TUNED_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "dspy_gepa_best_config.json")

# test.py enforces a 20s per-call budget. Give the LLM a generous primary
# window and keep a small retry slot for self-correction before we fall back.
LLM_PRIMARY_TIMEOUT = 13
LLM_RETRY_TIMEOUT = 4
DESCRIPTION_MAX_CHARS = 500
DESCRIPTION_TARGET_MIN = 90
CANDIDATES_IN_PROMPT = 14

PROMPT_STYLES = {
    "balanced": "Be concise, specific, and grounded in concrete movie attributes.",
    "cinematic": "Use vivid cinematic language while staying factual and avoiding spoilers.",
    "precision": "Prioritize tight preference matching and explicit evidence from the candidate context.",
    "novelty": "Favor fresh yet relevant options and explain what feels new versus watch history.",
}

# Phrases the recommender should not emit in descriptions. Used for both the
# LLM prompt (as a banned list) and post-hoc sanitization.
BANNED_PHRASES = (
    "masterpiece",
    "breathtaking",
    "must-watch",
    "must watch",
    "essential watch",
    "exceptional choice",
    "the perfect choice",
    "masterclass",
    "stunning visual journey",
    "complete change of pace",
    "edge of your seat",
    "tour de force",
    "heart-pounding",
    "cinematic experience",
)

# Expanded tone/mood vocabulary. Hitting any of these tokens in the preference
# should boost candidates whose keywords/overview mention related terms.
TONE_ALIASES: dict[str, tuple[str, ...]] = {
    "atmosphere": ("atmospheric", "dread", "eerie", "haunting", "moody", "unsettling"),
    "psychological": ("psychological", "mind", "identity", "subconscious", "trauma"),
    "slow-burn": ("slow burn", "slow-burn", "simmering", "methodical", "deliberate"),
    "twist": ("twist", "twists", "reveal", "whodunit", "subversive"),
    "gritty": ("gritty", "raw", "underworld", "brutal", "seedy"),
    "witty": ("witty", "clever", "sharp", "snappy", "quick-witted"),
    "feel-good": ("uplifting", "hopeful", "heartwarming", "cheerful", "charming"),
    "emotional": ("emotional", "heartfelt", "poignant", "tender", "bittersweet"),
    "epic": ("epic", "grand", "sweeping", "spectacle", "ambitious"),
    "visual": ("visual", "stylized", "cinematography", "neon", "painterly"),
    "action": ("high-octane", "action-packed", "explosive", "kinetic"),
    "dark": ("dark", "brooding", "ominous", "bleak"),
    "stylish": ("stylish", "sleek", "noir", "sophisticated"),
    "tense": ("tense", "suspenseful", "nerve", "high-stakes"),
    "weird": ("weird", "surreal", "offbeat", "absurd"),
}

# Canonical genre aliases (used for blocking and genre inference).
GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "Action": ("action", "fight", "combat", "shootout"),
    "Adventure": ("adventure", "quest", "expedition"),
    "Animation": ("animation", "animated", "anime"),
    "Comedy": ("comedy", "funny", "humor", "humorous", "hilarious", "feel good", "feel-good"),
    "Crime": ("crime", "heist", "gangster", "mob"),
    "Drama": ("drama", "dramatic", "character-driven", "character driven"),
    "Family": ("family", "kids", "kid-friendly"),
    "Fantasy": ("fantasy", "magical", "magic"),
    "Horror": ("horror", "scary", "slasher", "creepy", "terrifying"),
    "Mystery": ("mystery", "whodunit", "detective", "investigation"),
    "Romance": ("romance", "romantic", "love story"),
    "Science Fiction": ("science fiction", "sci-fi", "sci fi", "space", "alien", "cyberpunk", "dystopian"),
    "Thriller": ("thriller", "suspense"),
    "War": ("war", "military", "wartime"),
    "Western": ("western", "cowboy", "frontier"),
    "Music": ("music", "musical", "band", "concert"),
    "Documentary": ("documentary", "docu", "nonfiction"),
    "History": ("history", "historical", "period piece"),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9']+", _normalize_text(text)) if len(tok) >= 3}


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _stable_hash(parts: tuple) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.blake2b(raw, digest_size=8).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Config / tuned instruction loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Preference analysis
# ---------------------------------------------------------------------------


def _infer_genre_weights(preferences: str) -> dict[str, float]:
    pref = _normalize_text(preferences)
    weights: dict[str, float] = {}

    def add(genres: list[str], weight: float) -> None:
        for g in genres:
            weights[g] = weights.get(g, 0.0) + weight

    if any(w in pref for w in ["action", "fight", "adventure", "explosive"]):
        add(["Action", "Adventure", "Thriller"], 1.0)
    if any(w in pref for w in ["superhero", "comic", "marvel", "dc"]):
        add(["Action", "Adventure", "Science Fiction", "Fantasy"], 1.1)
    if any(w in pref for w in ["funny", "comedy", "laugh", "light", "feel good", "feel-good"]):
        add(["Comedy", "Family", "Adventure"], 1.0)
    if any(w in pref for w in ["romance", "romantic", "love story"]):
        add(["Romance", "Drama"], 1.0)
    if any(w in pref for w in ["horror", "scary", "creepy", "slasher", "dread"]):
        add(["Horror", "Mystery", "Thriller"], 1.0)
    if any(w in pref for w in ["mystery", "detective", "whodunit", "investigation"]):
        add(["Mystery", "Crime", "Thriller"], 1.0)
    if any(w in pref for w in ["crime", "heist", "gangster", "underworld"]):
        add(["Crime", "Thriller", "Action"], 1.0)
    if any(w in pref for w in ["sci fi", "sci-fi", "science fiction", "space", "alien", "future", "ai ", " ai"]):
        add(["Science Fiction", "Adventure", "Thriller"], 1.0)
    if any(w in pref for w in ["animation", "animated", "kids", "family"]):
        add(["Animation", "Family", "Comedy", "Adventure"], 1.0)
    if any(w in pref for w in ["drama", "emotional", "character-driven", "character driven"]):
        add(["Drama"], 0.8)
    if any(w in pref for w in ["thriller", "suspense", "tense"]):
        add(["Thriller", "Mystery"], 1.0)
    if any(w in pref for w in ["war", "military"]):
        add(["War", "Action", "Drama"], 0.9)
    if any(w in pref for w in ["western", "cowboy", "frontier"]):
        add(["Western"], 1.0)
    if any(w in pref for w in ["fantasy", "magical", "sword and sorcery"]):
        add(["Fantasy", "Adventure"], 1.0)

    return weights


def _infer_blocked_genres(preferences: str) -> set[str]:
    pref = _normalize_text(preferences)
    blocked: set[str] = set()
    prefixes = ("no ", "not ", "without ", "avoid ", "except ", "skip ")
    for genre, aliases in GENRE_ALIASES.items():
        for alias in aliases:
            if any((p + alias) in pref for p in prefixes):
                blocked.add(genre)
                break
    return blocked


def _infer_tone_tokens(preferences: str) -> set[str]:
    pref = _normalize_text(preferences)
    tokens: set[str] = set()
    for key, aliases in TONE_ALIASES.items():
        if key in pref or any(a in pref for a in aliases):
            tokens.add(key)
            for a in aliases:
                tokens |= _tokenize(a)
    return tokens


# ---------------------------------------------------------------------------
# Movie table preparation
# ---------------------------------------------------------------------------


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
    movies["overview_set"] = movies["overview"].map(_tokenize)
    movies["tagline_set"] = movies["tagline"].map(_tokenize)
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


_ALL_MOVIES = pd.read_csv(DATA_PATH)
_MOVIES = _prepare_movie_table(_ALL_MOVIES)

# Keep a broad but quality-leaning candidate pool for better recommendation quality.
TOP_MOVIES = _MOVIES.nlargest(350, ["vote_count", "popularity", "vote_average"]).copy()

_CACHE: dict[tuple, dict] = {}
_ACTIVE_PROMPT_STYLE = _load_prompt_style()
_ACTIVE_TUNED_INSTRUCTION = _load_tuned_instruction()


# ---------------------------------------------------------------------------
# Candidate retrieval + ranking (RAG)
# ---------------------------------------------------------------------------


def _history_profile(history_ids: list[int]) -> tuple[set[str], set[str]]:
    id_set = {_safe_int(i) for i in history_ids if _safe_int(i) > 0}
    rows = _MOVIES[_MOVIES["tmdb_id"].isin(id_set)]
    tokens: set[str] = set()
    genres: set[str] = set()
    for row in rows.itertuples():
        tokens |= row.token_set
        genres |= row.genres_set
    return tokens, genres


def _rag_retrieve(
    preferences: str,
    history: list[str],
    history_ids: list[int],
    top_k: int = 100,
) -> pd.DataFrame:
    pref_text = _normalize_text(preferences)
    pref_tokens = _tokenize(pref_text)
    pref_genre_weights = _infer_genre_weights(preferences)
    tone_tokens = _infer_tone_tokens(preferences)
    blocked_genres = _infer_blocked_genres(preferences)

    history_id_set = {_safe_int(i) for i in history_ids if _safe_int(i) > 0}
    history_titles_norm = {_normalize_text(t) for t in history if _normalize_text(t)}

    candidates = TOP_MOVIES[
        (~TOP_MOVIES["tmdb_id"].isin(history_id_set))
        & (~TOP_MOVIES["title_norm"].isin(history_titles_norm))
    ].copy()
    if candidates.empty:
        candidates = TOP_MOVIES.copy()

    history_tokens, history_genres = _history_profile(list(history_id_set))

    # Genres that appear in history but NOT in the preference intent. These are
    # the "conflict" genres the user is pivoting away from.
    pref_genre_set = set(pref_genre_weights)
    conflict_genres = history_genres - pref_genre_set
    preference_overrides_history = bool(pref_genre_set and conflict_genres)

    expanded_tokens = set(pref_tokens) | tone_tokens
    for g in pref_genre_set:
        expanded_tokens |= _tokenize(g)

    scores = []
    norm = max(6, len(expanded_tokens))
    for row in candidates.itertuples():
        lexical = len(expanded_tokens & row.token_set) / norm
        keyword = len(expanded_tokens & row.keywords_set) / norm
        overview_hit = len(pref_tokens & row.overview_set) / max(6, len(pref_tokens) or 6)
        tagline_hit = len(pref_tokens & row.tagline_set) / max(4, len(pref_tokens) or 4)

        genre_bonus = 0.0
        for genre, weight in pref_genre_weights.items():
            if genre in row.genres_set:
                genre_bonus += weight

        blocked_penalty = 1.0 if any(g in row.genres_set for g in blocked_genres) else 0.0

        # Penalize candidates that look like the user's history, more sharply
        # when preference diverges from history genres.
        history_penalty = 0.0
        if history_tokens:
            history_penalty = len(history_tokens & row.token_set) / max(12, len(row.token_set))
        conflict_penalty = 0.0
        if preference_overrides_history and conflict_genres:
            overlap = conflict_genres & row.genres_set
            # Penalize only when the candidate leans more on conflict genres
            # than on preference genres (to allow e.g. a Thriller that happens
            # to also be Romance if preference is strongly Thriller-leaning).
            pref_hit = len(pref_genre_set & row.genres_set)
            if overlap and pref_hit <= len(overlap):
                conflict_penalty = 0.20 * len(overlap)

        score = (
            0.70 * lexical
            + 0.45 * keyword
            + 0.22 * overview_hit
            + 0.16 * tagline_hit
            + 0.36 * genre_bonus
            + 0.38 * float(row.quality_score)
            - 0.16 * history_penalty
            - 0.35 * blocked_penalty
            - conflict_penalty
        )
        scores.append(score)

    candidates["rag_score"] = scores
    return candidates.sort_values("rag_score", ascending=False).head(top_k).copy()


def _rank_candidates(preferences: str, history: list[str], history_ids: list[int]) -> pd.DataFrame:
    pref_tokens = _tokenize(preferences)
    pref_genre_weights = _infer_genre_weights(preferences)
    pref_genre_set = set(pref_genre_weights)
    blocked_genres = _infer_blocked_genres(preferences)
    tone_tokens = _infer_tone_tokens(preferences)

    candidates = _rag_retrieve(preferences, history, history_ids, top_k=100)
    if candidates.empty:
        candidates = TOP_MOVIES.copy()

    _, history_genres = _history_profile([_safe_int(x) for x in history_ids if _safe_int(x) > 0])
    conflict_genres = history_genres - pref_genre_set
    preference_overrides_history = bool(pref_genre_set and conflict_genres)

    scored = []
    pref_token_norm = max(6, len(pref_tokens))
    for row in candidates.itertuples():
        token_overlap = len(pref_tokens & row.token_set) / pref_token_norm
        keyword_bonus = len(pref_tokens & row.keywords_set) / pref_token_norm
        tone_bonus = len(tone_tokens & (row.token_set | row.keywords_set)) / max(4, len(tone_tokens) or 4)

        genre_match = 0.0
        for g, w in pref_genre_weights.items():
            if g in row.genres_set:
                genre_match += w

        blocked_penalty = 1.0 if any(g in row.genres_set for g in blocked_genres) else 0.0
        title_bonus = 0.12 if any(tok in row.title_norm for tok in pref_tokens) else 0.0

        history_overlap = 0.0
        if history_genres:
            history_overlap = len(row.genres_set & history_genres) / max(1, len(row.genres_set | history_genres))

        conflict_penalty = 0.0
        if preference_overrides_history and conflict_genres:
            overlap = conflict_genres & row.genres_set
            pref_hit = len(pref_genre_set & row.genres_set)
            if overlap and pref_hit <= len(overlap):
                conflict_penalty = 0.25 * len(overlap)

        score = (
            0.55 * float(row.quality_score)
            + 0.34 * float(getattr(row, "rag_score", 0.0))
            + 1.05 * token_overlap
            + 0.28 * genre_match
            + 0.30 * keyword_bonus
            + 0.18 * tone_bonus
            + title_bonus
            - 0.30 * blocked_penalty
            - 0.08 * history_overlap
            - conflict_penalty
        )
        scored.append(score)

    candidates["hybrid_score"] = scored
    candidates = candidates.sort_values("hybrid_score", ascending=False)
    return candidates


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _history_pairs(history: list[str], history_ids: list[int]) -> str:
    if not history:
        return "none"
    bits = []
    for idx, title in enumerate(history):
        hid = history_ids[idx] if idx < len(history_ids) else None
        if hid:
            bits.append(f'"{title}" (tmdb_id={hid})')
        else:
            bits.append(f'"{title}"')
    return ", ".join(bits)


def _candidate_block(ranked: pd.DataFrame) -> str:
    rows = []
    for row in ranked.itertuples():
        overview = str(row.overview)[:200].replace("\n", " ")
        rows.append(
            "- "
            f"tmdb_id={int(row.tmdb_id)} | title=\"{row.title}\" | year={_safe_int(row.year)} | "
            f"genres={row.genres} | rating={float(row.vote_average):.2f} | "
            f"director={row.director} | cast={str(row.top_cast)[:90]} | "
            f"rag_score={float(getattr(row, 'rag_score', 0.0)):.3f} | overview={overview}"
        )
    return "\n".join(rows)


def _build_prompt(
    preferences: str,
    history: list[str],
    history_ids: list[int],
    ranked: pd.DataFrame,
    style_name: str,
    tuned_instruction: str,
    banned_ids: list[int] | None = None,
) -> str:
    short_list = ranked.head(CANDIDATES_IN_PROMPT)
    style = PROMPT_STYLES.get(style_name, PROMPT_STYLES["balanced"])

    history_text = _history_pairs(history, history_ids)
    candidate_text = _candidate_block(short_list)

    pref_genre_set = set(_infer_genre_weights(preferences))
    _, history_genres = _history_profile([_safe_int(x) for x in history_ids if _safe_int(x) > 0])
    conflict_genres = sorted(history_genres - pref_genre_set) if pref_genre_set else []
    conflict_note = ""
    if conflict_genres:
        conflict_note = (
            f"\nPreference-vs-history pivot: the user wants {sorted(pref_genre_set)} but "
            f"their history leans on {conflict_genres}. Prioritize preference genres in the "
            "selection, and in the description briefly acknowledge the shift in tone (one short clause).\n"
        )

    tuning_block = ""
    if tuned_instruction:
        tuning_block = f"\nOptimization instruction (from DSPy GEPA):\n{tuned_instruction}\n"

    banned_block = ""
    if banned_ids:
        banned_block = (
            f"\nDo NOT pick these tmdb_id values (already watched or rejected): "
            f"{sorted(set(int(i) for i in banned_ids))}.\n"
        )

    banned_phrase_list = ", ".join(f'"{p}"' for p in BANNED_PHRASES)

    return f"""You are an expert movie recommender.

Task:
Choose exactly ONE movie from the candidate list and return strict JSON only.
You are using RAG context: higher rag_score means stronger retrieval relevance.

User preferences (PRIMARY signal — outranks watch history):
{preferences}

Watch history (never recommend these; use them only for context/contrast):
{history_text}
{conflict_note}
Candidate movies:
{candidate_text}
{tuning_block}{banned_block}
Hard rules:
1) Return ONLY valid JSON with keys: tmdb_id (int), description (string).
2) tmdb_id MUST be one from the candidate list and MUST NOT be in watch history.
3) Selection priority: (a) match user preferences first; (b) use rag_score + rating as a tie-breaker; (c) avoid any genre the user asked to exclude.
4) description MUST be <= {DESCRIPTION_MAX_CHARS} characters (aim for 120-380). Plain text, no markdown, no line breaks, no labels like "Description:".
5) Description must reuse specific preference keywords, name the movie, and cite at least one concrete detail (genre, director, cast, tone, or plot hook).
6) If the user's history genres differ from the preference, acknowledge that contrast in ONE short clause (e.g., "shifting from X to Y").
7) Do NOT use any of these generic phrases: {banned_phrase_list}.
8) Output MUST contain EXACTLY two keys: tmdb_id and description. No reasoning, no extra keys.
9) Style guide: {style}.

Output format:
{{
  "tmdb_id": 123,
  "description": "..."
}}"""


# ---------------------------------------------------------------------------
# Description sanitization
# ---------------------------------------------------------------------------


_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:description|recommendation|pitch|answer|reasoning|why)\s*[:\-–]\s*",
    re.IGNORECASE,
)
_MARKDOWN_RE = re.compile(r"[`*_~]+")
_CODE_FENCE_RE = re.compile(r"```[^`]*```", re.DOTALL)


_ORPHAN_ARTICLE_RE = re.compile(
    r"\b(?:a|an|the|with|of|and|or|is|was|are|were|that|this)\s+(?=[,.;:!?])",
    re.IGNORECASE,
)
_DOUBLE_SPACE_RE = re.compile(r"\s{2,}")
_PUNCT_RUN_RE = re.compile(r"[\s]+([,.;:!?])")


def _strip_banned_phrases(text: str) -> str:
    for phrase in BANNED_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    text = _DOUBLE_SPACE_RE.sub(" ", text)
    text = _PUNCT_RUN_RE.sub(r"\1", text)
    # Drop dangling connectors left behind (e.g., "a with visuals" -> "with visuals").
    text = re.sub(r"\b(?:a|an|the)\s+(?:with|of|and|or|is|was)\b", lambda m: m.group(0).split()[-1], text, flags=re.IGNORECASE)
    text = _ORPHAN_ARTICLE_RE.sub("", text)
    text = _DOUBLE_SPACE_RE.sub(" ", text).strip()
    return text


def _smart_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for punct in (". ", "! ", "? "):
        idx = window.rfind(punct)
        if idx >= int(max_chars * 0.55):
            return window[: idx + 1].rstrip()
    idx = window.rfind(" ")
    if idx >= int(max_chars * 0.55):
        return window[:idx].rstrip(" ,;:") + "."
    return window.rstrip() + "."


_LEADING_PUNCT_RE = re.compile(r"^[\s\-:;,.–—]+")
_TRAILING_ARTICLE_RE = re.compile(
    r"\b(?:a|an|the|is|was|with|of)\s*(?=[.,;:!?])",
    re.IGNORECASE,
)


def _sanitize_description(text: str) -> str:
    raw = str(text or "")
    raw = _CODE_FENCE_RE.sub(" ", raw)
    raw = _MARKDOWN_RE.sub("", raw)
    raw = _LABEL_PREFIX_RE.sub("", raw)
    raw = raw.replace("\r", " ").replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = raw.strip(" \"'“”‘’")
    raw = _strip_banned_phrases(raw)
    raw = _TRAILING_ARTICLE_RE.sub("", raw)
    raw = _LEADING_PUNCT_RE.sub("", raw)
    raw = re.sub(r"\s+([,.;:!?])", r"\1", raw)
    raw = re.sub(r"([,.;:!?])\1+", r"\1", raw)
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    if not raw:
        return ""
    # Capitalize first character for a clean opening.
    raw = raw[0].upper() + raw[1:] if raw else raw
    raw = _smart_truncate(raw, DESCRIPTION_MAX_CHARS)
    if raw and raw[-1] not in ".!?":
        raw = raw.rstrip(" ,;:") + "."
    return raw[:DESCRIPTION_MAX_CHARS]


def _enforce_output_spec(tmdb_id: int, description: str) -> dict:
    # Return exactly two keys as required by llm.py output specification.
    return {
        "tmdb_id": int(tmdb_id),
        "description": _sanitize_description(description),
    }


# ---------------------------------------------------------------------------
# Fallback (no-LLM) composition with variety
# ---------------------------------------------------------------------------


_FALLBACK_OPENERS = (
    "Try {title} ({year}) — it leans into {genre_lower} with {mood_hook}",
    "{title} ({year}) is a strong match for your {pref_tag}: {mood_hook}",
    "For your taste in {pref_tag}, {title} ({year}) delivers {mood_hook}",
    "{title} ({year}) pairs {genre_lower} with {mood_hook}, aligned with what you asked for",
    "Go with {title} ({year}): {mood_hook}, and it stays anchored in {genre_lower}",
)

_FALLBACK_HOOKS = (
    "a focused take on the attributes you highlighted",
    "a grounded execution of the tone you described",
    "a sharp, specific angle on what you asked for",
    "a confident rhythm that matches your brief",
    "a clean fit for the mood you outlined",
)


_PREF_LEAD_RE = re.compile(
    r"^\s*(?:i\s+(?:want|like|love|prefer|need|am looking for|'?m looking for)|give me|looking for|show me|recommend)\s+",
    re.IGNORECASE,
)


def _summarize_preference(preferences: str) -> str:
    text = str(preferences or "").strip()
    if not text:
        return "your taste"
    text = _PREF_LEAD_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip(".!? ,;:")
    if not text:
        return "your taste"
    return text[:90]


def _fallback_result(
    preferences: str,
    history: list[str],
    history_ids: list[int],
    ranked: pd.DataFrame,
    banned_ids: set[int],
) -> dict:
    candidates = ranked[~ranked["tmdb_id"].astype(int).isin(banned_ids)]
    if candidates.empty:
        candidates = ranked
    best = candidates.iloc[0]

    genre_lower = str(best.genres or "").lower().strip() or "its genre"
    pref_tag = _summarize_preference(preferences)
    year = _safe_int(best.year)

    seed = _stable_hash((pref_tag, tuple(sorted(history_ids or [])), int(best.tmdb_id)))
    opener = _FALLBACK_OPENERS[seed % len(_FALLBACK_OPENERS)]
    hook = _FALLBACK_HOOKS[(seed // 7) % len(_FALLBACK_HOOKS)]

    sentence1 = opener.format(
        title=best.title,
        year=year,
        genre_lower=genre_lower,
        pref_tag=pref_tag,
        mood_hook=hook,
    )

    bits = [sentence1.rstrip(".") + "."]

    director = str(best.director or "").strip()
    cast = str(best.top_cast or "").strip()
    detail_parts = []
    if director:
        detail_parts.append(f"Directed by {director}")
    if cast:
        first_cast = cast.split(",")[0].strip()
        if first_cast:
            connector = ", with" if detail_parts else "With"
            detail_parts.append(f"{connector} {first_cast}")
    if detail_parts:
        detail = " ".join(detail_parts).rstrip(".") + "."
        bits.append(detail)

    pref_genre_set = set(_infer_genre_weights(preferences))
    _, history_genres = _history_profile([_safe_int(x) for x in history_ids if _safe_int(x) > 0])
    conflict_genres = sorted(history_genres - pref_genre_set) if pref_genre_set else []
    if history and conflict_genres:
        bits.append(
            f"A clear pivot from the {', '.join(conflict_genres).lower()} tones in your history."
        )

    description = " ".join(bits)
    return _enforce_output_spec(int(best.tmdb_id), description)


# ---------------------------------------------------------------------------
# LLM call + retry
# ---------------------------------------------------------------------------


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


def _call_llm(prompt: str, timeout: int = LLM_PRIMARY_TIMEOUT, temperature: float = 0.3) -> dict:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set")

    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You output strict JSON only. No markdown, no code fences, no labels. "
                    "Exactly two keys: tmdb_id (int), description (string <= "
                    f"{DESCRIPTION_MAX_CHARS} characters)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature, "top_p": 0.9, "num_predict": 220},
        format="json",
    )

    content = ""
    if isinstance(response, dict):
        content = str(response.get("message", {}).get("content", ""))
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
    return _extract_json_payload(content)


def _parse_llm_output(
    raw: dict,
    valid_ids: set[int],
    watched_ids: set[int],
) -> tuple[int, str, str | None]:
    """Return (tmdb_id, description, error). error is None on success."""
    try:
        tmdb_id = _safe_int(raw.get("tmdb_id"), default=-1)
        description = str(raw.get("description", "")).strip()
    except AttributeError:
        return -1, "", "LLM response was not a JSON object"

    if tmdb_id <= 0:
        return tmdb_id, description, "tmdb_id missing or invalid"
    if tmdb_id in watched_ids:
        return tmdb_id, description, "tmdb_id is in watch history"
    if tmdb_id not in valid_ids:
        return tmdb_id, description, "tmdb_id not in candidate list"
    cleaned = _sanitize_description(description)
    if len(cleaned) < 40:
        return tmdb_id, cleaned, "description too short after cleaning"
    return tmdb_id, cleaned, None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    preferences: str,
    history: list[str],
    history_ids: list[int],
) -> tuple[str, list[str], list[int]]:
    clean_pref = re.sub(r"\s+", " ", str(preferences or "").strip())[:1000]

    raw_history = list(history or [])
    raw_ids = list(history_ids or [])

    titles: list[str] = []
    ids: list[int] = []
    seen_ids: set[int] = set()
    seen_norm_titles: set[str] = set()

    for idx, title in enumerate(raw_history):
        title_str = str(title or "").strip()
        if not title_str:
            continue
        norm = _normalize_text(title_str)
        if norm in seen_norm_titles:
            continue
        seen_norm_titles.add(norm)

        hid = _safe_int(raw_ids[idx], default=0) if idx < len(raw_ids) else 0
        if hid > 0 and hid in seen_ids:
            continue
        if hid > 0:
            seen_ids.add(hid)

        titles.append(title_str[:200])
        ids.append(hid)

    # Also fold in any ids with no matching title (align by appending).
    for extra_id in raw_ids[len(raw_history):]:
        hid = _safe_int(extra_id, default=0)
        if hid > 0 and hid not in seen_ids:
            row = _MOVIES[_MOVIES["tmdb_id"] == hid]
            title_str = str(row.iloc[0]["title"]).strip() if not row.empty else f"tmdb:{hid}"
            titles.append(title_str)
            ids.append(hid)
            seen_ids.add(hid)

    return clean_pref, titles, ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    clean_pref, clean_history, clean_ids = _validate_inputs(preferences, history, history_ids)

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
    tried_ids: set[int] = set()

    prompt = _build_prompt(
        clean_pref,
        clean_history,
        clean_ids,
        ranked,
        style_name,
        tuned_instruction,
    )

    tmdb_id = -1
    description = ""
    error: str | None = "llm not attempted"

    try:
        raw = _call_llm(prompt, timeout=LLM_PRIMARY_TIMEOUT, temperature=0.3)
        tmdb_id, description, error = _parse_llm_output(raw, valid_ids, watched_ids)
        if tmdb_id > 0:
            tried_ids.add(tmdb_id)
    except Exception as exc:
        error = f"primary call failed: {exc}"

    # One corrective retry: if the model picked a watched/invalid id, re-prompt
    # with that id explicitly banned and a tightened candidate list.
    if error is not None:
        try:
            banned = sorted(watched_ids | tried_ids)
            filtered = ranked[~ranked["tmdb_id"].astype(int).isin(banned)]
            if filtered.empty:
                filtered = ranked
            retry_prompt = _build_prompt(
                clean_pref,
                clean_history,
                clean_ids,
                filtered,
                style_name,
                tuned_instruction,
                banned_ids=banned,
            )
            raw = _call_llm(retry_prompt, timeout=LLM_RETRY_TIMEOUT, temperature=0.2)
            tmdb_id2, description2, error2 = _parse_llm_output(raw, valid_ids, watched_ids)
            if error2 is None:
                tmdb_id, description, error = tmdb_id2, description2, None
                tried_ids.add(tmdb_id2)
        except Exception as exc:
            error = f"{error}; retry failed: {exc}"

    if error is None and tmdb_id in valid_ids and tmdb_id not in watched_ids:
        result = _enforce_output_spec(tmdb_id, description)
    else:
        result = _fallback_result(
            clean_pref,
            clean_history,
            clean_ids,
            ranked,
            banned_ids=watched_ids | tried_ids,
        )

    # Final safety: ensure result is valid and description non-empty.
    if (
        not isinstance(result, dict)
        or int(result.get("tmdb_id", 0)) not in valid_ids
        or int(result.get("tmdb_id", 0)) in watched_ids
        or not str(result.get("description", "")).strip()
    ):
        result = _fallback_result(
            clean_pref,
            clean_history,
            clean_ids,
            ranked,
            banned_ids=watched_ids | tried_ids,
        )

    _CACHE[cache_key] = dict(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
