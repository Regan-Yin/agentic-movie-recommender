import argparse
import json
import os
import random
import re
import ssl
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import importlib

from llm import (
    BANNED_PHRASES,
    DESCRIPTION_MAX_CHARS,
    MODEL,
    PROMPT_STYLES,
    TUNED_CONFIG_PATH,
    _MOVIES,
    _history_profile,
    _infer_genre_weights,
    _normalize_text,
    _rank_candidates,
    _safe_int,
    get_recommendation,
)


dspy = importlib.import_module("dspy")

TMDB_BASE = "https://api.themoviedb.org/3"
CASE_DUMP_PATH = "dspy_gepa_eval_cases.json"


@dataclass
class EvalCase:
    preferences: str
    history: list[str]
    history_ids: list[int]


BASE_DEV_CASES = [
    EvalCase("I want action movies with superheroes and strong visuals.", [], []),
    EvalCase("Give me something funny and feel-good.", ["The Dark Knight Rises"], [49026]),
    EvalCase("I like emotional sci-fi adventure with hope.", ["Avatar: Fire and Ash"], [83533]),
    EvalCase("Need a smart mystery thriller with twists.", [], []),
    EvalCase("I want a warm family animation for weekend.", [], []),
    EvalCase("I want tense crime action but not horror.", [], []),
    EvalCase("Looking for romantic drama with strong character arc.", [], []),
    EvalCase("Give me a dark but stylish action thriller.", ["Shelter"], [1290821]),
    # Conflict cases: preference intentionally diverges from history genre mix,
    # so the model must prioritize preference and acknowledge the pivot.
    EvalCase(
        "I want a pure thriller with real tension and no romance subplot.",
        ["Crime 101"],
        [1171145],
    ),
    EvalCase(
        "Give me an atmospheric horror with slow-burn dread, not gore.",
        ["Pretty Lethal"],
        [1084187],
    ),
    EvalCase(
        "I want a cerebral sci-fi, not an action-heavy one.",
        ["War Machine"],
        [1265609],
    ),
    EvalCase(
        "Looking for a light feel-good comedy, not a dark crime piece.",
        ["The Shadow's Edge"],
        [1419406],
    ),
]

PREFERENCE_TEMPLATES = {
    "Action": [
        "I want high-energy action with heroic stakes.",
        "Give me an action movie with intense set pieces.",
    ],
    "Adventure": [
        "I want an adventurous movie with epic world-building.",
        "Give me a fun adventure with big momentum.",
    ],
    "Animation": [
        "I want a heartfelt animated movie with charm.",
        "Give me creative animation and an uplifting tone.",
    ],
    "Comedy": [
        "I want something funny and feel-good tonight.",
        "Give me a comedy that is light and rewatchable.",
    ],
    "Crime": [
        "I want a sharp crime story with smart turns.",
        "Give me crime drama with strong tension.",
    ],
    "Drama": [
        "I want emotional character-driven storytelling.",
        "Give me a meaningful drama with strong performances.",
    ],
    "Fantasy": [
        "I want fantasy with imaginative lore and adventure.",
        "Give me magical world-building and emotional stakes.",
    ],
    "Horror": [
        "I want unsettling horror with suspense.",
        "Give me tense horror that stays with me.",
    ],
    "Mystery": [
        "I want a mystery with twists and clues.",
        "Give me an intelligent mystery thriller.",
    ],
    "Romance": [
        "I want a romantic movie with chemistry and heart.",
        "Give me a romance that is emotional but not cheesy.",
    ],
    "Science Fiction": [
        "I want science fiction with ideas and emotion.",
        "Give me smart sci-fi with immersive world design.",
    ],
    "Thriller": [
        "I want a tense thriller with strong pacing.",
        "Give me a suspenseful thriller with high stakes.",
    ],
    "War": [
        "I want a war movie with intensity and moral tension.",
        "Give me historical war drama with high stakes.",
    ],
}


class RecommendFromRAG(dspy.Signature):
    """Select one tmdb_id from candidates and write a short persuasive description."""

    preferences: str = dspy.InputField()
    history_text: str = dspy.InputField()
    candidate_block: str = dspy.InputField()
    style_guide: str = dspy.InputField()

    tmdb_id: int = dspy.OutputField(desc="One tmdb_id from candidates")
    description: str = dspy.OutputField(desc="Why this movie matches the user")


class MovieProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.recommend = dspy.ChainOfThought(RecommendFromRAG)

    def forward(self, preferences: str, history_text: str, candidate_block: str, style_guide: str):
        return self.recommend(
            preferences=preferences,
            history_text=history_text,
            candidate_block=candidate_block,
            style_guide=style_guide,
        )


def _http_get_json(url: str, headers: dict | None = None, timeout: int = 15) -> dict:
    req = Request(url, headers=headers or {})
    no_verify = str(os.getenv("TMDB_SSL_NO_VERIFY", "")).strip().lower() in {"1", "true", "yes"}
    if no_verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        resp = urlopen(req, timeout=timeout, context=ctx)
    else:
        resp = urlopen(req, timeout=timeout)
    with resp:
        return json.loads(resp.read().decode("utf-8"))


def _tmdb_get(path: str, params: dict, api_key: str = "", bearer_token: str = "") -> dict:
    merged = dict(params)
    headers = {"accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif api_key:
        merged["api_key"] = api_key
    else:
        raise RuntimeError("TMDB_API_KEY or TMDB_READ_ACCESS_TOKEN is required for TMDB fetch")

    url = f"{TMDB_BASE}{path}?{urlencode(merged)}"
    return _http_get_json(url, headers=headers)


def _fetch_tmdb_genres(api_key: str, bearer_token: str) -> dict[int, str]:
    try:
        payload = _tmdb_get("/genre/movie/list", {"language": "en-US"}, api_key, bearer_token)
        return {int(item["id"]): str(item["name"]) for item in payload.get("genres", [])}
    except (HTTPError, URLError, TimeoutError):
        return {}


def _discover_tmdb_movies(api_key: str, bearer_token: str, pages: int = 4) -> list[dict]:
    movies = []
    for page in range(1, pages + 1):
        try:
            payload = _tmdb_get(
                "/discover/movie",
                {
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "include_video": "false",
                    "vote_count.gte": 300,
                    "page": page,
                },
                api_key,
                bearer_token,
            )
            movies.extend(payload.get("results", []))
        except (HTTPError, URLError, TimeoutError):
            continue
    return movies


def _history_text(history: list[str], history_ids: list[int]) -> str:
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


def _pick_history_movie(rng: random.Random, exclude_id: int) -> tuple[list[str], list[int]]:
    pool = _MOVIES[_MOVIES["tmdb_id"] != exclude_id].nlargest(300, "vote_count")
    if pool.empty or rng.random() < 0.45:
        return [], []
    idx = rng.randrange(len(pool))
    row = pool.iloc[idx]
    return [str(row.title)], [int(row.tmdb_id)]


def _genre_preference(genre: str, title: str, rng: random.Random) -> str:
    templates = PREFERENCE_TEMPLATES.get(genre, PREFERENCE_TEMPLATES["Drama"])
    prompt = templates[rng.randrange(len(templates))]
    if rng.random() < 0.3:
        prompt += f" I liked {title}, but want something fresh."
    return prompt


def _generate_tmdb_api_cases(target_count: int, api_key: str, bearer_token: str, seed: int) -> list[EvalCase]:
    if target_count <= 0:
        return []

    rng = random.Random(seed)
    genre_map = _fetch_tmdb_genres(api_key, bearer_token)
    rows = _discover_tmdb_movies(api_key, bearer_token, pages=4)
    if not rows:
        return []

    rng.shuffle(rows)
    out = []
    seen_prefs = set()
    for row in rows:
        tmdb_id = _safe_int(row.get("id"), 0)
        if tmdb_id <= 0:
            continue
        title = str(row.get("title", "")).strip() or "this movie"
        genre_ids = row.get("genre_ids", []) or []
        genre_name = genre_map.get(_safe_int(genre_ids[0], -1), "Drama") if genre_ids else "Drama"

        pref = _genre_preference(genre_name, title, rng)
        pref_norm = _normalize_text(pref)
        if pref_norm in seen_prefs:
            continue
        seen_prefs.add(pref_norm)

        history, history_ids = _pick_history_movie(rng, exclude_id=tmdb_id)
        out.append(EvalCase(pref, history, history_ids))
        if len(out) >= target_count:
            break
    return out


def _generate_local_cases(target_count: int, seed: int) -> list[EvalCase]:
    if target_count <= 0:
        return []

    rng = random.Random(seed)
    local_rows = _MOVIES.nlargest(450, "vote_count").sample(frac=1.0, random_state=seed)
    out = []
    seen = set()
    for row in local_rows.itertuples():
        genres = [g.strip() for g in str(row.genres).split(",") if g.strip()]
        genre_name = genres[0] if genres else "Drama"
        pref = _genre_preference(genre_name, str(row.title), rng)
        pref_norm = _normalize_text(pref)
        if pref_norm in seen:
            continue
        seen.add(pref_norm)

        history, history_ids = _pick_history_movie(rng, exclude_id=int(row.tmdb_id))
        out.append(EvalCase(pref, history, history_ids))
        if len(out) >= target_count:
            break
    return out


def _build_eval_cases(num_cases: int, seed: int) -> tuple[list[EvalCase], bool]:
    cases = list(BASE_DEV_CASES)
    tmdb_used = False

    api_key = str(os.getenv("TMDB_API_KEY", "")).strip()
    bearer_token = str(os.getenv("TMDB_READ_ACCESS_TOKEN", "")).strip()
    if len(cases) < num_cases and (api_key or bearer_token):
        extra = _generate_tmdb_api_cases(num_cases - len(cases), api_key, bearer_token, seed)
        if extra:
            tmdb_used = True
            cases.extend(extra)

    if len(cases) < num_cases:
        cases.extend(_generate_local_cases(num_cases - len(cases), seed))

    return cases[:num_cases], tmdb_used


def _as_candidate_block(preferences: str, history: list[str], history_ids: list[int], top_k: int = 12):
    ranked = _rank_candidates(preferences, history, history_ids)
    short_list = ranked.head(top_k)
    lines = []
    valid_ids = []
    for row in short_list.itertuples():
        valid_ids.append(int(row.tmdb_id))
        lines.append(
            f"tmdb_id={int(row.tmdb_id)} | title={row.title} | genres={row.genres} | "
            f"rating={float(row.vote_average):.2f} | rag_score={float(getattr(row, 'rag_score', 0.0)):.3f}"
        )
    history_id_set = {_safe_int(i) for i in history_ids if _safe_int(i) > 0}
    return "\n".join(lines), valid_ids, history_id_set


def _genre_alignment_score(preferences: str, movie_genres: str) -> float:
    target = _infer_genre_weights(preferences)
    if not target:
        return 0.55
    genres = {g.strip() for g in str(movie_genres).split(",") if g.strip()}
    hit = 0.0
    total = 0.0
    for g, w in target.items():
        total += w
        if g in genres:
            hit += w
    return hit / (total or 1.0)


def _specificity_score(preferences: str, description: str) -> float:
    stop_words = {
        "the", "and", "with", "that", "this", "movie", "film", "your", "you", "for", "from", "into", "about",
    }
    generic = {"masterpiece", "stunning", "amazing", "incredible", "epic", "journey", "breathtaking", "must-watch"}

    pref_tokens = {t for t in re.findall(r"[a-z0-9']+", _normalize_text(preferences)) if len(t) >= 4 and t not in stop_words}
    desc_tokens = [t for t in re.findall(r"[a-z0-9']+", _normalize_text(description)) if len(t) >= 4]
    if not desc_tokens:
        return 0.0

    overlap = len(set(desc_tokens) & pref_tokens) / max(1, len(pref_tokens)) if pref_tokens else 0.45
    generic_density = sum(1 for t in desc_tokens if t in generic) / max(1, len(desc_tokens))
    return max(0.0, min(1.0, 0.72 * overlap + 0.28 * (1.0 - generic_density)))


def _banned_phrase_penalty(description: str) -> float:
    text = _normalize_text(description)
    if not text:
        return 0.0
    hits = sum(1 for phrase in BANNED_PHRASES if phrase.lower() in text)
    return min(1.0, 0.25 * hits)


def _history_acknowledgement_bonus(preferences: str, history_ids: set[int], description: str) -> float:
    if not history_ids:
        return 0.0
    pref_genre_set = set(_infer_genre_weights(preferences))
    _, history_genres = _history_profile(list(history_ids))
    conflict = history_genres - pref_genre_set
    if not conflict:
        return 0.0
    desc_norm = _normalize_text(description)
    mention = any(g.lower() in desc_norm for g in conflict)
    pivot_words = ("shift", "pivot", "moving", "unlike", "contrast", "instead of", "away from")
    pivot_mentioned = any(w in desc_norm for w in pivot_words)
    return 1.0 if (mention and pivot_mentioned) else (0.55 if pivot_mentioned else 0.0)


def _metric_from_output(preferences: str, history_ids: set[int], valid_ids: list[int], tmdb_id: int, description: str):
    if tmdb_id not in set(valid_ids):
        return 0.0, "Chosen tmdb_id is outside the candidate list."
    if tmdb_id in history_ids:
        return 0.0, "Chosen tmdb_id is already in watch history."

    row = _MOVIES[_MOVIES["tmdb_id"] == tmdb_id]
    if row.empty:
        return 0.0, "Movie id not found in metadata table."

    movie = row.iloc[0]
    genre_score = _genre_alignment_score(preferences, str(movie.genres))
    quality_score = float(movie.quality_score)
    desc_len = len(str(description or ""))
    if desc_len > DESCRIPTION_MAX_CHARS:
        desc_len_score = 0.0
    elif 90 <= desc_len <= 380:
        desc_len_score = 1.0
    elif 60 <= desc_len < 90 or 380 < desc_len <= 480:
        desc_len_score = 0.8
    elif 35 <= desc_len < 60:
        desc_len_score = 0.5
    elif 1 <= desc_len < 35:
        desc_len_score = 0.25
    else:
        desc_len_score = 0.0

    spec_score = _specificity_score(preferences, description)
    banned_penalty = _banned_phrase_penalty(description)
    history_bonus = _history_acknowledgement_bonus(preferences, history_ids, description)

    # Preference-first weighting: genre alignment with preference dominates.
    final = (
        0.36 * genre_score
        + 0.22 * quality_score
        + 0.16 * desc_len_score
        + 0.18 * spec_score
        + 0.08 * history_bonus
        - 0.20 * banned_penalty
    )
    final = max(0.0, min(1.0, final))

    feedback = (
        f"genre={genre_score:.3f}, quality={quality_score:.3f}, length={desc_len_score:.3f}, "
        f"specificity={spec_score:.3f}, history_ack={history_bonus:.2f}, banned={banned_penalty:.2f}. "
        "Prioritize preference keywords over history, keep description <=500 chars, name a concrete "
        "detail (director/cast/plot hook), and avoid banned marketing phrases. If history genres "
        "differ from preference, acknowledge the pivot in ONE short clause."
    )
    return final, feedback


def _build_dspy_example(case: EvalCase, style: str):
    block, valid_ids, history_id_set = _as_candidate_block(
        case.preferences,
        case.history,
        case.history_ids,
        top_k=12,
    )
    return dspy.Example(
        preferences=case.preferences,
        history_text=_history_text(case.history, case.history_ids),
        candidate_block=block,
        style_guide=PROMPT_STYLES.get(style, PROMPT_STYLES["balanced"]),
        valid_ids_csv=",".join(str(x) for x in valid_ids),
        history_ids_csv=",".join(str(x) for x in sorted(history_id_set)),
    ).with_inputs("preferences", "history_text", "candidate_block", "style_guide")


def _metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    valid_ids = [int(x) for x in str(getattr(gold, "valid_ids_csv", "")).split(",") if x.strip()]
    history_id_set = {int(x) for x in str(getattr(gold, "history_ids_csv", "")).split(",") if x.strip()}

    tmdb_id = _safe_int(getattr(pred, "tmdb_id", -1), default=-1)
    description = str(getattr(pred, "description", "")).strip()

    score, feedback = _metric_from_output(
        preferences=str(getattr(gold, "preferences", "")),
        history_ids=history_id_set,
        valid_ids=valid_ids,
        tmdb_id=tmdb_id,
        description=description,
    )
    return dspy.Prediction(score=float(score), feedback=feedback)


def _evaluate_program(program, dataset):
    scores = []
    for ex in dataset:
        pred = program(
            preferences=ex.preferences,
            history_text=ex.history_text,
            candidate_block=ex.candidate_block,
            style_guide=ex.style_guide,
        )
        metric_result = _metric(ex, pred)
        scores.append(float(getattr(metric_result, "score", 0.0)))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _evaluate_native_style(style: str, cases: list[EvalCase]):
    old_style = os.getenv("RECOMMENDER_PROMPT_STYLE")
    os.environ["RECOMMENDER_PROMPT_STYLE"] = style
    scores = []
    try:
        for case in cases:
            ranked = _rank_candidates(case.preferences, case.history, case.history_ids)
            valid_ids = [int(x) for x in ranked.head(12)["tmdb_id"].astype(int).tolist()]
            history_set = {_safe_int(x) for x in case.history_ids if _safe_int(x) > 0}
            out = get_recommendation(case.preferences, case.history, case.history_ids)
            s, _ = _metric_from_output(
                preferences=case.preferences,
                history_ids=history_set,
                valid_ids=valid_ids,
                tmdb_id=_safe_int(out.get("tmdb_id"), -1),
                description=str(out.get("description", "")),
            )
            scores.append(float(s))
    finally:
        if old_style is None:
            os.environ.pop("RECOMMENDER_PROMPT_STYLE", None)
        else:
            os.environ["RECOMMENDER_PROMPT_STYLE"] = old_style
    return sum(scores) / max(1, len(scores))


def _configure_dspy_models():
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is required")

    model = os.getenv("DSPY_MODEL", f"ollama_chat/{MODEL}")
    lm = dspy.LM(model=model, api_base="https://ollama.com", api_key=api_key, temperature=0.3, max_tokens=220)
    reflection_model = os.getenv("DSPY_REFLECTION_MODEL", model)
    reflection_lm = dspy.LM(
        model=reflection_model,
        api_base="https://ollama.com",
        api_key=api_key,
        temperature=0.9,
        max_tokens=600,
    )
    dspy.configure(lm=lm)
    return reflection_lm


def _extract_best_instruction(program) -> str:
    try:
        text = str(program.recommend.predict.signature.instructions).strip()
        if text:
            return text
    except Exception:
        pass

    try:
        details = getattr(program, "detailed_results", None)
        best_candidate = getattr(details, "best_candidate", None) if details is not None else None
        if isinstance(best_candidate, dict):
            for key in ["recommend.predict", "recommend", "predict"]:
                if key in best_candidate and str(best_candidate[key]).strip():
                    return str(best_candidate[key]).strip()
            for value in best_candidate.values():
                if str(value).strip():
                    return str(value).strip()
    except Exception:
        pass

    return ""


def run(args):
    target_cases = max(args.num_cases, args.min_cases)
    cases, tmdb_used = _build_eval_cases(target_cases, args.seed)
    if len(cases) < 6:
        raise RuntimeError("Need at least 6 eval cases for a meaningful GEPA run")

    with open(CASE_DUMP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "preferences": c.preferences,
                    "history": c.history,
                    "history_ids": c.history_ids,
                }
                for c in cases
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.prepare_only:
        summary = {
            "num_cases": len(cases),
            "tmdb_api_cases_used": tmdb_used,
            "eval_case_dump": CASE_DUMP_PATH,
            "seed": args.seed,
            "mode": "prepare-only",
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    reflection_lm = _configure_dspy_models()

    style_scores = {style: _evaluate_native_style(style, cases) for style in PROMPT_STYLES}
    best_style = max(style_scores, key=style_scores.get)

    split = max(4, int(0.6 * len(cases)))
    train_cases = cases[:split]
    val_cases = cases[split:] if len(cases) > split else cases[:max(2, len(cases) // 3)]

    trainset = [_build_dspy_example(c, best_style) for c in train_cases]
    valset = [_build_dspy_example(c, best_style) for c in val_cases]

    student = MovieProgram()
    baseline_score = _evaluate_program(student, valset)

    gepa_kwargs = {
        "metric": _metric,
        "reflection_lm": reflection_lm,
        "track_stats": True,
        "seed": args.seed,
        "num_threads": args.num_threads,
    }
    budget_mode = "auto"
    if args.max_metric_calls and args.max_metric_calls > 0:
        gepa_kwargs["max_metric_calls"] = int(args.max_metric_calls)
        budget_mode = "max_metric_calls"
    elif args.max_full_evals and args.max_full_evals > 0:
        gepa_kwargs["max_full_evals"] = int(args.max_full_evals)
        budget_mode = "max_full_evals"
    else:
        gepa_kwargs["auto"] = args.auto

    gepa = dspy.GEPA(**gepa_kwargs)
    optimized = gepa.compile(student, trainset=trainset, valset=valset)
    optimized_score = _evaluate_program(optimized, valset)
    best_instruction = _extract_best_instruction(optimized)

    summary = {
        "best_style": best_style,
        "best_instruction": best_instruction,
        "native_style_scores": style_scores,
        "dspy_baseline_score": baseline_score,
        "dspy_gepa_score": optimized_score,
        "num_cases": len(cases),
        "auto_budget": args.auto,
        "budget_mode": budget_mode,
        "max_metric_calls": int(args.max_metric_calls or 0),
        "max_full_evals": int(args.max_full_evals or 0),
        "tmdb_api_cases_used": tmdb_used,
        "eval_case_dump": CASE_DUMP_PATH,
        "seed": args.seed,
    }

    with open(TUNED_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG + DSPy/GEPA optimization and performance testing.")
    parser.add_argument("--num-cases", type=int, default=12, help="Requested number of eval cases.")
    parser.add_argument("--min-cases", type=int, default=30, help="Lower bound for eval cases.")
    parser.add_argument(
        "--auto",
        type=str,
        default="light",
        choices=["light", "medium", "heavy"],
        help="GEPA auto budget",
    )
    parser.add_argument("--num-threads", type=int, default=1, help="Threads for GEPA eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prepare-only", action="store_true", help="Only build eval cases; skip GEPA optimization")
    parser.add_argument("--max-metric-calls", type=int, default=0, help="Override GEPA budget with max metric calls")
    parser.add_argument("--max-full-evals", type=int, default=0, help="Override GEPA budget with max full evals")
    run(parser.parse_args())
