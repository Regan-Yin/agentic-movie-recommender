"""
Independent validation framework for the movie recommendation agent.

This is a SECOND evaluation methodology that runs ALONGSIDE the DSPy/GEPA
loop (see dspy_gepa_benchmark.py). Where GEPA scores our agent against a
preference-first metric, this framework compares the agent against a
completely independent, data-driven TF-IDF baseline recommender.

Why have both?
  - GEPA tunes the agent's prompt using programmatic feedback.
  - This validator sanity-checks the tuned agent against a classic
    information-retrieval baseline that does NOT use an LLM. If the two
    agree on lots of cases, the agent is behaving reasonably. Where they
    disagree, we can inspect the case.

Because multiple valid recommendations exist for the same preferences, we
use similarity-based metrics rather than just exact-match accuracy.

EVALUATION METRICS
==================
  1. Exact Match Rate: Agent picked exactly the baseline's top-1. Very strict.
  2. Soft Match Rate:  Agent picked a movie whose hybrid similarity to the
                       baseline's top-1 is >= 0.60.
  3. Top-K Hit Rate:   Agent's pick is in the baseline's top-5 / top-10 /
                       top-100. Good signal of "reasonable candidate".
  4. Rank Score:       101 - rank(agent_pick) in baseline top-100 (capped at
                       0). Rank 1 -> 100, rank 100 -> 1, outside top-100 -> 0.
  5. Similarity:       Genre Jaccard + keyword Jaccard + overview TF-IDF cosine,
                       blended into a hybrid composite in [0, 1].
  6. Preference Alignment: TF-IDF similarity between the user's preference
                       text and each recommendation's overview (sanity check).

PRIMARY ACCURACY DEFINITION (what we quote in the README):
  - PRIMARY:   Rank Score (continuous, accounts for near-misses)
  - SECONDARY: Soft Match Rate (accounts for valid alternatives)
  - TERTIARY:  Top-K Hit Rates (baseline set overlap)

Fair comparison note
--------------------
The baseline is restricted to the SAME candidate pool the agent can pick
from (`llm.TOP_MOVIES` — 350 movies). This avoids penalizing the agent for
being limited to a pool the baseline would otherwise outrun.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Vocabulary (kept self-contained so validation.py has no hard dependency on
# llm.py's internals; these mirror the production vocabulary for consistency)
# ---------------------------------------------------------------------------

GENRE_SYNONYMS: Dict[str, Set[str]] = {
    "action": {"action", "fight", "fighting", "explosive", "high-octane"},
    "adventure": {"adventure", "quest", "journey", "epic"},
    "animation": {"animation", "animated", "cartoon"},
    "comedy": {"comedy", "funny", "humor", "humorous", "hilarious", "lighthearted", "feel-good", "feel good"},
    "crime": {"crime", "criminal", "gangster", "heist", "mob", "mafia"},
    "documentary": {"documentary", "doc", "nonfiction"},
    "drama": {"drama", "dramatic", "emotional", "character-driven", "character study"},
    "family": {"family", "kids", "children", "heartwarming"},
    "fantasy": {"fantasy", "magical", "magic", "mythic"},
    "history": {"history", "historical", "period"},
    "horror": {"horror", "scary", "frightening", "disturbing", "terrifying", "slasher", "dread"},
    "music": {"music", "musical", "band", "singer"},
    "mystery": {"mystery", "mysterious", "whodunit", "detective"},
    "romance": {"romance", "romantic", "love"},
    "science fiction": {"science fiction", "sci-fi", "scifi", "futuristic", "space", "alien", "ai", "cyberpunk"},
    "thriller": {"thriller", "tense", "suspense", "suspenseful", "gripping", "psychological"},
    "war": {"war", "military", "battle", "wartime"},
    "western": {"western", "cowboy", "frontier"},
}

INTENT_KEYWORDS: Set[str] = {
    "superhero", "heroes", "hero", "antihero", "villain", "dark", "gritty", "psychological",
    "true story", "based on a true story", "realism", "realistic", "ambitious", "slow burn",
    "slow-burn", "violent", "disturbing", "satire", "political", "survival", "space", "crime",
    "serial killer", "investigation", "courtroom", "revenge", "heist", "twist", "twists",
    "mind-bending", "suspense", "character study", "corruption", "chaotic", "society",
    "feel-good", "feel good", "uplifting", "atmospheric", "stylish", "cerebral",
}

TONE_KEYWORDS: Dict[str, Set[str]] = {
    "dark": {"dark", "bleak", "grim", "gritty", "brooding"},
    "psychological": {"psychological", "mentally", "obsession", "descent", "unstable"},
    "antihero": {"antihero", "anti-hero", "vigilante", "outsider"},
    "crime": {"crime", "criminal", "corruption", "mob", "violence"},
    "character study": {"character study", "loner", "isolation", "identity", "society"},
    "atmospheric": {"atmospheric", "dread", "eerie", "haunting", "moody", "unsettling"},
    "slow-burn": {"slow burn", "slow-burn", "simmering", "methodical", "deliberate"},
    "feel-good": {"uplifting", "hopeful", "heartwarming", "cheerful", "charming"},
    "stylish": {"stylish", "sleek", "noir", "sophisticated"},
}

# Explicit-exclusion prefixes (e.g. "no horror", "not romance", "avoid gore")
NEGATION_PREFIXES = ("no ", "not ", "without ", "avoid ", "except ", "skip ")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class MovieDataset:
    """Thin wrapper around the TMDB CSV used by the validation framework."""

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        self.df = pd.read_csv(data_path)
        self._preprocess()

    def _preprocess(self) -> None:
        for col in ("year", "vote_average", "popularity", "vote_count"):
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        for col in ("overview", "genres", "keywords", "tagline", "director", "top_cast", "title"):
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")

    def get_movie_by_id(self, tmdb_id: int) -> Optional[Dict]:
        matches = self.df[self.df["tmdb_id"] == tmdb_id]
        if matches.empty:
            return None
        m = matches.iloc[0]
        return {
            "tmdb_id": int(m["tmdb_id"]),
            "title": str(m["title"]),
            "overview": str(m["overview"]),
            "genres": str(m["genres"]),
            "keywords": str(m["keywords"]),
            "year": int(m["year"]) if pd.notna(m["year"]) else None,
            "vote_average": float(m["vote_average"]) if pd.notna(m["vote_average"]) else None,
            "vote_count": int(m["vote_count"]) if pd.notna(m["vote_count"]) else 0,
        }

    def get_movie_by_title(self, title: str) -> Optional[Dict]:
        title_lower = str(title).lower().strip()
        exact = self.df[self.df["title"].str.lower() == title_lower]
        if not exact.empty:
            return self.get_movie_by_id(int(exact.iloc[0]["tmdb_id"]))
        fuzzy = self.df[self.df["title"].str.lower().str.contains(title_lower, na=False, regex=False)]
        if not fuzzy.empty:
            return self.get_movie_by_id(int(fuzzy.iloc[0]["tmdb_id"]))
        return None


# ---------------------------------------------------------------------------
# Baseline recommender (independent of llm.py)
# ---------------------------------------------------------------------------


class BaselineRecommender:
    """
    Intent-aware TF-IDF baseline recommender.

    Scoring (all components normalized to [0, 1]):
      0.30 * overview/keywords/genres TF-IDF similarity to preference text
      0.22 * explicit preference-term keyword matches on movie metadata
      0.18 * explicit genre alignment with preference
      0.12 * tone/mood boost (dark, psychological, atmospheric, ...)
      0.10 * watch-history similarity (light signal only)
      0.08 * quality score (vote_average + log(vote_count))
      -       hard penalty if the movie lands in a preference-blocked genre
      -       small quality penalties on very-low-rated titles

    The pool is restricted to `candidate_ids` when provided, so the baseline
    only competes on the same ground as the agent (default: llm.TOP_MOVIES).
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        candidate_ids: Optional[Set[int]] = None,
        min_year: int = 2010,
    ):
        all_movies = movies_df.copy()

        # Clean types once; cheap enough not to share with MovieDataset here.
        for col in ("year", "vote_average", "popularity", "vote_count"):
            all_movies[col] = pd.to_numeric(all_movies[col], errors="coerce")
        for col in ("overview", "genres", "keywords", "title"):
            all_movies[col] = all_movies[col].fillna("")

        if candidate_ids:
            pool = all_movies[all_movies["tmdb_id"].isin(candidate_ids)].copy()
        else:
            pool = all_movies[
                (all_movies["year"].fillna(0) >= min_year) &
                (all_movies["vote_count"].fillna(0) > 0)
            ].copy()

        self.movies = pool.reset_index(drop=True)
        combined = (
            self.movies["overview"].astype(str) + " " +
            self.movies["keywords"].astype(str) + " " +
            self.movies["genres"].astype(str)
        )
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(combined)

    # ------------------------- intent parsing -------------------------

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        if not text:
            return set()
        cleaned = str(text).lower()
        for ch in [",", ".", ":", ";", "!", "?", "(", ")", "/", "-", "_"]:
            cleaned = cleaned.replace(ch, " ")
        return {tok for tok in cleaned.split() if tok}

    def _infer_requested_genres(self, preferences: str) -> Set[str]:
        prefs_lower = str(preferences).lower()
        tokens = self._tokenize(preferences)
        hits: Set[str] = set()
        for genre, synonyms in GENRE_SYNONYMS.items():
            if genre in prefs_lower or any(term in tokens for term in synonyms):
                hits.add(genre)
        return hits

    def _infer_blocked_genres(self, preferences: str) -> Set[str]:
        prefs_lower = str(preferences).lower()
        blocked: Set[str] = set()
        for genre, synonyms in GENRE_SYNONYMS.items():
            for term in {genre, *synonyms}:
                if any((p + term) in prefs_lower for p in NEGATION_PREFIXES):
                    blocked.add(genre)
                    break
        return blocked

    def _extract_preference_terms(self, preferences: str) -> Set[str]:
        prefs_lower = str(preferences).lower()
        tokens = self._tokenize(preferences)
        terms: Set[str] = set()
        for genre, synonyms in GENRE_SYNONYMS.items():
            if genre in prefs_lower or any(term in tokens for term in synonyms):
                terms.add(genre)
                terms.update({t for t in synonyms if t in prefs_lower or t in tokens})
        for phrase in INTENT_KEYWORDS:
            if " " in phrase:
                if phrase in prefs_lower:
                    terms.add(phrase)
            elif phrase in tokens:
                terms.add(phrase)
        terms.update({t for t in tokens if len(t) >= 5})
        return terms

    # ------------------------- component scores -------------------------

    def _keyword_match_scores(self, preferences: str, candidates: pd.DataFrame) -> np.ndarray:
        terms = self._extract_preference_terms(preferences)
        if not terms:
            return np.zeros(len(candidates))
        scores = []
        for _, row in candidates.iterrows():
            kw = str(row.get("keywords", "")).lower()
            ge = str(row.get("genres", "")).lower()
            ov = str(row.get("overview", "")).lower()
            s = (
                1.5 * sum(1 for t in terms if t in kw)
                + 1.2 * sum(1 for t in terms if t in ge)
                + 0.8 * sum(1 for t in terms if t in ov)
            )
            scores.append(s)
        arr = np.array(scores, dtype=float)
        return arr / arr.max() if arr.max() > 0 else arr

    def _genre_alignment_scores(self, preferences: str, candidates: pd.DataFrame) -> np.ndarray:
        requested = self._infer_requested_genres(preferences)
        if not requested:
            return np.zeros(len(candidates))
        scores = []
        for genres in candidates["genres"].fillna(""):
            movie_genres = {g.strip().lower() for g in str(genres).split(",") if g.strip()}
            overlap = len(requested & movie_genres)
            scores.append(overlap / max(len(requested), 1))
        return np.array(scores, dtype=float)

    def _tone_boost_scores(self, preferences: str, candidates: pd.DataFrame) -> np.ndarray:
        prefs_lower = str(preferences).lower()
        active = [
            (tone, terms)
            for tone, terms in TONE_KEYWORDS.items()
            if tone in prefs_lower or any(t in prefs_lower for t in terms)
        ]
        if not active:
            return np.zeros(len(candidates))
        scores = []
        for _, row in candidates.iterrows():
            kw = str(row.get("keywords", "")).lower()
            ov = str(row.get("overview", "")).lower()
            ge = str(row.get("genres", "")).lower()
            s = 0.0
            for _, terms in active:
                s += 1.2 * sum(1 for t in terms if t in kw)
                s += 0.8 * sum(1 for t in terms if t in ov)
                s += 0.6 * sum(1 for t in terms if t in ge)
            scores.append(s)
        arr = np.array(scores, dtype=float)
        return arr / arr.max() if arr.max() > 0 else arr

    def _blocked_genre_mask(self, preferences: str, candidates: pd.DataFrame) -> np.ndarray:
        blocked = self._infer_blocked_genres(preferences)
        if not blocked:
            return np.zeros(len(candidates))
        flags = []
        for genres in candidates["genres"].fillna(""):
            movie_genres = {g.strip().lower() for g in str(genres).split(",") if g.strip()}
            flags.append(1.0 if movie_genres & blocked else 0.0)
        return np.array(flags, dtype=float)

    def _watch_history_similarity(self, watched_ids: Set[int], candidate_tfidf) -> np.ndarray:
        if not watched_ids:
            return np.zeros(candidate_tfidf.shape[0])
        watched_rows = self.movies[self.movies["tmdb_id"].isin(watched_ids)]
        if watched_rows.empty:
            return np.zeros(candidate_tfidf.shape[0])
        watched_tfidf = self.tfidf_matrix[watched_rows.index.to_list()]
        sims = cosine_similarity(candidate_tfidf, watched_tfidf)
        avg = sims.mean(axis=1) if sims.size else np.zeros(candidate_tfidf.shape[0])
        return avg / np.max(avg) if np.max(avg) > 0 else avg

    # ------------------------- public API -------------------------

    def recommend(
        self,
        preferences: str,
        watch_history: List[str],
        watch_history_ids: Optional[List[int]] = None,
        top_k: int = 1,
    ) -> List[Tuple[int, str, float]]:
        watched_ids: Set[int] = set(watch_history_ids or [])
        for title in watch_history or []:
            movie = self._find_movie_by_title(str(title))
            if movie is not None:
                watched_ids.add(movie["tmdb_id"])

        mask = ~self.movies["tmdb_id"].isin(watched_ids)
        candidates = self.movies[mask].reset_index(drop=True)
        candidate_tfidf = self.tfidf_matrix[mask.to_numpy()]
        if candidates.empty:
            return []

        pref_vec = self.vectorizer.transform([preferences])
        text_sim = cosine_similarity(pref_vec, candidate_tfidf).flatten()
        if text_sim.max() > 0:
            text_sim = text_sim / text_sim.max()

        keyword_match = self._keyword_match_scores(preferences, candidates)
        genre_align = self._genre_alignment_scores(preferences, candidates)
        tone_boost = self._tone_boost_scores(preferences, candidates)
        blocked_mask = self._blocked_genre_mask(preferences, candidates)
        history_sim = self._watch_history_similarity(watched_ids, candidate_tfidf)

        vote_avg_norm = (candidates["vote_average"].fillna(5) / 10.0).to_numpy()
        vote_count_max = candidates["vote_count"].max() or 1
        vote_count_norm = (candidates["vote_count"].fillna(0) / vote_count_max).to_numpy()
        quality = 0.75 * vote_avg_norm + 0.25 * vote_count_norm

        low_q_penalty = np.where(vote_avg_norm < 0.60, 0.06, 0.0)
        very_low_q_penalty = np.where(vote_avg_norm < 0.50, 0.10, 0.0)

        final = (
            0.22 * keyword_match
            + 0.30 * text_sim
            + 0.18 * genre_align
            + 0.12 * tone_boost
            + 0.10 * history_sim
            + 0.08 * quality
        )
        # Small boosts for strong direct matches (mirrors production intent)
        final += (keyword_match >= 0.60).astype(float) * 0.03
        final += (genre_align >= 0.50).astype(float) * 0.02
        final += (tone_boost >= 0.50).astype(float) * 0.05
        # Hard penalty for blocked genres
        final -= 0.35 * blocked_mask
        # Quality penalties
        final -= low_q_penalty + very_low_q_penalty

        top_indices = np.argsort(final)[::-1][:top_k]
        return [
            (int(candidates.iloc[i]["tmdb_id"]), str(candidates.iloc[i]["title"]), float(final[i]))
            for i in top_indices
        ]

    def _find_movie_by_title(self, title: str) -> Optional[Dict]:
        title_lower = title.lower().strip()
        exact = self.movies[self.movies["title"].str.lower() == title_lower]
        if not exact.empty:
            r = exact.iloc[0]
            return {"tmdb_id": int(r["tmdb_id"]), "title": str(r["title"])}
        fuzzy = self.movies[self.movies["title"].str.lower().str.contains(title_lower, na=False, regex=False)]
        if not fuzzy.empty:
            r = fuzzy.iloc[0]
            return {"tmdb_id": int(r["tmdb_id"]), "title": str(r["title"])}
        return None


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


class SimilarityMetrics:
    """Set- and text-based similarity between two movies."""

    @staticmethod
    def jaccard(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(1, len(set_a | set_b))

    @staticmethod
    def parse_delimited(field: str, delimiter: str = ",") -> set:
        if pd.isna(field):
            return set()
        return {t.strip().lower() for t in str(field).split(delimiter) if t.strip()}

    @staticmethod
    def genre_similarity(a: str, b: str) -> float:
        return SimilarityMetrics.jaccard(
            SimilarityMetrics.parse_delimited(a),
            SimilarityMetrics.parse_delimited(b),
        )

    @staticmethod
    def keyword_similarity(a: str, b: str) -> float:
        return SimilarityMetrics.jaccard(
            SimilarityMetrics.parse_delimited(a),
            SimilarityMetrics.parse_delimited(b),
        )

    @staticmethod
    def overview_similarity(a: str, b: str) -> float:
        if pd.isna(a) or pd.isna(b):
            return 0.0
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=100, lowercase=True)
            m = vec.fit_transform([str(a), str(b)])
            if m.shape[0] < 2:
                return 0.0
            sim = cosine_similarity(m[0], m[1])[0, 0]
            return float(max(0.0, min(1.0, sim)))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Validation framework
# ---------------------------------------------------------------------------


class ValidationFramework:
    """
    Runs the agent and the baseline on the same cases and reports the
    per-case similarity breakdown + aggregate metrics.
    """

    SOFT_MATCH_THRESHOLD = 0.60

    def __init__(
        self,
        movies_df: pd.DataFrame,
        candidate_ids: Optional[Set[int]] = None,
        min_year: int = 2010,
    ):
        self.dataset = MovieDataset.__new__(MovieDataset)
        self.dataset.df = movies_df.copy()
        self.dataset._preprocess()

        self.baseline = BaselineRecommender(
            movies_df,
            candidate_ids=candidate_ids,
            min_year=min_year,
        )
        self.results: List[Dict] = []

    # ------------------------- single-case validation -------------------------

    def validate_case(
        self,
        case_id: str,
        preferences: str,
        watch_history: List[str],
        agent_tmdb_id: int,
        watch_history_ids: Optional[List[int]] = None,
        agent_description: str = "",
        agent_latency_s: Optional[float] = None,
    ) -> Dict:
        """
        Validate a single case given a pre-computed agent pick.

        For the live-agent flow (recommended), use `validate_with_agent()`.
        """
        baseline_recs = self.baseline.recommend(
            preferences,
            watch_history,
            watch_history_ids=watch_history_ids,
            top_k=100,
        )
        if not baseline_recs:
            result = {"case_id": case_id, "status": "error_no_candidates", "error": "No candidate movies available"}
            self.results.append(result)
            return result

        baseline_tmdb_id, baseline_title, baseline_score = baseline_recs[0]
        top5_ids = [r[0] for r in baseline_recs[:5]]
        top10_ids = [r[0] for r in baseline_recs[:10]]
        top100_ids = [r[0] for r in baseline_recs[:100]]
        baseline_rank = next((i + 1 for i, r in enumerate(baseline_recs) if r[0] == agent_tmdb_id), None)
        rank_score = max(0, 101 - baseline_rank) if baseline_rank is not None else 0

        baseline_movie = self.dataset.get_movie_by_id(baseline_tmdb_id)
        agent_movie = self.dataset.get_movie_by_id(agent_tmdb_id)

        if baseline_movie is None:
            result = {"case_id": case_id, "status": "error_baseline_not_found", "error": f"Baseline movie not found: {baseline_tmdb_id}"}
            self.results.append(result)
            return result
        if agent_movie is None:
            result = {"case_id": case_id, "status": "error_agent_not_found", "error": f"Agent movie not found: {agent_tmdb_id}"}
            self.results.append(result)
            return result

        genre_sim = SimilarityMetrics.genre_similarity(baseline_movie["genres"], agent_movie["genres"])
        keyword_sim = SimilarityMetrics.keyword_similarity(baseline_movie["keywords"], agent_movie["keywords"])
        overview_sim = SimilarityMetrics.overview_similarity(baseline_movie["overview"], agent_movie["overview"])
        hybrid_sim = (
            0.40 * overview_sim
            + 0.25 * genre_sim
            + 0.25 * keyword_sim
            + 0.10 * (1.0 if (genre_sim > 0.3 or keyword_sim > 0.3) else 0.0)
        )

        agent_pref = SimilarityMetrics.overview_similarity(preferences, agent_movie["overview"])
        baseline_pref = SimilarityMetrics.overview_similarity(preferences, baseline_movie["overview"])

        desc_len = len(str(agent_description or ""))
        desc_ok = 1 if 1 <= desc_len <= 500 else 0
        history_ok = 1 if agent_tmdb_id not in set(watch_history_ids or []) else 0

        result = {
            "case_id": case_id,
            "status": "success",
            "preferences_snippet": str(preferences)[:80],
            "watch_history_count": len(watch_history or []),
            "baseline_tmdb_id": baseline_tmdb_id,
            "baseline_title": baseline_title,
            "baseline_score": baseline_score,
            "baseline_rank": baseline_rank,
            "rank_score": rank_score,
            "agent_tmdb_id": agent_tmdb_id,
            "agent_title": agent_movie["title"],
            "agent_description_length": desc_len,
            "agent_description_within_limit": desc_ok,
            "agent_respected_history": history_ok,
            "agent_latency_s": float(agent_latency_s) if agent_latency_s is not None else None,
            "exact_match": 1 if baseline_tmdb_id == agent_tmdb_id else 0,
            "soft_match": 1 if hybrid_sim >= self.SOFT_MATCH_THRESHOLD else 0,
            "top5_hit": 1 if agent_tmdb_id in top5_ids else 0,
            "top10_hit": 1 if agent_tmdb_id in top10_ids else 0,
            "top100_hit": 1 if agent_tmdb_id in top100_ids else 0,
            "genre_similarity": genre_sim,
            "keyword_similarity": keyword_sim,
            "overview_similarity": overview_sim,
            "hybrid_similarity": hybrid_sim,
            "agent_pref_alignment": agent_pref,
            "baseline_pref_alignment": baseline_pref,
        }

        self.results.append(result)
        return result

    def validate_with_agent(
        self,
        case_id: str,
        preferences: str,
        watch_history: List[str],
        watch_history_ids: Optional[List[int]] = None,
        agent_callable=None,
    ) -> Dict:
        """
        Convenience entry point that (a) calls the live agent, (b) validates.

        `agent_callable` should be a function with the same signature as
        `llm.get_recommendation(preferences, history, history_ids)` returning
        `{"tmdb_id": int, "description": str}`. If omitted, we import and use
        `llm.get_recommendation` directly.
        """
        if agent_callable is None:
            from llm import get_recommendation as agent_callable  # type: ignore[no-redef]

        start = time.perf_counter()
        try:
            out = agent_callable(preferences, list(watch_history or []), list(watch_history_ids or []))
            elapsed = time.perf_counter() - start
        except Exception as exc:
            elapsed = time.perf_counter() - start
            result = {
                "case_id": case_id,
                "status": "error_agent_exception",
                "error": f"Agent raised: {exc!r}",
                "agent_latency_s": float(elapsed),
            }
            self.results.append(result)
            return result

        try:
            agent_tmdb_id = int(out.get("tmdb_id"))
            agent_description = str(out.get("description", ""))
        except Exception as exc:
            result = {
                "case_id": case_id,
                "status": "error_agent_bad_shape",
                "error": f"Agent returned invalid shape: {exc!r}",
                "agent_latency_s": float(elapsed),
            }
            self.results.append(result)
            return result

        return self.validate_case(
            case_id=case_id,
            preferences=preferences,
            watch_history=watch_history,
            watch_history_ids=watch_history_ids,
            agent_tmdb_id=agent_tmdb_id,
            agent_description=agent_description,
            agent_latency_s=elapsed,
        )

    # ------------------------- reporting -------------------------

    def get_summary_report(self) -> Dict:
        if not self.results:
            return {"error": "No results to report"}

        successful = [r for r in self.results if r.get("status") == "success"]
        if not successful:
            return {
                "total_cases": len(self.results),
                "successful_cases": 0,
                "failed_cases": len(self.results),
                "error": "No successful validations",
            }

        df = pd.DataFrame(successful)
        metrics = {
            "exact_match_rate": float(df["exact_match"].mean()),
            "soft_match_rate": float(df["soft_match"].mean()),
            "avg_rank_score": float(df["rank_score"].mean()),
            "top5_hit_rate": float(df["top5_hit"].mean()),
            "top10_hit_rate": float(df["top10_hit"].mean()),
            "top100_hit_rate": float(df["top100_hit"].mean()),
            "avg_genre_similarity": float(df["genre_similarity"].mean()),
            "avg_keyword_similarity": float(df["keyword_similarity"].mean()),
            "avg_overview_similarity": float(df["overview_similarity"].mean()),
            "avg_hybrid_similarity": float(df["hybrid_similarity"].mean()),
            "avg_agent_pref_alignment": float(df["agent_pref_alignment"].mean()),
            "avg_baseline_pref_alignment": float(df["baseline_pref_alignment"].mean()),
            "max_hybrid_similarity": float(df["hybrid_similarity"].max()),
            "min_hybrid_similarity": float(df["hybrid_similarity"].min()),
            "description_within_limit_rate": float(df["agent_description_within_limit"].mean()),
            "history_respected_rate": float(df["agent_respected_history"].mean()),
        }
        if "agent_latency_s" in df.columns and df["agent_latency_s"].notna().any():
            metrics["avg_agent_latency_s"] = float(df["agent_latency_s"].dropna().mean())
            metrics["max_agent_latency_s"] = float(df["agent_latency_s"].dropna().max())

        return {
            "total_cases": len(self.results),
            "successful_cases": len(successful),
            "failed_cases": len(self.results) - len(successful),
            "soft_match_threshold": self.SOFT_MATCH_THRESHOLD,
            "metrics": metrics,
        }

    def print_report(self) -> None:
        report = self.get_summary_report()
        if "error" in report and report.get("successful_cases", 0) == 0:
            print("\n" + "=" * 75)
            print("VALIDATION REPORT - ERROR")
            print("=" * 75)
            print(f"Error: {report['error']}")
            return

        metrics = report.get("metrics", {})
        print("\n" + "=" * 75)
        print("VALIDATION REPORT - AGENT VS BASELINE RECOMMENDER")
        print("=" * 75)
        print("\nTest Coverage:")
        print(f"  Total Cases:       {report['total_cases']}")
        print(f"  Successful:        {report['successful_cases']}")
        print(f"  Failed:            {report['failed_cases']}")

        print("\n" + "-" * 75)
        print("PRIMARY ACCURACY METRICS")
        print("-" * 75)
        print(f"  Avg Rank Score:     {metrics.get('avg_rank_score', 0):.2f}")
        print("    (Rank 1 = 100, Rank 100 = 1, outside top-100 = 0)")
        print(f"  Exact Match Rate:   {metrics.get('exact_match_rate', 0):.1%}")
        print(f"  Soft Match Rate:    {metrics.get('soft_match_rate', 0):.1%}")
        print(f"    (hybrid similarity >= {self.SOFT_MATCH_THRESHOLD})")

        print("\n" + "-" * 75)
        print("TOP-K HIT RATES (agent pick in baseline top-K)")
        print("-" * 75)
        print(f"  Top-5  Hit Rate:    {metrics.get('top5_hit_rate', 0):.1%}")
        print(f"  Top-10 Hit Rate:    {metrics.get('top10_hit_rate', 0):.1%}")
        print(f"  Top-100 Hit Rate:   {metrics.get('top100_hit_rate', 0):.1%}")

        print("\n" + "-" * 75)
        print("AVERAGE SIMILARITY METRICS")
        print("-" * 75)
        print(f"  Genre Overlap:      {metrics.get('avg_genre_similarity', 0):.3f}")
        print(f"  Keyword Overlap:    {metrics.get('avg_keyword_similarity', 0):.3f}")
        print(f"  Overview Text:      {metrics.get('avg_overview_similarity', 0):.3f}")
        print(f"  Hybrid Composite:   {metrics.get('avg_hybrid_similarity', 0):.3f}")
        print(f"    Range: [{metrics.get('min_hybrid_similarity', 0):.3f}, {metrics.get('max_hybrid_similarity', 0):.3f}]")

        print("\n" + "-" * 75)
        print("PREFERENCE ALIGNMENT (Sanity Checks)")
        print("-" * 75)
        print(f"  Agent Alignment:    {metrics.get('avg_agent_pref_alignment', 0):.3f}")
        print(f"  Baseline Alignment: {metrics.get('avg_baseline_pref_alignment', 0):.3f}")

        print("\n" + "-" * 75)
        print("AGENT CONTRACT CHECKS")
        print("-" * 75)
        print(f"  Description ≤ 500 chars:  {metrics.get('description_within_limit_rate', 0):.1%}")
        print(f"  History Respected:        {metrics.get('history_respected_rate', 0):.1%}")
        if "avg_agent_latency_s" in metrics:
            print(f"  Avg Agent Latency:        {metrics['avg_agent_latency_s']:.2f}s")
            print(f"  Max Agent Latency:        {metrics['max_agent_latency_s']:.2f}s (20s DQ limit)")
        print("\n" + "=" * 75 + "\n")

    # ------------------------- persistence -------------------------

    def save_results_csv(self, output_path: str) -> None:
        if not self.results:
            print("No results to save")
            return
        pd.DataFrame(self.results).to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def save_summary_json(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary_report(), f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {output_path}")


# ---------------------------------------------------------------------------
# Helpers for integrating with this repo
# ---------------------------------------------------------------------------


def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load tmdb_top1000_movies.csv (defaults to the one next to this file)."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
    return pd.read_csv(data_path)


def default_candidate_ids() -> Set[int]:
    """Return the agent's candidate pool so the baseline competes on the same ground."""
    try:
        from llm import TOP_MOVIES  # lazy import: avoid forcing users who only want the baseline
        return set(int(x) for x in TOP_MOVIES["tmdb_id"].astype(int).to_list())
    except Exception:
        return set()


def build_validator(
    movies_df: Optional[pd.DataFrame] = None,
    use_agent_pool: bool = True,
    min_year: int = 2010,
) -> ValidationFramework:
    """
    Construct a ValidationFramework with sensible defaults for this repo:
    - movies_df defaults to the local CSV
    - by default the baseline is restricted to llm.TOP_MOVIES (fair comparison)
    """
    if movies_df is None:
        movies_df = load_dataset()
    candidate_ids = default_candidate_ids() if use_agent_pool else None
    return ValidationFramework(movies_df, candidate_ids=candidate_ids, min_year=min_year)


# ---------------------------------------------------------------------------
# CLI entry point: run validation against the live agent across eval cases
# ---------------------------------------------------------------------------


def _load_eval_cases(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for i, c in enumerate(cases):
        c.setdefault("case_id", f"case_{i+1:03d}")
    return cases


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run validation (baseline vs live agent).")
    parser.add_argument(
        "--cases",
        default=os.path.join(os.path.dirname(__file__), "dspy_gepa_eval_cases.json"),
        help="Path to eval cases JSON (default: dspy_gepa_eval_cases.json).",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv"),
        help="Path to TMDB CSV.",
    )
    parser.add_argument(
        "--full-pool",
        action="store_true",
        help="Let the baseline search the entire CSV instead of just llm.TOP_MOVIES.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases (0 = all).")
    parser.add_argument(
        "--results-csv",
        default=os.path.join(os.path.dirname(__file__), "validation_results.csv"),
    )
    parser.add_argument(
        "--summary-json",
        default=os.path.join(os.path.dirname(__file__), "validation_summary.json"),
    )
    args = parser.parse_args()

    if not os.environ.get("OLLAMA_API_KEY"):
        print("Warning: OLLAMA_API_KEY not set — the agent will use the deterministic fallback path.")
        print("         (Validation will still run, but latency and quality will reflect fallback behavior.)")

    print(f"Loading dataset: {args.data}")
    movies_df = pd.read_csv(args.data)

    print(f"Loading eval cases: {args.cases}")
    cases = _load_eval_cases(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"  -> {len(cases)} cases to validate")

    print("Building validator ...")
    validator = build_validator(movies_df=movies_df, use_agent_pool=not args.full_pool)
    print(f"  Baseline pool size: {len(validator.baseline.movies)} movies")

    print("\nRunning live-agent validation (this makes one get_recommendation() call per case)")
    for i, case in enumerate(cases, 1):
        cid = case["case_id"]
        prefs_preview = str(case.get("preferences", ""))[:60]
        print(f"  [{i:02d}/{len(cases)}] {cid}: {prefs_preview}")
        result = validator.validate_with_agent(
            case_id=cid,
            preferences=case["preferences"],
            watch_history=case.get("history", []) or [],
            watch_history_ids=case.get("history_ids", []) or [],
        )
        status = result.get("status", "unknown")
        if status == "success":
            lat = result.get("agent_latency_s")
            lat_str = f"{lat:.2f}s" if lat is not None else "-"
            print(
                f"       -> rank={result.get('baseline_rank')} "
                f"hybrid={result.get('hybrid_similarity', 0):.2f} "
                f"latency={lat_str} "
                f"agent='{result.get('agent_title')}' vs baseline='{result.get('baseline_title')}'"
            )
        else:
            print(f"       -> {status}: {result.get('error', '')}")

    validator.print_report()
    validator.save_results_csv(args.results_csv)
    validator.save_summary_json(args.summary_json)
