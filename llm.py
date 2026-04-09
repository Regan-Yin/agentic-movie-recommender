"""
TODO: This is the file you should edit.

get_recommendation() is called once per request with the user's input and the
candidate DataFrame. It should return a dict with keys "tmdb_id" and
"description".

build_prompt() and call_llm() are broken out as separate functions so they are
easy to swap or extend individually, but you are free to restructure this file
however you like.
"""

import json
import os

import pandas as pd
from openai import OpenAI

MODEL = "gpt-5-nano"


def build_prompt(preferences: str, history: list[str], candidates: pd.DataFrame) -> str:
    movie_list = "\n".join(
        f'- tmdb_id={row.tmdb_id} | "{row.title}" ({row.year}) | genres: {row.genres} | overview: {row.overview[:200]}'
        for row in candidates.itertuples()
    )
    history_text = ", ".join(f'"{name}"' for name in history) if history else "none"
    return f"""You are a movie recommendation assistant.

A user is looking for a movie to watch. Here are their preferences:
"{preferences}"

Movies they have already watched (do NOT recommend these):
{history_text}

Below is the list of candidate movies you may recommend. You MUST pick exactly one.

{movie_list}

Respond with ONLY a JSON object — no markdown, no extra text — in this exact format:
{{
  "tmdb_id": <integer>,
  "description": "<a compelling blurb ≤500 chars that tells the user why this movie matches their preferences>"
}}"""


def call_llm(prompt: str) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def get_recommendation(
    preferences: str,
    history: list[str],
    candidates: pd.DataFrame,
) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    prompt = build_prompt(preferences, history, candidates)
    return call_llm(prompt)


if __name__ == "__main__":
    import os, pandas as pd

    DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
    candidates = pd.read_csv(DATA_PATH).nlargest(40, "vote_count")

    print("Movie recommender – type your preferences and press Enter.")
    print("For watch history, enter comma-separated movie titles (or leave blank).")
    print()

    preferences = input("Preferences: ").strip()
    history_raw = input("Watch history (optional): ").strip()
    history = [t.strip() for t in history_raw.split(",")] if history_raw else []

    print("\nThinking...\n")
    result = get_recommendation(preferences, history, candidates)

    match = candidates[candidates["tmdb_id"] == result["tmdb_id"]]
    title = match.iloc[0]["title"] if not match.empty else "unknown"

    print(f"Recommendation: {title} (tmdb_id={result['tmdb_id']})")
    print(f"\n{result['description']}")
