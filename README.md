# Movie Recommender

Your task is to implement `get_recommendation()` in `llm.py`. The function receives a user's movie preferences and watch history, calls an LLM, and returns a movie recommendation.

---

## What to submit

Submit a **zip file** containing at minimum:

- `llm.py` — your implementation
- `requirements.txt` — any packages your code needs (one per line, e.g. `ollama`, `pandas`)

You may include additional files (e.g. helper modules, data files) if your implementation needs them. Do not include your `.env` file, the `.venv/` folder, or `__pycache__`. Keep the zip under **10 MB**.

**Do NOT hard-code your API key.** The grader will inject `OLLAMA_API_KEY` at run time. Read it from the environment:

```python
os.environ["OLLAMA_API_KEY"]   # good
os.getenv("OLLAMA_API_KEY")    # also fine
```

If you need additional environment variables (e.g. a key for TMDB), list them in a comment at the top of `llm.py`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you add packages, add them to `requirements.txt` — one package name per line:

```
ollama
pandas
requests
```

You can also pin versions if needed: `ollama>=0.4.0`. The grader runs `pip install -r requirements.txt` before calling your code.

---

## Development workflow

Get a free API key at [ollama.com/settings/keys](https://ollama.com/settings/keys), then prefix it on any command you run:

```bash
OLLAMA_API_KEY=your_key_here python llm.py --preferences "I want a funny, light, action-packed movie." --history "The Dark Knight Rises"
```

Omit either flag and you'll be prompted interactively:

```bash
OLLAMA_API_KEY=your_key_here python llm.py
# Preferences: I love sci-fi thrillers
# Watch history (optional): Inception
```

When you're happy with the output, run the test suite:

```bash
OLLAMA_API_KEY=your_key_here python test.py
```

If you'd rather not type the key every time, you can export it for your current terminal session:

```bash
export OLLAMA_API_KEY=your_key_here
python test.py   # key is picked up automatically
```

This checks that your `get_recommendation()` returns a valid response: correct keys, a `tmdb_id` from the candidate list, no repeats from watch history, and within the time limit.

---

## The function signature

```python
def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    ...
```

| Argument | Type | Description |
|---|---|---|
| `preferences` | `str` | Free-text description of what the user wants to watch |
| `history` | `list[str]` | Movie titles the user has already seen |
| `history_ids` | `list[int]` | TMDB IDs corresponding to `history` |

Return a `dict` with:

| Key | Type | Description |
|---|---|---|
| `tmdb_id` | `int` | Must be from the candidate list in `TOP_MOVIES` |
| `description` | `str` | A short pitch (≤500 chars) explaining why this movie fits |

---

## Ideas for improvement

- Expand the candidate pool beyond the top 5 (filter by genre first, then rank).
- Include more metadata in the prompt (genres, cast, keywords).
- Use watch history to steer away from similar movies.
- Try chain-of-thought or few-shot prompting.
- Cache responses for repeated inputs to stay under the time limit.
