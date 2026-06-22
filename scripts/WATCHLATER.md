# Watchlater Processing Workflow

Turns a YouTube Watch Later export into a structured, AI-filtered learning queue in two steps.

```
data/watchlater.txt  →  (enrich)  →  data/watchlater.json  →  (import)  →  data/queue.json
```

---

## Step 1 — Enrich (`enrich_watchlater.py`)

Reads `watchlater.txt` (format: `title , url` per line), fetches YouTube metadata via yt-dlp,
auto-classifies each video by topic, and drops non-AI content.

**Output:** `data/watchlater.json` — one entry per AI-relevant video with:
`title`, `url`, `channel`, `upload_date`, `duration_seconds`, `description_snippet`, `topic`, `queued`

**Topics:** LLMs · RAG · Agents · Embeddings · Fine-tuning · RL · Deep Learning · MLOps · AI Engineering · Graph ML · Evaluation · Career

```bash
# Enrich all entries
make enrich-watchlater
# or:
uv run python scripts/enrich_watchlater.py

# Resume after interruption (skips already-enriched URLs)
uv run python scripts/enrich_watchlater.py --resume

# Test run — first 5 entries only
uv run python scripts/enrich_watchlater.py --limit 5

# Include non-AI content (no topic filter)
uv run python scripts/enrich_watchlater.py --all
```

---

## Step 2 — Import (`import_watchlater.py`)

Reads `watchlater.json`, fetches full YouTube transcripts, and stages each video in
`data/queue.json` for studying. Does NOT add to the knowledge base — you must study each
resource first (via the app), then consolidate it.

```bash
# Preview without fetching transcripts
uv run python scripts/import_watchlater.py --dry-run

# Import all 78 AI-relevant entries (slow — fetches each transcript)
make import-watchlater

# Import a single topic (recommended starting point)
uv run python scripts/import_watchlater.py --topic RAG
uv run python scripts/import_watchlater.py --topic Agents --difficulty intermediate

# First N entries only
uv run python scripts/import_watchlater.py --limit 5

# Filter + limit together
uv run python scripts/import_watchlater.py --topic LLMs --limit 3 --dry-run
```

**Available topics** (case-insensitive substring match):
`LLMs`, `RAG`, `Agents`, `Embeddings`, `Fine-tuning`, `RL`, `Deep Learning`,
`MLOps`, `AI Engineering`, `Graph ML`, `Evaluation`, `Career`

---

## Watchlater.txt format

The script expects one entry per line:

```
Video Title , https://www.youtube.com/watch?v=xxxxx
Another Video , https://youtu.be/yyyyy
```

Export yours from YouTube → Library → Watch Later → three-dot menu → Save playlist.
The exported format varies by tool; the parser handles the `title , url` pattern.

---

## After importing

Open the app (`make run`) → go to **Queue** page to see all staged videos with previews.
Click **▶ Start learning this** on any entry to begin an interactive learning session.
When done, the agent will prompt you to consolidate it into the knowledge base.
