# AI Learning OS

An AI Learning Chief of Staff with a personal knowledge brain — built as my AI Engineering Buildcamp capstone.

<!-- Screenshots: add images to docs/screenshots/ and update the paths below -->

## The Problem

Ambitious self-learners save dozens of YouTube videos, articles, and PDFs to "watch/read later" — but most of it never gets consumed, and even what does get consumed rarely turns into structured, retrievable knowledge. Resources are scattered across browser bookmarks, YouTube playlists, and notes apps. There is no system connecting saved content to what you actually know.

## What It Does

AI Learning OS solves the watch/read-later graveyard problem with two distinct states:

- **Queue** — content you've saved but not yet studied (YouTube videos, articles, markdown files, images)
- **Knowledge Base** — content you've actively learned from, chunked, embedded, and indexed for retrieval

A typical session: you tell the agent to add a YouTube video to your queue. Later, you ask it to walk you through the content — the agent loads the full transcript and explains key concepts. When you're done, you tell the agent to consolidate it into your knowledge base. From that point on, you can ask "what do I know about attention mechanisms?" and get cited answers drawn from your own studied material.

**The knowledge base only contains things you've actually learned.** This makes retrieval honest — the agent synthesizes from your real understanding, not a pile of bookmarks.

## Screenshots

| Chat — learning session with tool calls visible | Queue — staged resources with previews |
|---|---|
| ![Chat page](docs/screenshots/chat-learning.png) | ![Queue page](docs/screenshots/queue.png) |

| KB search with source citations | Monitoring dashboard |
|---|---|
| ![KB search](docs/screenshots/kb-search.png) | ![Monitoring](docs/screenshots/monitoring.png) |

## Architecture

```
User (Streamlit chat)
        │
        ▼
   Agent Loop  ──── 6 tools ────▶  ai_learning_os/tools.py
        │
        ├── Ingest layer          ai_learning_os/ingest.py
        │   ├── YouTube transcript (youtube-transcript-api)
        │   ├── Web article (BeautifulSoup)
        │   ├── Markdown file (marker output + vision LLM for images)
        │   └── Image file (vision LLM)
        │
        ├── Hybrid search         ai_learning_os/retrieval.py
        │   ├── BM25 (rank-bm25)
        │   ├── Dense embeddings (OpenAI text-embedding-3-small)
        │   └── Reciprocal Rank Fusion (RRF)
        │
        └── Monitoring            ai_learning_os/monitoring.py
            └── logs.jsonl → Streamlit dashboard
```

## Project Structure

```
ai-learning-os/
├── ai_learning_os/       # core Python package
│   ├── config.py         # settings (all overridable via ALOS_* env vars)
│   ├── ingest.py         # content fetching for all source types
│   ├── retrieval.py      # hybrid search: BM25 + embeddings + RRF
│   ├── tools.py          # 6 agent tools + OpenAI function schemas
│   ├── agent.py          # agent loop (plain OpenAI chat completions)
│   └── monitoring.py     # JSONL logging + feedback update
├── app.py                # Streamlit UI (Chat | Queue | Monitoring)
├── evals/
│   ├── ground_truth.json # 15 hand-crafted Q&A evaluation pairs
│   ├── judge.py          # LLM judge (structured output via gpt-4o-mini)
│   └── run_eval.py       # eval runner + tuning mode (--tune flag)
├── tests/
│   ├── conftest.py       # fixtures (tmp config, sample KB/queue entries)
│   ├── judge.py          # test helper wrapping evals/judge.py
│   └── test_agent.py     # 5 scenario tests (unit + LLM judge)
├── notebooks/
│   ├── 02-rag.ipynb      # RAG baseline (course artifact)
│   └── retrieval_eval.ipynb  # retrieval comparison: TF-IDF vs BM25 vs Hybrid
├── scripts/
│   └── import_watchlater.py  # batch-import YouTube URLs from data/watchlater.txt
├── data/
│   ├── queue.json        # staged resources (not yet learned)
│   ├── kb.json           # knowledge base with chunks + embeddings
│   ├── progress.json     # time_by_topic, concepts_mastered
│   └── logs.jsonl        # monitoring log (one JSON line per interaction)
├── Makefile
├── .env.example
└── pyproject.toml
```

## The 6 Agent Tools

| Tool | Description |
|------|-------------|
| `add_to_queue` | Fetch full content from a YouTube URL, article URL, `.md` file, or image and stage it for studying |
| `browse_queue` | List unlearned resources with topic, difficulty, and word count |
| `load_for_learning` | Load a queued resource's full text for an in-session learning walkthrough |
| `consolidate_to_kb` | Generate synthesis + key concepts, chunk and embed full text, index into KB |
| `search_knowledge_base` | Hybrid search (BM25 + semantic + RRF) over KB chunks; returns cited passages |
| `get_progress` | Queue depth, KB size, topics covered, time by topic, concepts mastered |

## Supported Input Types

| Source | How it's ingested |
|--------|------------------|
| YouTube URL | Full transcript via `youtube-transcript-api` |
| Web article URL | HTML scrape + tag stripping via BeautifulSoup |
| Markdown file (`.md`) | Direct read; image references described by vision LLM |
| Image file (`.png/.jpg/.webp`) | Described by vision LLM |

**PDF files:** Convert to markdown first using [marker](https://github.com/datalab-to/marker) (`marker input.pdf`), then add the output `.md` file.

## Setup

**1. Install uv** (if you don't have it):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and install dependencies:**
```bash
git clone <repo-url>
cd ai-learning-os
uv sync
```

**3. Configure API key:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

**4. Run the app:**
```bash
make run
# or: uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Running with Docker

No Python setup required — the entire app runs in one command:

```bash
cp .env.example .env   # add your OPENAI_API_KEY
docker compose up --build
```

Open http://localhost:8501. The knowledge base, queue, and logs persist via a volume mount on `./data/` so your content survives container restarts.

## Usage

```bash
make run              # launch Streamlit UI
make test             # run unit + judge tests
make eval             # run LLM evaluation over ground truth dataset
make eval-tune        # compare chunk sizes and retrieval configs
make notebook         # start Jupyter for retrieval_eval.ipynb
make import-watchlater  # batch-import data/watchlater.txt
```

## Optional Config

Override defaults via environment variables (all prefixed `ALOS_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ALOS_LLM_MODEL` | `gpt-4o-mini` | Main agent LLM |
| `ALOS_VISION_MODEL` | `gpt-4o-mini` | Vision LLM for image descriptions |
| `ALOS_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ALOS_CHUNK_SIZE_WORDS` | `400` | Chunk size in words |
| `ALOS_CHUNK_OVERLAP_WORDS` | `50` | Overlap between chunks |
| `ALOS_SEARCH_NUM_RESULTS` | `5` | Default number of search results |

## Retrieval Approach

Three approaches were compared in `notebooks/retrieval_eval.ipynb`:

| Approach | Description |
|----------|-------------|
| Baseline | TF-IDF (minsearch) over short resource descriptions |
| BM25 chunks | BM25Okapi over 400-word content chunks |
| **Hybrid (chosen)** | BM25 + OpenAI embeddings + Reciprocal Rank Fusion |

Hybrid search was chosen because BM25 handles exact keyword queries better than TF-IDF, dense embeddings capture semantic similarity that keyword search misses, and RRF fusion is robust without weight tuning. See `notebooks/retrieval_eval.ipynb` for results.

## Testing

```bash
make test
# or: uv run pytest tests/ -v -s
```

5 tests covering:
1. Tool call order — `get_progress` before `search_knowledge_base` in study plans
2. Source citations — KB answers must cite source titles
3. Empty KB — no hallucination when knowledge base is empty
4. Queue browsing — `browse_queue` called and titles returned correctly
5. LLM judge — output quality evaluation on a full study plan query

## Evaluation

```bash
make eval        # run judge over 15 hand-crafted ground truth entries
make eval-tune   # compare chunk sizes [200/400/600] and num_results [3/5/7]
```

Results saved to `evals/results/run_<timestamp>.json`.

The ground truth dataset (`evals/ground_truth.json`) is **hand-crafted** — not LLM-generated. 15 questions were written manually to cover KB search, study planning, queue browsing, progress tracking, and edge cases (empty KB, out-of-scope topics). Each entry specifies the expected tools to call and 2–4 per-criterion pass/fail verdicts from the LLM judge.

### Manual evaluation

Five agent responses were manually reviewed against the ground truth questions, assessing relevance and citation accuracy:

| Question | Relevant? | Citations correct? | Notes |
|----------|-----------|-------------------|-------|
| What do I know about RAG? | ✅ Yes | ✅ Yes | Cited AdamW and Vector DB sources correctly |
| Explain AdamW vs Adam | ✅ Yes | ✅ Yes | Grounded in transcript; weight decay explained accurately |
| 3-hour study plan | ✅ Yes | ✅ Yes | Time blocks summed to 3h; recommended real queue items |
| What topics have I covered? | ✅ Yes | n/a | Correctly separated KB topics from queue topics |
| Topic not in KB (quantum computing) | ✅ Yes | n/a | Correctly said "not studied yet"; no hallucination |

**Result: 5/5 relevant, 3/3 citation checks passed.** No hallucinated content detected in sampled responses.

### Retrieval tuning results

`make eval-tune` compares three chunk size × num_results configurations:

| Config | chunk_size | num_results | Pass rate |
|--------|-----------|-------------|-----------|
| Small | 200 words | 3 | 15/15 (100%) |
| **Medium (chosen)** | **400 words** | **5** | **15/15 (100%)** |
| Large | 600 words | 7 | 12/15 (80%) |

Medium chunks (400 words, 5 results) was selected as the default. Large chunks (600 words) underperformed because oversized chunks dilute keyword signal for BM25 and return fewer distinct passages for RRF to re-rank.

### Sample evaluation results

```
Running eval on 15 ground truth entries...
Config: model=gpt-4o-mini, chunk_size=400, num_results=5

  gt-01  PASS  What do I know about attention mechanisms from my knowledge base?
  gt-02  PASS  I have 3 hours this weekend. What should I focus on to get closer to being job-ready?
  gt-03  PASS  What topics do I have in my queue that I haven't studied yet?
  gt-04  PASS  Explain backpropagation based on what I've learned.
  gt-05  PASS  What do I know about RAG from my knowledge base?
  gt-06  PASS  I want to learn about quantum computing. What do I have on that?
  gt-07  PASS  How many concepts have I mastered so far?
  gt-08  PASS  Based on what I've studied, how does AdamW differ from the standard Adam optimizer?
  gt-09  PASS  What topics have I studied so far, and how many resources are in my knowledge base vs queue?
  gt-12  PASS  What AI engineering topics have I already covered in my knowledge base?
  gt-13  PASS  What do I know about vector databases and semantic search from my knowledge base?
  gt-14  PASS  What should I study first to understand how agents work?
  gt-15  PASS  Let's go through the first item in my queue together.
  gt-10  PASS  Add this YouTube video to my queue: https://www.youtube.com/watch?v=dQw4w9WgXcQ
  gt-11  PASS  I just finished studying the first resource in my queue. Add it to my knowledge base.

RESULT: 15/15 passed (100.0%)  |  avg latency: ~5s per question
```

## Monitoring

Every agent interaction is automatically logged to `data/logs.jsonl` (one JSON line per turn, including tool calls, latency, and feedback).

**To access the dashboard:** run `make run` and click **Monitoring** in the sidebar. The dashboard shows:
- KPI row: total interactions, average latency, feedback score (% thumbs up), unique sessions
- Tool call frequency bar chart (all 6 tools)
- Recent interactions table (last 20 turns)
- Export button for thumbs-up interactions

**User feedback:** Every agent response in Chat has 👍/👎 buttons. Ratings are written back to `logs.jsonl` and reflected live in the feedback score metric.

**Logs → evals pipeline:** The Monitoring page's "Export thumbs-up interactions" button writes highly-rated interactions to `evals/ground_truth_candidates.json`. These can be manually reviewed and promoted to `evals/ground_truth.json` to expand the evaluation dataset over time.

## Self-Evaluation

| Criterion | Score | Evidence |
|-----------|-------|----------|
| Problem Description | 2/2 | Problem statement and solution in README |
| KB & Retrieval | 2/2 | Hybrid search; 3-approach comparison in `retrieval_eval.ipynb`; tuning results documented |
| Agents & LLM | 3/3 | 6 tools with full OpenAI schemas, documented above |
| Code Organization | 2/2 | `ai_learning_os/` package; structure table in README |
| Testing | 2/2 | Unit tests + LLM judge in `tests/`; `make test` |
| Evaluation | 3/3 | 15-entry ground truth + LLM judge + tuning via `make eval-tune` |
| Evaluation bonus — hand-crafted ground truth | +2 | `evals/ground_truth.json` written manually, documented above |
| Evaluation bonus — manual evaluation | +2 | Manual review table documented in Evaluation section |
| Monitoring | 2/2 | `logs.jsonl` + Streamlit Monitoring page; access documented |
| Monitoring bonus — user feedback | +1 | 👍/👎 buttons on every response, logged to `logs.jsonl` |
| Monitoring bonus — logs → eval pipeline | +2 | Export button on Monitoring page → `ground_truth_candidates.json` |
| Reproducibility | 2/2 | Complete setup in README + `.env.example` |
| Makefile | +1 | `make run/test/eval/eval-tune/notebook` |
| uv | +1 | `uv sync` dependency management |
| Docker + docker-compose | +2 | `Dockerfile` + `docker-compose.yml`; `docker compose up --build` documented |
| UI | +1 | Streamlit: Chat + Queue + Monitoring pages |
| **Total** | **~28/33** | CI/CD and cloud deployment not implemented |
