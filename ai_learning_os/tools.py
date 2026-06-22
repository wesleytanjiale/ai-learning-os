"""
The 6 agent tools: add_to_queue, browse_queue, load_for_learning,
consolidate_to_kb, search_knowledge_base, get_progress.
"""

import json
import time
import uuid
from pathlib import Path

from openai import OpenAI

from .config import Config
from .ingest import fetch_source
from .retrieval import chunk_text, embed_texts, rebuild_index


# ── JSON store helpers ─────────────────────────────────────────────────────────

def _load(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": [], "time_by_topic": {}, "concepts_mastered": []}


def _save_progress(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Tool 1 — add_to_queue ─────────────────────────────────────────────────────

def add_to_queue(source: str, topic: str, difficulty: str, cfg: Config, client: OpenAI) -> dict:
    """
    Fetch full content from a YouTube URL, article URL, local .md file, or
    local image file, then stage it in the learning queue.
    Does NOT add to the knowledge base — the user must study it first.
    """
    result = fetch_source(source, client, cfg.vision_model)
    if "error" in result:
        return result

    queue = _load(cfg.queue_file)

    # deduplicate by URL / source path
    existing_urls = {item.get("url") for item in queue}
    if source in existing_urls:
        return {"already_queued": True, "message": f"'{result['title']}' is already in your queue."}

    entry = {
        "id": str(uuid.uuid4()),
        "title": result["title"],
        "url": source,
        "source_type": result["source_type"],
        "topic": topic,
        "difficulty": difficulty,
        "full_text": result["text"],
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    queue.append(entry)
    _save(cfg.queue_file, queue)

    word_count = len(result["text"].split())
    return {
        "success": True,
        "title": entry["title"],
        "source_type": entry["source_type"],
        "topic": topic,
        "difficulty": difficulty,
        "word_count": word_count,
        "message": f"Added '{entry['title']}' to your queue ({word_count:,} words). Study it, then use consolidate_to_kb when you're done.",
    }


# ── Tool 2 — browse_queue ─────────────────────────────────────────────────────

def browse_queue(topic: str | None = None, cfg: Config = None) -> dict:  # type: ignore[assignment]
    """List resources staged in the queue (not yet learned)."""
    queue = _load(cfg.queue_file)
    if topic:
        items = [q for q in queue if topic.lower() in q.get("topic", "").lower()]
    else:
        items = queue

    if not items:
        msg = f"No resources in queue" + (f" for topic '{topic}'" if topic else "") + "."
        return {"count": 0, "items": [], "message": msg}

    summary = [
        {
            "title": q["title"],
            "url": q.get("url", ""),
            "topic": q["topic"],
            "difficulty": q["difficulty"],
            "source_type": q["source_type"],
            "added_at": q["added_at"],
            "word_count": len(q.get("full_text", "").split()),
            "preview": q.get("full_text", "")[:400].strip(),
        }
        for q in items
    ]
    return {"count": len(summary), "items": summary}


# ── Tool 3 — load_for_learning ────────────────────────────────────────────────

def load_for_learning(title_or_url: str, cfg: Config) -> dict:
    """
    Load a queued resource's full text into context for a learning session.
    The agent uses this content to walk through key concepts with the user.
    """
    queue = _load(cfg.queue_file)
    match = next(
        (q for q in queue if title_or_url.lower() in q["title"].lower() or title_or_url == q["url"]),
        None,
    )
    if not match:
        return {"error": f"'{title_or_url}' not found in queue. Use browse_queue to see what's available."}

    return {
        "title": match["title"],
        "url": match["url"],
        "source_type": match["source_type"],
        "topic": match["topic"],
        "difficulty": match["difficulty"],
        "full_text": match["full_text"],
        "word_count": len(match["full_text"].split()),
        "instruction": (
            "You now have the full content of this resource. Walk the user through the key concepts, "
            "explain difficult sections, answer their questions. When they say they're done, "
            "call consolidate_to_kb to add it to their knowledge base."
        ),
    }


# ── Tool 4 — consolidate_to_kb ────────────────────────────────────────────────

def consolidate_to_kb(title_or_url: str, cfg: Config, client: OpenAI) -> dict:
    """
    Move a learned resource from the queue into the knowledge base.
    Generates a synthesis + key concepts, chunks the full text, computes
    embeddings, and rebuilds the hybrid search index.
    """
    queue = _load(cfg.queue_file)
    match = next(
        (q for q in queue if title_or_url.lower() in q["title"].lower() or title_or_url == q["url"]),
        None,
    )
    if not match:
        return {"error": f"'{title_or_url}' not found in queue. It may already be in your knowledge base."}

    full_text = match["full_text"]

    # Generate synthesis via LLM
    synthesis_prompt = f"""You are helping consolidate a learning resource into a personal knowledge base.

Resource: {match['title']}
Topic: {match['topic']}

Content (first 6000 words):
{" ".join(full_text.split()[:6000])}

Write a concise synthesis (150-200 words) covering:
1. The main concepts and ideas
2. Key takeaways a learner should remember
3. How this connects to related topics in AI/ML

Then list 5-10 key concepts as a JSON array.

Respond in this exact format:
SYNTHESIS:
<synthesis text>

KEY_CONCEPTS:
["concept1", "concept2", ...]"""

    response = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content

    synthesis = ""
    key_concepts: list[str] = []
    try:
        if "SYNTHESIS:" in raw and "KEY_CONCEPTS:" in raw:
            parts = raw.split("KEY_CONCEPTS:")
            synthesis = parts[0].replace("SYNTHESIS:", "").strip()
            import re
            arr_match = re.search(r"\[.*?\]", parts[1], re.DOTALL)
            if arr_match:
                key_concepts = json.loads(arr_match.group())
    except Exception:
        synthesis = raw[:500]

    # Chunk full text
    chunks_text = chunk_text(full_text, cfg.chunk_size_words, cfg.chunk_overlap_words)
    chunk_dicts_no_embed = [
        {
            "chunk_id": f"{match['id']}-{i}",
            "text": text,
            "source_title": match["title"],
            "source_url": match["url"],
            "topic": match["topic"],
        }
        for i, text in enumerate(chunks_text)
    ]

    # Embed chunks
    embeddings = embed_texts([c["text"] for c in chunk_dicts_no_embed], client, cfg.embedding_model)
    chunk_dicts = [
        {**c, "embedding": emb}
        for c, emb in zip(chunk_dicts_no_embed, embeddings)
    ]

    # Build KB entry
    kb = _load(cfg.kb_file)
    kb_entry = {
        "id": match["id"],
        "title": match["title"],
        "url": match["url"],
        "source_type": match["source_type"],
        "topic": match["topic"],
        "difficulty": match["difficulty"],
        "synthesis": synthesis,
        "key_concepts": key_concepts,
        "learned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunks": chunk_dicts,
    }
    kb.append(kb_entry)
    _save(cfg.kb_file, kb)

    # Remove from queue
    queue = [q for q in queue if q["id"] != match["id"]]
    _save(cfg.queue_file, queue)

    # Update progress
    progress = _load_progress(cfg.progress_file)
    if match["title"] not in progress["completed"]:
        progress["completed"].append(match["title"])
    for concept in key_concepts:
        if concept not in progress["concepts_mastered"]:
            progress["concepts_mastered"].append(concept)
    _save_progress(cfg.progress_file, progress)

    # Rebuild index
    rebuild_index(kb, cfg.rrf_k)

    return {
        "success": True,
        "title": match["title"],
        "chunks_created": len(chunk_dicts),
        "key_concepts": key_concepts,
        "synthesis": synthesis,
        "message": f"'{match['title']}' is now in your knowledge base ({len(chunk_dicts)} chunks indexed).",
    }


# ── Tool 5 — search_knowledge_base ───────────────────────────────────────────

def search_knowledge_base(query: str, cfg: Config, client: OpenAI, num_results: int | None = None) -> dict:
    """
    Hybrid search (BM25 + semantic embeddings + RRF) over the personal knowledge base.
    Returns relevant passages with explicit source citations.
    Only searches resources you have already studied and consolidated.
    """
    from .retrieval import get_index

    kb = _load(cfg.kb_file)
    if not kb:
        return {
            "results": [],
            "message": "Your knowledge base is empty. Study some resources and consolidate them first.",
        }

    index = get_index()
    if not index.chunks:
        rebuild_index(kb, cfg.rrf_k)

    n = num_results or cfg.search_num_results
    query_embedding = embed_texts([query], client, cfg.embedding_model)[0] if index.embeddings is not None else None
    results = index.search(query, query_embedding=query_embedding, num_results=n)

    if not results:
        return {"results": [], "message": f"No relevant content found for '{query}' in your knowledge base."}

    return {
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "source_title": r["source_title"],
                "source_url": r["source_url"],
                "score": r["score"],
            }
            for r in results
        ],
        "instruction": "Synthesize these passages to answer the user's question. Cite each source as: (Source: <source_title>)",
    }


# ── Tool 6 — get_progress ─────────────────────────────────────────────────────

def get_progress(topic: str | None = None, cfg: Config = None) -> dict:  # type: ignore[assignment]
    """Get learning progress: queue depth, KB size, time by topic, concepts mastered."""
    progress = _load_progress(cfg.progress_file)
    queue = _load(cfg.queue_file)
    kb = _load(cfg.kb_file)

    if topic:
        queue_items = [q["title"] for q in queue if topic.lower() in q.get("topic", "").lower()]
        kb_items = [k["title"] for k in kb if topic.lower() in k.get("topic", "").lower()]
    else:
        queue_items = [q["title"] for q in queue]
        kb_items = [k["title"] for k in kb]

    topics_in_kb = sorted({k["topic"] for k in kb})

    return {
        "queue_depth": len(queue_items),
        "queue_titles": queue_items,
        "kb_size": len(kb_items),
        "kb_titles": kb_items,
        "topics_covered": topics_in_kb,
        "completed_count": len(progress.get("completed", [])),
        "time_by_topic": progress.get("time_by_topic", {}),
        "concepts_mastered": progress.get("concepts_mastered", []),
        "concepts_count": len(progress.get("concepts_mastered", [])),
    }


# ── Tool definitions (OpenAI schema) ──────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "add_to_queue",
            "description": (
                "Fetch full content from a YouTube URL, article URL, local .md file, or local image file "
                "and stage it in the learning queue. Does NOT add to the knowledge base — "
                "the user must study it first, then call consolidate_to_kb."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "YouTube URL, article URL, local .md path, or image path."},
                    "topic": {"type": "string", "description": "Topic category, e.g. LLMs, RAG, Agents, Python, MLOps."},
                    "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                },
                "required": ["source", "topic", "difficulty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_queue",
            "description": "List resources staged in the queue that have not yet been studied and added to the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional: filter by topic."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_for_learning",
            "description": (
                "Load a queued resource's full text so you can walk through its key concepts with the user. "
                "Call this when the user wants to study a specific resource from their queue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_or_url": {"type": "string", "description": "Title (partial match) or exact URL of the queued resource."},
                },
                "required": ["title_or_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consolidate_to_kb",
            "description": (
                "Move a resource from the queue into the personal knowledge base after the user has studied it. "
                "Generates a synthesis, chunks the full text, computes embeddings, and indexes everything. "
                "Call this when the user says they are done with a resource."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_or_url": {"type": "string", "description": "Title (partial match) or exact URL of the queued resource to consolidate."},
                },
                "required": ["title_or_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Hybrid search (BM25 + semantic) over the personal knowledge base. "
                "Returns relevant passages with source citations. "
                "Only searches resources the user has already studied and consolidated. "
                "Always call this before answering factual questions or generating study plans."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "num_results": {"type": "integer", "description": "Number of chunks to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_progress",
            "description": (
                "Get learning progress: queue depth, knowledge base size, topics covered, "
                "time spent by topic, and concepts mastered. "
                "Call this at the start of any study plan or roadmap request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional: filter stats by topic."},
                },
            },
        },
    },
]
