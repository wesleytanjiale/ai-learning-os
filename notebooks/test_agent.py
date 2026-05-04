"""Smoke test for the agent — run from the notebooks/ directory with: uv run python test_agent.py"""

import json
import re
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from minsearch import Index
from youtube_transcript_api import YouTubeTranscriptApi

openai_client = OpenAI()

# ── Setup ──────────────────────────────────────────────────────────────────────

with open("../data/resources.json") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} resources\n")

index = Index(
    text_fields=["title", "description", "topic"],
    keyword_fields=["type", "difficulty"],
)
index.fit(documents)

# ── Progress helpers ───────────────────────────────────────────────────────────

PROGRESS_FILE = Path("../data/progress.json")

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "time_by_topic": {}, "concepts_mastered": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# ── Tools ──────────────────────────────────────────────────────────────────────

def _extract_video_id(url):
    for pattern in [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"shorts/([0-9A-Za-z_-]{11})",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

def _youtube_title(url):
    try:
        req = urllib.request.Request(f"https://www.youtube.com/oembed?url={url}&format=json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("title", url)
    except Exception:
        return url

def fetch_content(source):
    if "youtube.com" in source or "youtu.be" in source:
        video_id = _extract_video_id(source)
        if not video_id:
            return {"error": f"Could not extract video ID from: {source}"}
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        text = " ".join(snippet.text for snippet in transcript)
        return {
            "title": _youtube_title(source),
            "text": text,
            "url": source,
            "content_type": "video",
            "estimated_read_minutes": len(text.split()) // 130,
        }
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else source
        html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        return {
            "title": title,
            "text": text[:8000],
            "url": source,
            "content_type": "article",
            "estimated_read_minutes": len(text.split()) // 200,
        }
    p = Path(source)
    if not p.exists():
        return {"error": f"File not found: {source}"}
    if p.suffix.lower() == ".pdf":
        return {"error": "PDF support requires pypdf. Install with: uv add pypdf"}
    text = p.read_text(encoding="utf-8")
    return {
        "title": p.stem,
        "text": text,
        "url": str(p.absolute()),
        "content_type": "file",
        "estimated_read_minutes": len(text.split()) // 200,
    }

def search_knowledge_base(query, exclude_completed=False, difficulty=None, type=None, num_results=5):
    keyword_filters = {}
    if difficulty:
        keyword_filters["difficulty"] = difficulty
    if type:
        keyword_filters["type"] = type
    fetch_n = num_results * 3 if exclude_completed else num_results
    results = index.search(query, filter_dict=keyword_filters, num_results=fetch_n)
    if exclude_completed:
        completed = set(load_progress()["completed"])
        results = [r for r in results if r["title"] not in completed][:num_results]
    return results

def add_to_knowledge_base(title, text, url, topic, resource_type, difficulty):
    global documents
    new_doc = {
        "title": title,
        "description": text[:500],
        "url": url,
        "topic": topic,
        "type": resource_type,
        "difficulty": difficulty,
    }
    documents.append(new_doc)
    with open("../data/resources.json", "w") as f:
        json.dump(documents, f, indent=2)
    index.fit(documents)
    return {"success": True, "message": f"Added '{title}' ({len(documents)} total resources)"}

def update_progress(resource_id, event, metadata=None):
    metadata = metadata or {}
    progress = load_progress()
    if event == "completed":
        if resource_id not in progress["completed"]:
            progress["completed"].append(resource_id)
    elif event == "time_logged":
        topic = metadata.get("topic", "general")
        progress["time_by_topic"][topic] = progress["time_by_topic"].get(topic, 0) + metadata.get("minutes", 0)
    elif event == "concept_mastered":
        concept = metadata.get("concept", "")
        if concept and concept not in progress["concepts_mastered"]:
            progress["concepts_mastered"].append(concept)
    save_progress(progress)
    return {"success": True, "event": event, "resource_id": resource_id}

def get_progress(topic=None):
    progress = load_progress()
    completed = set(progress["completed"])
    not_started = [
        d["title"] for d in documents
        if d["title"] not in completed
        and (topic is None or topic.lower() in d.get("topic", "").lower())
    ]
    return {
        "completed": list(completed),
        "time_by_topic": progress["time_by_topic"],
        "concepts_mastered": progress["concepts_mastered"],
        "not_started": not_started,
    }

# ── Tool definitions ───────────────────────────────────────────────────────────

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "fetch_content",
            "description": "Fetch the full text of a web article, YouTube video transcript, or local file. Call this when the user provides a URL or file path to add or summarize.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "A URL (web article or YouTube) or an absolute local file path."}
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the personal knowledge base for relevant learning resources. Always call this before creating a study plan or making any recommendation. Pass exclude_completed=true to skip resources the user has already finished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "exclude_completed": {"type": "boolean", "description": "If true, skip resources the user has already finished."},
                    "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                    "type": {"type": "string", "enum": ["article", "video", "course", "tutorial", "file"]},
                    "num_results": {"type": "integer", "description": "Number of results to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_knowledge_base",
            "description": "Save a fetched resource to the personal knowledge base. Always call fetch_content first; only call this if the user confirms they want to save it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "text": {"type": "string", "description": "Full text content returned by fetch_content."},
                    "url": {"type": "string"},
                    "topic": {"type": "string", "description": "Topic category, e.g. LLMs, RAG, MLOps, Python."},
                    "type": {"type": "string", "enum": ["article", "video", "course", "tutorial", "file"]},
                    "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
                },
                "required": ["title", "text", "url", "topic", "type", "difficulty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_progress",
            "description": "Record a progress event: the user completed a resource, logged study time, or mastered a concept.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string", "description": "Title or URL of the resource."},
                    "event": {"type": "string", "enum": ["completed", "time_logged", "concept_mastered"]},
                    "metadata": {
                        "type": "object",
                        "description": "For time_logged: {topic, minutes}. For concept_mastered: {concept}.",
                        "properties": {
                            "topic": {"type": "string"},
                            "minutes": {"type": "integer"},
                            "concept": {"type": "string"},
                        },
                    },
                },
                "required": ["resource_id", "event"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_progress",
            "description": "Get the user's current learning progress: completed resources, time spent by topic, concepts mastered, and what has not been started yet. Always call this before generating a study plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional: filter the not_started list by topic."}
                },
            },
        },
    },
]

# ── Dispatcher ─────────────────────────────────────────────────────────────────

def call_tool(name, args):
    if name == "fetch_content":
        return fetch_content(**args)
    elif name == "search_knowledge_base":
        args.pop("filters", None)  # defensive: ignore if model nests filters anyway
        return search_knowledge_base(**args)
    elif name == "add_to_knowledge_base":
        args["resource_type"] = args.pop("type")
        return add_to_knowledge_base(**args)
    elif name == "update_progress":
        return update_progress(**args)
    elif name == "get_progress":
        return get_progress(**args)
    else:
        return {"error": f"Unknown tool: {name}"}

# ── Instructions ───────────────────────────────────────────────────────────────

agent_instructions = """
You are an AI Learning Chief of Staff. Your job is to help the user build structured, focused skills using their personal saved resources.

TOOL CALL RULES:
- search_knowledge_base: Call before ANY study plan, recommendation, or "what's next" answer. Always pass exclude_completed=true unless the user explicitly asks about already-completed work.
- get_progress: Call at the start of every study plan or roadmap request. It tells you what is done, what is not, and where study time has gone.
- fetch_content: Call when the user provides a URL or file path, whether they want to add it or just summarize it.
- add_to_knowledge_base: Call only after a successful fetch_content call, and only if the user confirms they want to save the resource.
- update_progress: Call when the user reports finishing a resource, logging study time, or naming a concept they have learned.

AFTER RECOMMENDING A RESOURCE:
When the user accepts your suggestion to study something specific, confirm the plan and end your reply with:
"Let me know when you're done and I'll update your progress."

HOW TO ANSWER:
- Study plans: Name each resource. Explain why it is relevant right now. Suggest a realistic order and time allocation based on the user's available hours.
- "What's next": Call get_progress first to avoid recommending completed resources, then search for what naturally follows.
- Summaries: Extract key concepts, main arguments, and concrete takeaways. Be specific, not generic.
- Practical projects: Reason from what the user has completed or concepts they have mastered to project ideas they can start today.

Stay grounded in the user's personal knowledge base. Do not recommend resources that are not saved there unless the user explicitly asks for outside suggestions.
""".strip()

# ── Agent loop ─────────────────────────────────────────────────────────────────

def agent(user_question, model="gpt-4o-mini", max_iterations=10):
    messages = [
        {"role": "system", "content": agent_instructions},
        {"role": "user", "content": user_question},
    ]
    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            print("--- Final answer ---")
            return msg.content
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"[TOOL] {name}({json.dumps(args)})")
            result = call_tool(name, args)
            print(f"[RESULT] {str(result)[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return "Reached maximum iterations without a final answer."

# ── Run ────────────────────────────────────────────────────────────────────────

QUERIES = [
    (
        "QUERY 1 — study plan (should call get_progress + search_knowledge_base)",
        "I have 5 hours this weekend. What should I focus on to get closer to being job-ready in AI engineering?",
    ),
    (
        "QUERY 2 — summarize a URL (should call fetch_content)",
        "Can you summarize this article for me? https://simonwillison.net/2023/Apr/25/dual-llm-pattern",
    ),
    (
        "QUERY 3 — edge case: topic not in KB (should search, find nothing relevant, say so)",
        "I want to learn quantum computing. What do I have in my knowledge base about it?",
    ),
]

for label, query in QUERIES:
    print(f"\n{'='*70}")
    print(label)
    print(f"Q: {query}")
    print("=" * 70)
    answer = agent(query)
    print(f"\n>>> ANSWER:\n{answer}\n")
