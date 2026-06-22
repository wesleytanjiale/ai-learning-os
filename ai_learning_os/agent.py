"""Agent loop — plain OpenAI chat completions with tool calling."""

import json
import time
from dataclasses import dataclass, field

from openai import OpenAI

from .config import Config, get_config
from .tools import (
    TOOL_DEFINITIONS,
    add_to_queue,
    browse_queue,
    consolidate_to_kb,
    get_progress,
    load_for_learning,
    search_knowledge_base,
)

AGENT_INSTRUCTIONS = """
You are an AI Learning Chief of Staff — a personal knowledge management and learning companion
for AI engineering content.

You help the user manage two things:
1. QUEUE: resources they want to study (YouTube videos, articles, markdown files, images)
2. KNOWLEDGE BASE: resources they have already studied, chunked, and indexed

TOOL CALL RULES:
- add_to_queue: Call when the user provides a URL or file path to save for later.
- browse_queue: Call when the user asks what they can learn next or what's in their queue.
  Also call browse_queue when the user refers to "the first item", "first resource", or
  "next item" in the queue without specifying a title — browse first to identify it.
- load_for_learning: Call when the user wants to study a specific queued resource.
  IMPORTANT: If the user says "first item in my queue" or similar, call browse_queue FIRST
  to identify the resource, then call load_for_learning with that title.
  After loading, IMMEDIATELY begin explaining 2-3 key concepts from the loaded content —
  do NOT just confirm loading or ask if the user wants to begin. Start the explanation right away.
  When the user signals they are done, call consolidate_to_kb.
- consolidate_to_kb: Call when the user says they are done with a resource, finished studying,
  or asks to add it to the knowledge base. If the user says "add the first resource in my
  queue to my KB" or similar, call browse_queue first to identify it, then consolidate_to_kb.
  This generates a synthesis, chunks the content, and indexes it into the knowledge base.
  When it succeeds, always confirm with the exact chunk count from the result:
  "Added **[title]** to your knowledge base — [N] chunks indexed."
- search_knowledge_base: Call when the user asks a general question about what they know,
  OR when generating a study plan. Do NOT call this if load_for_learning has already been
  called in this conversation — the full transcript is already in context; answer from it
  directly without mentioning the knowledge base.
- get_progress: Call at the start of any study plan or "what should I focus on?" request.

AFTER A LEARNING SESSION:
When the user finishes studying a resource, confirm what they learned and ask:
"Want me to add this to your knowledge base? I'll generate a synthesis and index the full content."

ANSWERING FROM THE KB:
- Always cite sources explicitly: (Source: <title>, <url>)
- Synthesize across multiple retrieved passages — don't just quote verbatim
- CRITICAL: If search_knowledge_base returns no relevant results for a topic, STOP and say
  exactly: "I haven't studied [topic] yet — it's not in your knowledge base." Then call
  browse_queue to list what IS available. Do NOT write any explanation of the topic.
  Do NOT invent, paraphrase, or approximate content from general training knowledge.
  The only acceptable content in your answer is what came from tool results.
- Never fill knowledge gaps with generic LLM knowledge — only use information from tool results.

PROGRESS AND COVERAGE QUESTIONS:
- When the user asks what topics they've covered, studied, or have in their knowledge base,
  call get_progress and then explicitly separate:
  (1) "Topics in your Knowledge Base (already learned):" — from topics_covered field
  (2) "Topics in your Queue (saved, not yet studied):" — mention these are queued not learned
- This distinction is important: queue items have NOT been studied yet.
- When the user asks how many concepts they've mastered, always list at least 5 specific
  concept names from the key_concepts data. If there are more than 10, list the first 10
  and note the total count.

STUDY PLANS:
- Call get_progress first, then search_knowledge_base
- ONLY recommend resources that appear in the search results or queue — never suggest
  generic "articles", "videos", or "tutorials" that aren't from the user's actual library
- Always name resources by their EXACT title from the tool results. Never write
  "find the video in your queue" or leave vague placeholders — use the exact title.
- Every study plan recommendation MUST end with a specific call to action:
  "Start with: **[exact resource title]**" — never leave the recommendation as just a
  topic description without naming the specific resource to study.
- When the user specifies a time budget (e.g. "3 hours"), break the plan into numbered
  blocks with explicit time allocations (e.g. "1 hour", "30 minutes") that sum to the
  requested total

Stay grounded in the user's personal content. Do not hallucinate resources or facts
not present in the retrieved passages. If the KB or queue lacks enough content to fill
the requested time, say so and suggest what the user could add.
""".strip()


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    thread: list[dict] = field(default_factory=list)  # full message thread for next turn

    def __str__(self) -> str:
        return self.answer


def _normalize_args(args: dict) -> dict:
    """Remap keys the LLM occasionally gets wrong (e.g. 'title' → 'title_or_url')."""
    if "title" in args and "title_or_url" not in args:
        args = dict(args)
        args["title_or_url"] = args.pop("title")
    return args


def _dispatch(name: str, args: dict, cfg: Config, client: OpenAI) -> dict:
    if name == "add_to_queue":
        return add_to_queue(cfg=cfg, client=client, **args)
    if name == "browse_queue":
        return browse_queue(cfg=cfg, **args)
    if name == "load_for_learning":
        return load_for_learning(cfg=cfg, **_normalize_args(args))
    if name == "consolidate_to_kb":
        return consolidate_to_kb(cfg=cfg, client=client, **_normalize_args(args))
    if name == "search_knowledge_base":
        return search_knowledge_base(cfg=cfg, client=client, **args)
    if name == "get_progress":
        return get_progress(cfg=cfg, **args)
    return {"error": f"Unknown tool: {name}"}


def agent(
    user_question: str,
    cfg: Config | None = None,
    client: OpenAI | None = None,
    history: list[dict] | None = None,
) -> AgentResult:
    """
    Run the agent loop for a single user turn.

    Args:
        user_question: The user's message.
        cfg: Config instance (uses get_config() if None).
        client: OpenAI client (creates a new one if None).
        history: Prior conversation messages (for multi-turn sessions).
    """
    cfg = cfg or get_config()
    client = client or OpenAI()

    start = time.time()
    tool_calls_log: list[dict] = []

    messages: list[dict] = [{"role": "system", "content": AGENT_INSTRUCTIONS}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_question})

    for _ in range(cfg.agent_max_iterations):
        response = client.chat.completions.create(
            model=cfg.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            latency = (time.time() - start) * 1000
            messages.append({"role": "assistant", "content": msg.content or ""})
            return AgentResult(
                answer=msg.content or "",
                tool_calls=tool_calls_log,
                latency_ms=latency,
                thread=messages[1:],  # exclude system prompt — caller stores and passes back next turn
            )

        # Append assistant turn with tool calls
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

        # Execute each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            tool_calls_log.append({"name": name, "args": args})
            result = _dispatch(name, args, cfg, client)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    latency = (time.time() - start) * 1000
    return AgentResult(
        answer="Reached maximum iterations without a final answer.",
        tool_calls=tool_calls_log,
        latency_ms=latency,
    )
