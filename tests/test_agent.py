"""
Agent scenario tests.

Run from repo root:
    uv run pytest tests/ -v -s

Scenarios:
  1. Study plan calls get_progress before search_knowledge_base
  2. KB search returns citations (source_title present in answer)
  3. Empty KB — clear message, no hallucination
  4. Queue browsing — browse_queue called, correct titles returned
  5. LLM judge — output quality on study plan query
"""

import sys
from unittest.mock import patch

import pytest

# ai_learning_os.__init__ exports `agent` (the function), shadowing the submodule
# in the package namespace. Access the actual module via sys.modules.
import ai_learning_os.agent  # noqa: F401 — ensures module is registered
import ai_learning_os.tools  # noqa: F401
_agent_mod = sys.modules["ai_learning_os.agent"]
_tools_mod = sys.modules["ai_learning_os.tools"]

from ai_learning_os.agent import agent
from tests.judge import assert_criteria


# ── Scenario 1: Tool call order — get_progress before search ──────────────────

def test_study_plan_calls_get_progress_then_search(cfg_with_kb, openai_client):
    """
    For any study plan query, the agent must call get_progress before
    search_knowledge_base to know what the user has already completed.
    """
    call_order = []
    _orig_get_progress = _tools_mod.get_progress
    _orig_search = _tools_mod.search_knowledge_base

    def tracking_get_progress(*args, **kwargs):
        call_order.append("get_progress")
        return _orig_get_progress(*args, **kwargs)

    def tracking_search(*args, **kwargs):
        call_order.append("search_knowledge_base")
        return _orig_search(*args, **kwargs)

    with (
        patch.object(_agent_mod, "get_progress", side_effect=tracking_get_progress),
        patch.object(_agent_mod, "search_knowledge_base", side_effect=tracking_search),
    ):
        result = agent("I have 3 hours this weekend. What should I focus on?",
                       cfg=cfg_with_kb, client=openai_client)

    print(f"\ncall_order: {call_order}")
    assert "get_progress" in call_order, "get_progress was never called"
    assert "search_knowledge_base" in call_order, "search_knowledge_base was never called"
    assert call_order.index("get_progress") < call_order.index("search_knowledge_base"), (
        "get_progress must be called before search_knowledge_base"
    )


# ── Scenario 2: KB search includes source citation ────────────────────────────

def test_kb_search_answer_cites_source(cfg_with_kb, openai_client):
    """
    When the agent answers from the knowledge base, it must cite the source title
    in the answer. Generic uncited answers are a failure mode.
    """
    result = agent("What do I know about backpropagation?",
                   cfg=cfg_with_kb, client=openai_client)

    print(f"\nanswer: {result.answer}")
    assert "search_knowledge_base" in [tc["name"] for tc in result.tool_calls], (
        "search_knowledge_base was never called"
    )
    assert "Neural Networks: Zero to Hero" in result.answer, (
        "Source title not cited in answer — agent must cite sources"
    )


# ── Scenario 3: Empty KB — no hallucination ───────────────────────────────────

def test_empty_kb_no_hallucination(tmp_cfg, openai_client):
    """
    When the KB is empty, the agent must not hallucinate resources.
    It should clearly state that the knowledge base is empty.
    """
    tmp_cfg.kb_file.write_text("[]")
    tmp_cfg.queue_file.write_text("[]")
    tmp_cfg.progress_file.write_text('{"completed":[],"time_by_topic":{},"concepts_mastered":[]}')

    result = agent("What do I know about transformers?",
                   cfg=tmp_cfg, client=openai_client)

    print(f"\nanswer: {result.answer}")
    lower = result.answer.lower()
    assert any(phrase in lower for phrase in [
        "knowledge base is empty",
        "nothing in your knowledge base",
        "no content",
        "haven't studied",
        "not in your knowledge base",
        "don't have any",
        "no resources",
        "haven't added",
        "empty",
        "study some",
    ]), "Agent did not clearly indicate the KB is empty"


# ── Scenario 4: Queue browsing ────────────────────────────────────────────────

def test_browse_queue_called_and_returns_title(cfg_with_queue, openai_client):
    """
    When the user asks what's in their queue, the agent must call browse_queue
    and include the actual queued resource title in the answer.
    """
    result = agent("What do I have in my queue to study?",
                   cfg=cfg_with_queue, client=openai_client)

    print(f"\nanswer: {result.answer}")
    assert "browse_queue" in [tc["name"] for tc in result.tool_calls], (
        "browse_queue was never called"
    )
    assert "RAG From Scratch" in result.answer, (
        "Queued resource title not mentioned in answer"
    )


# ── Scenario 5: LLM judge — study plan output quality ────────────────────────

def test_study_plan_output_quality(cfg_with_kb, openai_client):
    """
    LLM judge evaluates whether the study plan meets quality criteria that
    are hard to assert with simple string checks.
    """
    user_prompt = "I have 3 hours this weekend. What should I study to improve my LLM understanding?"
    result = agent(user_prompt, cfg=cfg_with_kb, client=openai_client)

    print(f"\nanswer: {result.answer}")

    assert_criteria(user_prompt, result.answer, result.tool_calls, [
        "the agent calls search_knowledge_base before making recommendations",
        "the answer does not invent specific titled resources (e.g. 'The Attention Is All You Need Guide') that were not returned by search_knowledge_base or browse_queue",
        "the answer cites at least one source explicitly (e.g. 'Source: ...' or mentions a title)",
        "time allocations in the plan sum to approximately 3 hours (2-4 hours acceptable)",
    ])
