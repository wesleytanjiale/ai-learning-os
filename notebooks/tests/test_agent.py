"""
Agent scenario tests.

Run from the notebooks/ directory:
    uv run pytest tests/ -v -s

Each test maps to a scenario defined during brainstorming:
  - Scenario 1: study plan calls get_progress before search_knowledge_base
  - Scenario 2: completed resources are not recommended
  - Scenario 3: "I finished X" triggers update_progress (exposes title-matching bug)
"""

from unittest.mock import patch
import test_agent as agent_module
from tests.judge import assert_criteria


# ── Scenario 1: Tool call order ────────────────────────────────────────────────
def test_study_plan_calls_get_progress_then_search(agent, fresh_progress):
    """
    For any study plan query, the agent must call get_progress before
    search_knowledge_base so it knows what the user has already completed.
    Calling search first risks recommending already-finished resources.
    """
    call_order = []

    # Capture originals BEFORE patch replaces them in the module namespace.
    # Calling agent_module.get_progress inside the tracker would hit the mock
    # and recurse infinitely — calling the saved original avoids that.
    _orig_get_progress = agent_module.get_progress
    _orig_search = agent_module.search_knowledge_base

    def tracking_get_progress(*args, **kwargs):
        call_order.append("get_progress")
        return _orig_get_progress(*args, **kwargs)

    def tracking_search(query, **kwargs):
        call_order.append("search_knowledge_base")
        return _orig_search(query, **kwargs)

    with (
        patch.object(agent_module, "get_progress", side_effect=tracking_get_progress),
        patch.object(agent_module, "search_knowledge_base", side_effect=tracking_search),
    ):
        result = agent("I have 5 hours this weekend. What should I focus on?")

    print(f"\ncall_order: {call_order}")
    print(f"\nanswer: {result.answer}")

    assert "get_progress" in call_order, "get_progress was never called"
    assert "search_knowledge_base" in call_order, "search_knowledge_base was never called"
    assert call_order.index("get_progress") < call_order.index("search_knowledge_base"), (
        "get_progress must be called before search_knowledge_base"
    )


# ── Scenario 2: Output quality — completed resources excluded ─────────────────
def test_completed_resources_not_recommended(agent, progress_with_karpathy_done):
    """
    If the user has already completed 'Neural Networks: Zero to Hero',
    the agent must not recommend it in a study plan.
    This is the worst-case trust failure: recommending something already done.
    """
    result = agent("I have 5 hours this weekend. What should I focus on?")

    print(f"\nanswer: {result.answer}")

    assert "Neural Networks: Zero to Hero" not in result.answer, (
        "Agent recommended a resource the user has already completed"
    )


# ── Scenario 3: Progress tracking — title matching ────────────────────────────
def test_finished_resource_triggers_update_progress(agent, fresh_progress):
    """
    When the user says 'I finished the Karpathy video series', the agent should
    call update_progress with the correct KB title: 'Neural Networks: Zero to Hero'.

    This test is EXPECTED TO SURFACE A BUG: the user says 'Karpathy video series'
    but the KB title is 'Neural Networks: Zero to Hero'. The agent may pass the wrong
    string to update_progress, leaving the resource un-marked in progress.json.
    """
    recorded_calls = []

    def tracking_update_progress(resource_id, event, metadata=None):
        recorded_calls.append({"resource_id": resource_id, "event": event})
        return agent_module.update_progress(resource_id, event, metadata)

    with patch.object(agent_module, "update_progress", side_effect=tracking_update_progress):
        result = agent("I just finished the Karpathy video series. What should I study next?")

    print(f"\nrecorded update_progress calls: {recorded_calls}")
    print(f"\nanswer: {result.answer}")

    assert len(recorded_calls) > 0, (
        "update_progress was never called — agent ignored the completion report"
    )

    resource_ids = [c["resource_id"] for c in recorded_calls]
    assert "Neural Networks: Zero to Hero" in resource_ids, (
        f"update_progress was called but with wrong resource_id: {resource_ids}. "
        "The agent used the user's phrasing instead of the KB title."
    )


# ── LLM Judge: output quality ─────────────────────────────────────────────────
def test_study_plan_output_quality(agent, progress_with_karpathy_done):
    """
    LLM judge evaluates whether the study plan answer meets quality criteria
    that are hard to assert with simple string checks.

    Criteria are specific and concrete — each one checks exact behaviour,
    not vague properties like 'is helpful'.
    """
    user_prompt = "I have 5 hours this weekend. What should I focus on?"
    result = agent(user_prompt)

    print(f"\nanswer: {result.answer}")
    print(f"\ntool_calls: {result.tool_calls}")

    assert_criteria(user_prompt, result, [
        "every resource title mentioned in the study plan exists verbatim in the "
        "knowledge base — the agent must not invent or paraphrase resource names",

        "the time allocations listed in the study plan sum to approximately 5 hours "
        "(between 4 and 6 hours total) — the agent must respect the user's time constraint",

        "the resource 'Neural Networks: Zero to Hero' does not appear anywhere in the "
        "response, since the user has already completed it",
    ])
