"""Reusable judge helper for test assertions — wraps evals/judge.py."""

from evals.judge import evaluate, JudgeVerdict


def assert_criteria(user_prompt: str, answer: str, tool_calls: list[dict], criteria: list[str]) -> JudgeVerdict:
    verdict = evaluate(user_prompt, answer, tool_calls, criteria)

    print(f"\njudge summary: {verdict.summary}")
    for r in verdict.criteria_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.criterion}")
        print(f"         {r.evidence}")

    for r in verdict.criteria_results:
        assert r.passed, f"FAILED: {r.criterion}\nEvidence: {r.evidence}"

    return verdict
