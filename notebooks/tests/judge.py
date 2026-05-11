"""
LLM judge — reusable evaluation logic.
Mirrors the course's judge.py but uses plain OpenAI (no pydantic_ai).
"""

import json
from pydantic import BaseModel, Field
from openai import OpenAI

openai_client = OpenAI()

judge_instructions = """
You are an expert judge evaluating the performance of an AI learning assistant agent.
Evaluate each criterion carefully based on the agent's output and tool calls provided.
Be specific in your judgements — reference exact content from the output as evidence.
""".strip()

judge_prompt_template = """
Evaluate the agent's performance based on the following criteria:
<CRITERIA>
{criteria}
</CRITERIA>

The user asked:
<USER_PROMPT>
{user_prompt}
</USER_PROMPT>

The agent's final answer:
<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

Tool calls made during this run (in order):
<TOOL_CALLS>
{tool_calls}
</TOOL_CALLS>
""".strip()


class JudgeCriterion(BaseModel):
    criterion_description: str = Field(
        description="The specific requirement being evaluated."
    )
    passed: bool = Field(
        description="Whether the agent's response satisfied this requirement."
    )
    judgement: str = Field(
        description="Why it passed or failed, with specific evidence from the output."
    )


class JudgeFeedback(BaseModel):
    criteria: list[JudgeCriterion] = Field(
        description="Individual evaluation for each criterion."
    )
    feedback: str = Field(
        description="Overall summary of the agent's performance."
    )


def evaluate(user_prompt, result, criteria) -> JudgeFeedback:
    tool_calls_str = "\n".join(
        f"{tc['name']}({json.dumps(tc['args'])})" for tc in result.tool_calls
    ) or "(no tool calls)"

    prompt = judge_prompt_template.format(
        criteria="\n".join(f"- {c}" for c in criteria),
        user_prompt=user_prompt,
        output=result.answer,
        tool_calls=tool_calls_str,
    )

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": judge_instructions},
            {"role": "user", "content": prompt},
        ],
        response_format=JudgeFeedback,
    )

    return response.choices[0].message.parsed


def assert_criteria(user_prompt, result, criteria):
    feedback = evaluate(user_prompt, result, criteria)

    print(f"\njudge feedback: {feedback.feedback}")
    for criterion in feedback.criteria:
        status = "PASS" if criterion.passed else "FAIL"
        print(f"  [{status}] {criterion.criterion_description}")
        print(f"         {criterion.judgement}")

    for criterion in feedback.criteria:
        assert criterion.passed, (
            f"{criterion.criterion_description}: {criterion.judgement}"
        )
