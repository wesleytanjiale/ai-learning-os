"""LLM judge for evaluating agent responses against criteria."""

import json
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

JUDGE_INSTRUCTIONS = """
You are an expert judge evaluating an AI Learning OS agent — a personal knowledge management
and learning companion. You assess whether the agent correctly uses its tools and produces
high-quality, grounded responses.

IMPORTANT RULES:
- Evaluate ONLY against the exact criteria listed. Do not add, invent, or substitute your own criteria.
- If a criterion is conditional (e.g. "if content exists, cite sources; if not, state it's missing"),
  mark it PASSED if the agent correctly handled whichever branch applied.
- "Suggests" or "recommends" criteria: the agent passes if it makes any reasonable suggestion, even if not phrased exactly as expected.
- Tool call criteria: check the TOOL_CALLS list, not the answer text.
- Be specific — reference exact content from the output as evidence.
""".strip()

JUDGE_PROMPT = """
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

Tool calls made (in order):
<TOOL_CALLS>
{tool_calls}
</TOOL_CALLS>
""".strip()


class CriterionResult(BaseModel):
    criterion: str = Field(description="The criterion being evaluated.")
    passed: bool = Field(description="Whether the agent satisfied this criterion.")
    evidence: str = Field(description="Specific evidence from the output supporting the verdict.")


class JudgeVerdict(BaseModel):
    criteria_results: list[CriterionResult]
    overall_passed: bool = Field(description="True if all criteria passed.")
    summary: str = Field(description="One-sentence overall assessment.")


def evaluate(user_prompt: str, answer: str, tool_calls: list[dict], criteria: list[str]) -> JudgeVerdict:
    tool_calls_str = "\n".join(
        f"{tc['name']}({json.dumps(tc['args'])})" for tc in tool_calls
    ) or "(no tool calls)"

    prompt = JUDGE_PROMPT.format(
        criteria="\n".join(f"- {c}" for c in criteria),
        user_prompt=user_prompt,
        output=answer,
        tool_calls=tool_calls_str,
    )

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        response_format=JudgeVerdict,
        temperature=0,
    )
    return response.choices[0].message.parsed
