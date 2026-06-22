"""Monitoring — append every agent interaction to logs.jsonl."""

import json
import time
import uuid
from pathlib import Path


def log_interaction(
    logs_file: Path,
    session_id: str,
    user_message: str,
    tool_calls: list[dict],
    answer: str,
    latency_ms: float,
    feedback: str | None = None,
) -> str:
    """Append one interaction to logs.jsonl. Returns the log entry id."""
    entry_id = str(uuid.uuid4())
    entry = {
        "id": entry_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "user_message": user_message,
        "tool_calls": tool_calls,
        "answer": answer,
        "latency_ms": round(latency_ms),
        "feedback": feedback,
    }
    logs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(logs_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry_id


def update_feedback(logs_file: Path, entry_id: str, feedback: str) -> bool:
    """Rewrite logs.jsonl updating the feedback field for a specific entry id."""
    if not logs_file.exists():
        return False
    lines = logs_file.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("id") == entry_id:
                entry["feedback"] = feedback
                updated = True
        except json.JSONDecodeError:
            pass
        new_lines.append(json.dumps(entry, ensure_ascii=False))
    if updated:
        logs_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


def load_logs(logs_file: Path) -> list[dict]:
    """Load all log entries from logs.jsonl."""
    if not logs_file.exists():
        return []
    entries = []
    for line in logs_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def export_thumbsup_as_eval_candidates(logs_file: Path) -> list[dict]:
    """Return thumbs-up interactions formatted as eval candidates."""
    return [
        {
            "question": e["user_message"],
            "answer": e["answer"],
            "tool_calls": e["tool_calls"],
        }
        for e in load_logs(logs_file)
        if e.get("feedback") == "up"
    ]
