"""
Parse watchlater.txt, enrich each entry with YouTube metadata via yt-dlp,
auto-classify topic, and keep only AI engineering relevant content.

Produces data/watchlater.json — the structured source of truth.

Usage:
    uv run python scripts/enrich_watchlater.py              # enrich all entries
    uv run python scripts/enrich_watchlater.py --limit 10   # first 10 (test run)
    uv run python scripts/enrich_watchlater.py --resume     # skip already-enriched entries
    uv run python scripts/enrich_watchlater.py --all        # include non-AI content too
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WATCHLATER_TXT = REPO_ROOT / "data" / "watchlater.txt"
WATCHLATER_JSON = REPO_ROOT / "data" / "watchlater.json"

# ── Topic classification ───────────────────────────────────────────────────────

TOPIC_RULES: list[tuple[str, list[str]]] = [
    ("LLMs",            ["llm", "language model", "chatgpt", "gpt", "claude", "gemini",
                          "deepseek", "karpathy", "transformer", "tokeniz", "biology of a large"]),
    ("RAG",             ["rag", "retrieval", "vector search", "openrag", "chunking"]),
    ("Agents",          ["agent", "agentic", "mcp", "model context protocol", "orchestration",
                          "multi-agent", "tool use", "function call", "memory"]),
    ("Embeddings",      ["embedding", "word2vec", "sentence transformer", "semantic search",
                          "vector"]),
    ("Fine-tuning",     ["fine.tun", "finetune", "lora", "qlora", "rlhf", "grpo", "ppo",
                          "reinforcement learning from human", "dpo", "sft"]),
    ("RL",              ["reinforcement learning", "rlhf", "grpo", "ppo", "reward model"]),
    ("Deep Learning",   ["deep learning", "neural network", "pytorch", "backprop",
                          "gradient", "optimizer", "adamw", "autoencoder", "vae", "diffusion",
                          "gan", "stylegan"]),
    ("MLOps",           ["mlops", "deploy", "docker", "kubernetes", "fastapi", "production",
                          "monitoring", "observability"]),
    ("AI Engineering",  ["ai engineer", "llmops", "vibe cod", "skills.md", "claude code",
                          "ai app", "ai system", "build.*ai", "mcp"]),
    ("Graph ML",        ["graph neural", "gnn", "gcn", "cs224w", "graph convolutional",
                          "community detection"]),
    ("Evaluation",      ["eval", "benchmark", "judg", "assess"]),
    ("Career",          ["career", "interview", "job", "guide to becoming", "day of ai",
                          "ai engineer role", "stay informed"]),
]

AI_TOPICS = {t for t, _ in TOPIC_RULES}

NON_AI_SIGNALS = [
    "tennis", "footwork", "forehand", "backhand", "serve",
    "street photography", "color grading", "colour", "outfit", "fashion",
    "tokyo", "cpf", "retirement", "bench press", "workout", "xcode", "ios app",
    "avengers", "avatar", "trailer", "macbook repair",
]


def classify_topic(title: str, description: str) -> str | None:
    """
    Return the best-matching AI topic, or None if not AI-relevant.
    Checks title + description snippet against topic rules.
    """
    combined = (title + " " + description).lower()

    # Reject obvious non-AI content first
    for signal in NON_AI_SIGNALS:
        if signal in combined:
            return None

    for topic, keywords in TOPIC_RULES:
        for kw in keywords:
            if re.search(kw, combined, re.IGNORECASE):
                return topic

    return None


# ── yt-dlp metadata fetch ──────────────────────────────────────────────────────

def fetch_metadata(url: str) -> dict:
    """Fetch video metadata via yt-dlp without downloading."""
    fields = "%(channel)s\t%(upload_date)s\t%(duration)s\t%(description)s"
    try:
        result = subprocess.run(
            ["uv", "run", "yt-dlp", "--skip-download", "--no-warnings",
             "--print", fields, url],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}

        parts = result.stdout.strip().split("\t", 3)
        if len(parts) < 3:
            return {}

        channel = parts[0].strip() if parts[0] != "NA" else ""

        raw_date = parts[1].strip()
        upload_date = (
            f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            if raw_date and raw_date != "NA" and len(raw_date) == 8
            else ""
        )

        try:
            duration_seconds = int(float(parts[2].strip()))
        except (ValueError, TypeError):
            duration_seconds = 0

        description = (parts[3].strip() if len(parts) > 3 else "").replace("\n", " ")
        description_snippet = description[:300]

        return {
            "channel": channel,
            "upload_date": upload_date,
            "duration_seconds": duration_seconds,
            "description_snippet": description_snippet,
        }

    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_txt(path: Path) -> list[dict]:
    """Parse 'title , url' lines from watchlater.txt."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(.+?)\s*,\s*(https?://\S+)$", line)
        if match:
            entries.append({
                "title": match.group(1).strip(),
                "url": match.group(2).strip(),
            })
    return entries


def load_existing() -> dict[str, dict]:
    if not WATCHLATER_JSON.exists():
        return {}
    return {e["url"]: e for e in json.loads(WATCHLATER_JSON.read_text())}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-enriched entries")
    parser.add_argument("--all", action="store_true", help="Include non-AI content")
    args = parser.parse_args()

    if not WATCHLATER_TXT.exists():
        print(f"Error: {WATCHLATER_TXT} not found.")
        sys.exit(1)

    entries = parse_txt(WATCHLATER_TXT)
    print(f"Found {len(entries)} entries in watchlater.txt")

    existing = load_existing()
    if args.limit:
        entries = entries[:args.limit]

    results: list[dict] = []
    skipped = kept = dropped = failed = 0

    for i, entry in enumerate(entries, 1):
        url = entry["url"]

        if args.resume and url in existing:
            results.append(existing[url])
            skipped += 1
            continue

        print(f"[{i}/{len(entries)}] {entry['title'][:65]}...")
        meta = fetch_metadata(url)

        if meta.get("error"):
            print(f"  → Error: {meta['error']}")
            failed += 1
            # Still classify from title alone
            topic = classify_topic(entry["title"], "")
            if args.all or topic:
                results.append({
                    "title": entry["title"],
                    "url": url,
                    "channel": "",
                    "upload_date": "",
                    "duration_seconds": 0,
                    "description_snippet": "",
                    "topic": topic or "Unknown",
                    "queued": existing.get(url, {}).get("queued", False),
                    "error": meta["error"],
                })
            continue

        desc = meta.get("description_snippet", "")
        topic = classify_topic(entry["title"], desc)

        if not args.all and topic is None:
            print(f"  → Skipped (not AI-relevant)")
            dropped += 1
            continue

        mins = meta.get("duration_seconds", 0) // 60
        print(f"  → [{topic or 'Other'}] {meta.get('channel', '?')} | {meta.get('upload_date', '?')} | {mins}m")
        kept += 1

        results.append({
            "title": entry["title"],
            "url": url,
            "channel": meta.get("channel", ""),
            "upload_date": meta.get("upload_date", ""),
            "duration_seconds": meta.get("duration_seconds", 0),
            "description_snippet": desc,
            "topic": topic or "Other",
            "queued": existing.get(url, {}).get("queued", False),
        })

        time.sleep(0.3)

    # Sort by topic then upload_date (newest first)
    results.sort(key=lambda e: (e.get("topic", ""), e.get("upload_date", ""), ), reverse=False)

    WATCHLATER_JSON.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*55}")
    print(f"Kept (AI-relevant): {kept}  |  Dropped: {dropped}  |  Errors: {failed}  |  Skipped: {skipped}")
    print(f"Saved {len(results)} entries to: {WATCHLATER_JSON}")

    # Print summary by topic
    by_topic: dict[str, int] = {}
    for r in results:
        t = r.get("topic", "Unknown")
        by_topic[t] = by_topic.get(t, 0) + 1
    print("\nBy topic:")
    for t, count in sorted(by_topic.items()):
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
