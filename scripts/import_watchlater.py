"""
Batch-import from data/watchlater.json into the learning queue.

Reads the enriched JSON produced by enrich_watchlater.py — each entry already has
topic, channel, duration, etc. Fetches full transcripts and stages them for learning.

Usage:
    uv run python scripts/import_watchlater.py                   # import all
    uv run python scripts/import_watchlater.py --dry-run         # preview without importing
    uv run python scripts/import_watchlater.py --topic LLMs      # filter by topic
    uv run python scripts/import_watchlater.py --limit 5         # first N entries
    uv run python scripts/import_watchlater.py --difficulty beginner
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from ai_learning_os.config import get_config
from ai_learning_os.tools import add_to_queue

WATCHLATER_JSON = Path(__file__).parent.parent / "data" / "watchlater.json"


def main():
    parser = argparse.ArgumentParser(description="Import watchlater.json into the queue")
    parser.add_argument("--dry-run", action="store_true", help="Show entries without importing")
    parser.add_argument("--topic", default=None, help="Filter by topic (e.g. LLMs, RAG, Agents)")
    parser.add_argument("--difficulty", default="intermediate",
                        choices=["beginner", "intermediate", "advanced"],
                        help="Difficulty to assign (default: intermediate)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of entries to import")
    args = parser.parse_args()

    if not WATCHLATER_JSON.exists():
        print(f"Error: {WATCHLATER_JSON} not found.")
        print("Run 'make enrich-watchlater' first to produce watchlater.json.")
        sys.exit(1)

    entries = json.loads(WATCHLATER_JSON.read_text(encoding="utf-8"))

    if args.topic:
        entries = [e for e in entries if args.topic.lower() in e.get("topic", "").lower()]

    if args.limit:
        entries = entries[:args.limit]

    print(f"Found {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} to import")
    if args.topic:
        print(f"  Filtered to topic: {args.topic}")

    if args.dry_run:
        for e in entries:
            mins = e.get("duration_seconds", 0) // 60
            print(f"  [dry-run] [{e['topic']}] {e['title'][:70]} ({mins}m) — {e['url']}")
        return

    cfg = get_config()
    client = OpenAI()

    success, skipped, failed = 0, 0, 0
    for i, entry in enumerate(entries, 1):
        url = entry["url"]
        topic = entry.get("topic", "general")
        mins = entry.get("duration_seconds", 0) // 60
        print(f"\n[{i}/{len(entries)}] [{topic}] {entry['title'][:65]}... ({mins}m)")

        result = add_to_queue(
            source=url,
            topic=topic,
            difficulty=args.difficulty,
            cfg=cfg,
            client=client,
        )
        if result.get("already_queued"):
            print(f"  → Already in queue: skipped")
            skipped += 1
        elif result.get("error"):
            print(f"  → Error: {result['error']}")
            failed += 1
        else:
            print(f"  → Added: {result['title']} ({result['word_count']:,} words)")
            success += 1
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Done: {success} added, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
