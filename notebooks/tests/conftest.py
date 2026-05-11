"""
conftest.py — shared fixtures for all agent tests.

Run from the notebooks/ directory:
    uv run pytest tests/ -v -s
"""

import json
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
NOTEBOOKS_DIR = Path(__file__).parent.parent
REPO_ROOT = NOTEBOOKS_DIR.parent
DATA_DIR = REPO_ROOT / "data"

# Add notebooks/ to path so tests can import test_agent
sys.path.insert(0, str(NOTEBOOKS_DIR))

load_dotenv(dotenv_path=REPO_ROOT / ".env")

# Import agent module — this builds the index and defines all tool functions
import test_agent as agent_module


# ── Session-scoped: index is built once ───────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def setup_index():
    """Rebuild the minsearch index from resources.json once per test session."""
    with open(DATA_DIR / "resources.json") as f:
        docs = json.load(f)
    agent_module.documents = docs
    agent_module.index.fit(docs)


# ── Function-scoped: each test gets an isolated progress file ─────────────────
@pytest.fixture
def fresh_progress(tmp_path, monkeypatch):
    """Empty progress state — no resources completed."""
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(json.dumps({
        "completed": [],
        "time_by_topic": {},
        "concepts_mastered": [],
    }))
    monkeypatch.setattr(agent_module, "PROGRESS_FILE", progress_file)
    return progress_file


@pytest.fixture
def progress_with_karpathy_done(tmp_path, monkeypatch):
    """Progress state where 'Neural Networks: Zero to Hero' is already completed."""
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(json.dumps({
        "completed": ["Neural Networks: Zero to Hero"],
        "time_by_topic": {"LLMs": 600},
        "concepts_mastered": ["backpropagation"],
    }))
    monkeypatch.setattr(agent_module, "PROGRESS_FILE", progress_file)
    return progress_file


@pytest.fixture
def agent(setup_index):
    """Return the agent function from the agent module."""
    return agent_module.agent
