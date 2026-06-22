"""Shared fixtures for agent tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from ai_learning_os.agent import agent as run_agent
from ai_learning_os.config import Config
from ai_learning_os import tools as tools_module
from ai_learning_os import retrieval as retrieval_module

REPO_ROOT = Path(__file__).parent.parent


# ── Sample KB entry (pre-chunked, pre-embedded) ───────────────────────────────

SAMPLE_KB_ENTRY = {
    "id": "test-001",
    "title": "Neural Networks: Zero to Hero",
    "url": "https://karpathy.ai/zero-to-hero.html",
    "source_type": "youtube",
    "topic": "LLMs",
    "difficulty": "intermediate",
    "synthesis": "Karpathy's series builds neural networks from scratch, covering backpropagation, MLPs, and the transformer architecture.",
    "key_concepts": ["backpropagation", "transformer", "attention mechanism", "MLP"],
    "learned_at": "2026-06-21T10:00:00Z",
    "chunks": [
        {
            "chunk_id": "test-001-0",
            "text": "We start by building a bigram language model to understand how gradients flow through a neural network. The key insight is backpropagation: each parameter receives a gradient that tells it which direction to move to reduce the loss.",
            "source_title": "Neural Networks: Zero to Hero",
            "source_url": "https://karpathy.ai/zero-to-hero.html",
            "topic": "LLMs",
            "embedding": [0.1] * 1536,
        },
        {
            "chunk_id": "test-001-1",
            "text": "The attention mechanism allows the model to weigh different tokens differently when predicting the next token. Query, key, and value matrices are learned projections that compute relevance scores.",
            "source_title": "Neural Networks: Zero to Hero",
            "source_url": "https://karpathy.ai/zero-to-hero.html",
            "topic": "LLMs",
            "embedding": [0.2] * 1536,
        },
    ],
}

SAMPLE_QUEUE_ENTRY = {
    "id": "q-001",
    "title": "RAG From Scratch",
    "url": "https://www.youtube.com/playlist?list=PLfaIDFEXuae2",
    "source_type": "youtube",
    "topic": "RAG",
    "difficulty": "intermediate",
    "full_text": "Retrieval-augmented generation combines a retriever with a language model. The retriever finds relevant documents; the LLM synthesizes an answer.",
    "added_at": "2026-06-21T09:00:00Z",
}


# ── Config backed by tmp files ─────────────────────────────────────────────────

@pytest.fixture
def tmp_cfg(tmp_path) -> Config:
    """Config pointing to isolated tmp data files."""
    cfg = Config(data_dir=tmp_path)
    return cfg


@pytest.fixture
def cfg_with_kb(tmp_cfg) -> Config:
    """Config with one KB entry and empty queue."""
    kb = [SAMPLE_KB_ENTRY]
    tmp_cfg.kb_file.write_text(json.dumps(kb))
    tmp_cfg.queue_file.write_text("[]")
    tmp_cfg.progress_file.write_text(json.dumps({
        "completed": ["Neural Networks: Zero to Hero"],
        "time_by_topic": {"LLMs": 300},
        "concepts_mastered": ["backpropagation", "attention mechanism"],
    }))
    retrieval_module.rebuild_index(kb, tmp_cfg.rrf_k)
    return tmp_cfg


@pytest.fixture
def cfg_with_queue(tmp_cfg) -> Config:
    """Config with one queue entry and empty KB."""
    tmp_cfg.kb_file.write_text("[]")
    tmp_cfg.queue_file.write_text(json.dumps([SAMPLE_QUEUE_ENTRY]))
    tmp_cfg.progress_file.write_text(json.dumps({
        "completed": [],
        "time_by_topic": {},
        "concepts_mastered": [],
    }))
    retrieval_module.rebuild_index([], tmp_cfg.rrf_k)
    return tmp_cfg


# ── OpenAI client (real, uses env key) ────────────────────────────────────────

@pytest.fixture(scope="session")
def openai_client() -> OpenAI:
    return OpenAI()
