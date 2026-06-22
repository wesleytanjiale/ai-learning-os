import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # LLMs
    llm_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    agent_max_iterations: int = 10

    # Retrieval
    chunk_size_words: int = 400
    chunk_overlap_words: int = 50
    search_num_results: int = 5
    rrf_k: int = 60

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @property
    def queue_file(self) -> Path:
        return self.data_dir / "queue.json"

    @property
    def kb_file(self) -> Path:
        return self.data_dir / "kb.json"

    @property
    def progress_file(self) -> Path:
        return self.data_dir / "progress.json"

    @property
    def logs_file(self) -> Path:
        return self.data_dir / "logs.jsonl"


def get_config() -> Config:
    """Build config from environment variables (ALOS_ prefix overrides defaults)."""
    return Config(
        llm_model=os.getenv("ALOS_LLM_MODEL", "gpt-4o-mini"),
        vision_model=os.getenv("ALOS_VISION_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("ALOS_EMBEDDING_MODEL", "text-embedding-3-small"),
        agent_max_iterations=int(os.getenv("ALOS_MAX_ITERATIONS", "10")),
        chunk_size_words=int(os.getenv("ALOS_CHUNK_SIZE_WORDS", "400")),
        chunk_overlap_words=int(os.getenv("ALOS_CHUNK_OVERLAP_WORDS", "50")),
        search_num_results=int(os.getenv("ALOS_SEARCH_NUM_RESULTS", "5")),
        rrf_k=int(os.getenv("ALOS_RRF_K", "60")),
        data_dir=Path(os.getenv("ALOS_DATA_DIR", "data")),
    )
