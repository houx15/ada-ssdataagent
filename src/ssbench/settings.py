"""Global settings: paths and LLM endpoint configuration (from .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_concurrency: int
    repo_root: str
    data_dir: str
    processed_dir: str
    raw_data_dir: str
    runs_dir: str
    configs_dir: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    return Settings(
        llm_base_url=os.getenv("SDTL_LLM_BASE_URL", "http://localhost:39001/v1"),
        llm_api_key=os.getenv("SDTL_LLM_API_KEY", "not-needed"),
        llm_model=os.getenv("SDTL_LLM_MODEL", "glm-5.2"),
        llm_concurrency=int(os.getenv("SDTL_LLM_CONCURRENCY", "30")),
        repo_root=REPO_ROOT,
        data_dir=os.path.join(REPO_ROOT, "data"),
        processed_dir=os.path.join(REPO_ROOT, "data", "processed"),
        raw_data_dir=os.path.join(REPO_ROOT, "data", "real_data"),
        runs_dir=os.path.join(REPO_ROOT, "runs"),
        configs_dir=os.path.join(REPO_ROOT, "configs"),
    )
