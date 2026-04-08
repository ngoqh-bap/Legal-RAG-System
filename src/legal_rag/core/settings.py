from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    default_corpus_path: str = "data/processed/corpus.jsonl"


class ProvidersConfig(BaseModel):
    embeddings: str = "local_hash"
    llm: str = "local_extractive"


class IngestConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150


class RetrievalConfig(BaseModel):
    top_k: int = 5


class AppConfig(BaseModel):
    name: str = "legal-rag"


class Settings(BaseSettings):
    """
    Settings precedence:
    - explicit init kwargs
    - environment variables (.env loaded if present)
    - defaults in this class

    YAML config is optional and can be merged in via `load_settings(...)`.
    """

    model_config = SettingsConfigDict(env_prefix="LEGAL_RAG_", env_nested_delimiter="__", extra="ignore")

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def load_settings(config_path: str | None = None) -> Settings:
    load_dotenv(override=False)
    settings = Settings()

    if not config_path:
        return settings

    path = Path(config_path)
    if not path.exists():
        return settings

    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Re-parse by overlaying YAML on top of current settings dict.
    merged = settings.model_dump()
    _deep_update(merged, data)
    return Settings.model_validate(merged)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst
