"""
Central application configuration.

Configuration sources:
1. config.yaml - application defaults
2. .env - machine/environment overrides

The configuration objects exposed by this module are:

    settings
    application
    llm
    vectorstore
    embedding
    rag
    reranker
    memory
    speech
    logging_config
    api
    whisper
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =====================================================================
# Paths
# =====================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ENV_FILE = BASE_DIR / ".env"
CONFIG_FILE = BASE_DIR / "config.yaml"


# =====================================================================
# Environment
# =====================================================================

load_dotenv(ENV_FILE)


# =====================================================================
# YAML
# =====================================================================

def load_yaml() -> dict[str, Any]:
    """
    Load and validate the root configuration YAML.
    """

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "config.yaml must contain a YAML mapping/object."
        )

    return data


yaml_config = load_yaml()


# =====================================================================
# YAML Models
# =====================================================================

class ApplicationConfig(BaseModel):
    name: str
    author: str
    version: str
    environment: str
    debug: bool


class LLMConfig(BaseModel):
    provider: str
    model_name: str
    base_url: str
    temperature: float = Field(
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        gt=0,
    )
    timeout: int = Field(
        gt=0,
    )
    streaming: bool


class VectorStoreConfig(BaseModel):
    provider: str
    persist_directory: str
    collection_name: str


class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str
    device: str
    normalize_embeddings: bool


class RAGConfig(BaseModel):
    chunk_size: int = Field(
        gt=0,
    )
    chunk_overlap: int = Field(
        ge=0,
    )
    top_k: int = Field(
        gt=0,
    )
    similarity_threshold: float = Field(
        ge=0.0,
    )
    max_context_chars: int = Field(
        gt=0,
    )


class RerankerConfig(BaseModel):
    enabled: bool
    model_name: str
    top_k: int = Field(
        gt=0,
    )


class MemoryConfig(BaseModel):
    max_history: int = Field(
        default=20,
        gt=0,
    )


class SpeechConfig(BaseModel):
    whisper_model: str
    piper_voice: str


class LoggingConfig(BaseModel):
    level: str
    log_file: str


class APIConfig(BaseModel):
    host: str
    port: int = Field(
        gt=0,
        le=65535,
    )


class WhisperSettings(BaseModel):
    model_name: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"


# =====================================================================
# Environment Settings
# =====================================================================

class Settings(BaseSettings):
    """
    Environment-level settings.

    Values from .env override these defaults.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = yaml_config["application"]["name"]
    APP_VERSION: str = yaml_config["application"]["version"]

    DEBUG: bool = yaml_config["application"]["debug"]

    HOST: str = yaml_config["api"]["host"]
    PORT: int = yaml_config["api"]["port"]

    LOG_LEVEL: str = yaml_config["logging"]["level"]

    OLLAMA_BASE_URL: str = yaml_config["llm"]["base_url"]
    MODEL_NAME: str = yaml_config["llm"]["model_name"]

    EMBEDDING_MODEL: str = yaml_config["embedding"]["model_name"]

    VECTOR_DB: str = yaml_config["vectorstore"]["persist_directory"]


settings = Settings()


# =====================================================================
# Structured Configuration
# =====================================================================

application = ApplicationConfig(
    **yaml_config["application"]
)

llm = LLMConfig(
    **yaml_config["llm"]
)

vectorstore = VectorStoreConfig(
    **yaml_config["vectorstore"]
)

embedding = EmbeddingConfig(
    **yaml_config["embedding"]
)

rag = RAGConfig(
    **yaml_config["rag"]
)

reranker = RerankerConfig(
    **yaml_config["reranker"]
)

memory = MemoryConfig(
    **yaml_config["memory"]
)

speech = SpeechConfig(
    **yaml_config["speech"]
)

logging_config = LoggingConfig(
    **yaml_config["logging"]
)

api = APIConfig(
    **yaml_config["api"]
)

# Use the dedicated YAML whisper section if present.
whisper = WhisperSettings(
    **yaml_config.get(
        "whisper",
        {},
    )
)


# =====================================================================
# Configuration Validation
# =====================================================================

def validate_configuration() -> None:
    """
    Validate important cross-configuration assumptions.
    """

    if rag.chunk_overlap >= rag.chunk_size:
        raise ValueError(
            "RAG chunk_overlap must be smaller than chunk_size."
        )

    if reranker.enabled and reranker.top_k > rag.top_k:
        raise ValueError(
            "Reranker top_k cannot be greater than RAG top_k."
        )

    if llm.provider.lower() == "ollama":
        if not llm.base_url:
            raise ValueError(
                "Ollama provider requires llm.base_url."
            )

    if not embedding.model_name:
        raise ValueError(
            "Embedding model name cannot be empty."
        )


validate_configuration()


# =====================================================================
# Debug Helper
# =====================================================================

def print_configuration() -> None:
    """
    Print the effective application configuration.

    Secrets are intentionally not printed.
    """

    print("=" * 70)
    print("PersonalAiAssistant Configuration")
    print("=" * 70)

    print(
        f"Application : {application.name}"
    )
    print(
        f"Version     : {application.version}"
    )
    print(
        f"Environment : {application.environment}"
    )

    print()

    print(
        f"LLM Provider: {llm.provider}"
    )
    print(
        f"LLM Model   : {llm.model_name}"
    )
    print(
        f"LLM URL     : {llm.base_url}"
    )
    print(
        f"Temperature : {llm.temperature}"
    )
    print(
        f"Max Tokens  : {llm.max_tokens}"
    )
    print(
        f"Timeout     : {llm.timeout}"
    )

    print()

    print(
        f"Embedding   : {embedding.model_name}"
    )
    print(
        f"Device      : {embedding.device}"
    )

    print()

    print(
        f"Vector DB   : {vectorstore.provider}"
    )
    print(
        f"Collection  : {vectorstore.collection_name}"
    )
    print(
        f"Persistence : {vectorstore.persist_directory}"
    )

    print()

    print(
        f"RAG top_k   : {rag.top_k}"
    )
    print(
        f"Similarity  : {rag.similarity_threshold}"
    )
    print(
        f"Context max : {rag.max_context_chars}"
    )

    print()

    print(
        f"Reranker    : {reranker.enabled}"
    )
    print(
        f"Reranker    : {reranker.model_name}"
    )
    print(
        f"Reranker k  : {reranker.top_k}"
    )

    print()

    print(
        f"Memory      : {memory.max_history}"
    )

    print(
        f"Whisper     : {whisper.model_name}"
    )
    print(
        f"Whisper dev : {whisper.device}"
    )
    print(
        f"Whisper type: {whisper.compute_type}"
    )

    print("=" * 70)


# =====================================================================
# Standalone Test
# =====================================================================

if __name__ == "__main__":
    print_configuration()