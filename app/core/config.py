"""
Central application configuration.

Configuration priority:
    1. config.yaml provides application defaults.
    2. .env provides machine/environment overrides.
    3. Pydantic validates the resulting configuration.

The module exposes:
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


# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_FILE = BASE_DIR / "config.yaml"
ENV_FILE = BASE_DIR / ".env"


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv(dotenv_path=ENV_FILE)


# ============================================================================
# YAML LOADER
# ============================================================================


def load_yaml() -> dict[str, Any]:
    """
    Load and validate the root structure of config.yaml.
    """

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"Invalid YAML configuration: {CONFIG_FILE}"
        ) from exc

    if data is None:
        raise RuntimeError(
            f"Configuration file is empty: {CONFIG_FILE}"
        )

    if not isinstance(data, dict):
        raise RuntimeError(
            "config.yaml must contain a YAML mapping/object at the root."
        )

    return data


yaml_config = load_yaml()


def _section(name: str) -> dict[str, Any]:
    """
    Return a required configuration section.
    """

    value = yaml_config.get(name)

    if value is None:
        raise RuntimeError(
            f"Missing required configuration section: '{name}'"
        )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Configuration section '{name}' must be a mapping/object."
        )

    return value


# ============================================================================
# YAML CONFIGURATION MODELS
# ============================================================================


class ApplicationConfig(BaseModel):
    """Application metadata and runtime settings."""

    name: str
    author: str
    version: str
    environment: str
    debug: bool


class LLMConfig(BaseModel):
    """Local LLM provider configuration."""

    provider: str
    model_name: str
    base_url: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    timeout: int = Field(gt=0)
    streaming: bool


class VectorStoreConfig(BaseModel):
    """Vector database configuration."""

    provider: str
    persist_directory: str
    collection_name: str


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: str
    model_name: str
    device: str
    normalize_embeddings: bool


class RAGConfig(BaseModel):
    """Retrieval-Augmented Generation configuration."""

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    similarity_threshold: float
    max_context_chars: int = Field(gt=0)


class RerankerConfig(BaseModel):
    """Cross-encoder reranker configuration."""

    enabled: bool
    model_name: str
    top_k: int = Field(gt=0)


class MemoryConfig(BaseModel):
    """Conversation memory configuration."""

    max_history: int = Field(default=20, gt=0)


class SpeechConfig(BaseModel):
    """Speech-related configuration."""

    whisper_model: str
    piper_voice: str


class LoggingConfig(BaseModel):
    """Application logging configuration."""

    level: str
    log_file: str


class APIConfig(BaseModel):
    """FastAPI server configuration."""

    host: str
    port: int = Field(gt=0, le=65535)


class WhisperConfig(BaseModel):
    """Whisper speech-to-text runtime configuration."""

    model_name: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"


# ============================================================================
# ENVIRONMENT OVERRIDE SETTINGS
# ============================================================================


class Settings(BaseSettings):
    """
    Machine/environment-level settings.

    Values from .env override the defaults loaded from config.yaml.

    Example .env:

        APP_NAME=PersonalAiAssistant
        DEBUG=false
        HOST=127.0.0.1
        PORT=8000
        OLLAMA_BASE_URL=http://127.0.0.1:11434
        MODEL_NAME=qwen3:4b
        EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = _section("application")["name"]

    APP_VERSION: str = _section("application")["version"]

    DEBUG: bool = _section("application")["debug"]

    HOST: str = _section("api")["host"]

    PORT: int = _section("api")["port"]

    LOG_LEVEL: str = _section("logging")["level"]

    OLLAMA_BASE_URL: str = _section("llm")["base_url"]

    MODEL_NAME: str = _section("llm")["model_name"]

    EMBEDDING_MODEL: str = _section("embedding")["model_name"]


# ============================================================================
# GLOBAL ENVIRONMENT SETTINGS
# ============================================================================


settings = Settings()


# ============================================================================
# STRUCTURED CONFIGURATION
# ============================================================================


application = ApplicationConfig(
    **_section("application")
)

llm = LLMConfig(
    **_section("llm")
)

vectorstore = VectorStoreConfig(
    **_section("vectorstore")
)

embedding = EmbeddingConfig(
    **_section("embedding")
)

rag = RAGConfig(
    **_section("rag")
)

reranker = RerankerConfig(
    **_section("reranker")
)

memory = MemoryConfig(
    **_section("memory")
)

speech = SpeechConfig(
    **_section("speech")
)

logging_config = LoggingConfig(
    **_section("logging")
)

api = APIConfig(
    **_section("api")
)


# ============================================================================
# WHISPER
# ============================================================================

# Keep Whisper configuration independent from the speech YAML section
# until we verify the exact structure of config.yaml.
#
# These defaults match the current project configuration.

whisper = WhisperConfig(
    model_name="base",
    device="cpu",
    compute_type="int8",
)


# ============================================================================
# PATH HELPERS
# ============================================================================


def resolve_path(path: str | Path) -> Path:
    """
    Resolve a project-relative path against BASE_DIR.

    Absolute paths are returned unchanged.

    Example:
        resolve_path("./data/chroma")
        -> E:\\AiProjects\\PersonalAiAssistant\\data\\chroma
    """

    resolved = Path(path)

    if resolved.is_absolute():
        return resolved

    return BASE_DIR / resolved


def get_vectorstore_path() -> Path:
    """
    Return the absolute Chroma/vector-store directory.
    """

    return resolve_path(
        vectorstore.persist_directory
    )


def get_log_file_path() -> Path:
    """
    Return the absolute application log file path.
    """

    return resolve_path(
        logging_config.log_file
    )


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================


def validate_configuration() -> None:
    """
    Validate important runtime configuration values.

    This does not start Ollama, load models, or access Chroma.
    It only checks configuration consistency.
    """

    if rag.chunk_overlap >= rag.chunk_size:
        raise ValueError(
            "RAG chunk_overlap must be smaller than chunk_size."
        )

    if reranker.top_k > rag.top_k:
        raise ValueError(
            "Reranker top_k cannot be greater than RAG top_k."
        )

    if not llm.provider.strip():
        raise ValueError(
            "LLM provider cannot be empty."
        )

    if not llm.model_name.strip():
        raise ValueError(
            "LLM model_name cannot be empty."
        )

    if not llm.base_url.strip():
        raise ValueError(
            "LLM base_url cannot be empty."
        )

    if not embedding.model_name.strip():
        raise ValueError(
            "Embedding model_name cannot be empty."
        )


validate_configuration()


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================


def get_config_summary() -> dict[str, Any]:
    """
    Return a safe configuration summary.

    Secrets are intentionally not included.
    """

    return {
        "base_dir": str(BASE_DIR),
        "config_file": str(CONFIG_FILE),
        "environment_file": str(ENV_FILE),
        "application": {
            "name": application.name,
            "version": application.version,
            "environment": application.environment,
            "debug": application.debug,
        },
        "llm": {
            "provider": llm.provider,
            "model_name": llm.model_name,
            "base_url": llm.base_url,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
            "timeout": llm.timeout,
            "streaming": llm.streaming,
        },
        "embedding": {
            "provider": embedding.provider,
            "model_name": embedding.model_name,
            "device": embedding.device,
            "normalize_embeddings": embedding.normalize_embeddings,
        },
        "vectorstore": {
            "provider": vectorstore.provider,
            "persist_directory": str(
                get_vectorstore_path()
            ),
            "collection_name": vectorstore.collection_name,
        },
        "rag": {
            "chunk_size": rag.chunk_size,
            "chunk_overlap": rag.chunk_overlap,
            "top_k": rag.top_k,
            "similarity_threshold": rag.similarity_threshold,
            "max_context_chars": rag.max_context_chars,
        },
        "reranker": {
            "enabled": reranker.enabled,
            "model_name": reranker.model_name,
            "top_k": reranker.top_k,
        },
        "memory": {
            "max_history": memory.max_history,
        },
        "speech": {
            "whisper_model": speech.whisper_model,
            "piper_voice": speech.piper_voice,
        },
        "whisper": {
            "model_name": whisper.model_name,
            "device": whisper.device,
            "compute_type": whisper.compute_type,
        },
        "api": {
            "host": api.host,
            "port": api.port,
        },
        "logging": {
            "level": logging_config.level,
            "log_file": str(
                get_log_file_path()
            ),
        },
    }


# ============================================================================
# STANDALONE TEST
# ============================================================================


if __name__ == "__main__":

    print("=" * 80)
    print("PersonalAiAssistant Configuration")
    print("=" * 80)

    print()

    print(f"BASE_DIR      : {BASE_DIR}")
    print(f"CONFIG_FILE   : {CONFIG_FILE}")
    print(f"ENV_FILE      : {ENV_FILE}")

    print()

    print("-" * 80)
    print("Configuration")
    print("-" * 80)

    for section_name, section_data in get_config_summary().items():

        print(f"\n[{section_name}]")

        if isinstance(section_data, dict):

            for key, value in section_data.items():
                print(f"{key} = {value}")

        else:
            print(section_data)

    print()

    print("=" * 80)
    print("CONFIGURATION OK")
    print("=" * 80)