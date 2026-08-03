"""
Application configuration.

Loads:
1. Environment variables (.env)
2. YAML configuration (config.yaml)

Environment variables are used for machine-specific values.
YAML is used for application settings.
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------
# YAML Loader
# ---------------------------------------------------------------------


def load_yaml() -> dict:
    """Load config.yaml."""

    config_path = BASE_DIR / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


yaml_config = load_yaml()

# ---------------------------------------------------------------------
# YAML Configuration Models
# ---------------------------------------------------------------------


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
    temperature: float
    max_tokens: int
    timeout: int
    streaming: bool

class VectorStoreConfig(BaseModel):
    provider: str
    persist_directory: str
    collection_name: str

class EmbeddingConfig(BaseModel):
    """
    Embedding model configuration.
    """
    provider: str
    model_name: str
    device: str
    normalize_embeddings: bool


class RAGConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float
    max_context_chars: int

class RerankerConfig(BaseModel):
    """
    Cross-encoder reranker configuration.
    """

    enabled: bool
    model_name: str
    top_k: int

class MemoryConfig(BaseModel):
    max_history: int = 20


class SpeechConfig(BaseModel):
    whisper_model: str
    piper_voice: str


class LoggingConfig(BaseModel):
    level: str
    log_file: str


class APIConfig(BaseModel):
    host: str
    port: int

class WhisperSettings(BaseModel):
    model_name: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"

# ---------------------------------------------------------------------
# Environment Settings
# ---------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Environment-specific settings.

    These values can be overridden using .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
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


settings = Settings()

# ---------------------------------------------------------------------
# Structured Configuration Objects
# ---------------------------------------------------------------------

application = ApplicationConfig(**yaml_config["application"])

llm = LLMConfig(**yaml_config["llm"])

vectorstore = VectorStoreConfig(**yaml_config["vectorstore"])

embedding = EmbeddingConfig(**yaml_config["embedding"])

rag = RAGConfig(**yaml_config["rag"])

reranker = RerankerConfig(**yaml_config["reranker"])

memory = MemoryConfig(**yaml_config["memory"])

speech = SpeechConfig(**yaml_config["speech"])

logging_config = LoggingConfig(**yaml_config["logging"])

api = APIConfig(**yaml_config["api"])

whisper = WhisperSettings()