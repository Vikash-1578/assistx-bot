"""
Strict configuration loader.

All environment variables are validated at import time via
`get_settings()`. If validation fails, the application MUST NOT start.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class _Base(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )


def _parse_allowed_ids(v):
    """Parse comma-separated IDs or a list. Called before pydantic validation."""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, str):
        return [int(x.strip()) for x in v.split(",") if x.strip()]
    raise ValueError("TELEGRAM_ALLOWED_USER_IDS must be comma-separated integers")


def _empty_to_none(v):
    """Convert empty string to None for optional fields."""
    if v is None or v == "":
        return None
    return v


class TelegramSettings(_Base):
    bot_token: str = Field(..., alias="BOT_TOKEN")
    allowed_user_ids: Annotated[
        list[int], NoDecode, BeforeValidator(_parse_allowed_ids)
    ] = Field(default_factory=list, alias="TELEGRAM_ALLOWED_USER_IDS")
    storage_channel_id: Annotated[
        int | None, BeforeValidator(_empty_to_none)
    ] = Field(default=None, alias="STORAGE_CHANNEL_ID")

    @field_validator("bot_token")
    @classmethod
    def _token_not_empty(cls, v: str) -> str:
        if not v or ":" not in v:
            raise ValueError("BOT_TOKEN is missing or malformed")
        return v


class AISettings(_Base):
    request_timeout: int = Field(45, alias="AI_REQUEST_TIMEOUT")
    max_retries: int = Field(2, alias="AI_MAX_RETRIES")
    max_output_tokens: int = Field(4096, alias="AI_MAX_OUTPUT_TOKENS")
    temperature: float = Field(0.7, alias="AI_TEMPERATURE")
    cooldown_seconds: int = Field(60, alias="AI_COOLDOWN_SECONDS")
    failure_threshold: int = Field(3, alias="AI_FAILURE_THRESHOLD")
    global_concurrency: int = Field(5, alias="GLOBAL_AI_CONCURRENCY")
    user_concurrency: int = Field(1, alias="USER_AI_CONCURRENCY")
    quota_db_path: str = Field("data/ai_usage.db", alias="AI_QUOTA_DB")


class ProviderSettings(_Base):
    """Per-provider configuration block."""

    enabled: bool = False
    api_key: str = ""
    base_url: str = ""
    daily_quota: int = 1000
    priority: int = 100

    model_chat: str = ""
    model_code: str = ""
    model_content: str = ""
    model_freelance: str = ""
    model_reasoning: str = ""
    model_summary: str = ""
    model_vision: str = ""

    def model_for(self, task: str) -> str:
        return getattr(self, f"model_{task}", "") or self.model_chat


class GroqSettings(ProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROQ_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )


class GeminiSettings(ProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )


class OpenRouterSettings(ProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )


class CerebrasSettings(ProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="CEREBRAS_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )


class SambaNovaSettings(ProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAMBANOVA_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )


class HuggingFaceSettings(ProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="HF_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )


class NvidiaSettings(ProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="NVIDIA_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )


class VectorSettings(_Base):
    backend: Literal["qdrant", "memory", "pinecone"] = Field(
        "memory", alias="VECTOR_BACKEND"
    )
    top_k: int = Field(5, alias="VECTOR_TOP_K")
    score_threshold: float = Field(0.35, alias="VECTOR_SCORE_THRESHOLD")

    qdrant_url: str = Field("", alias="QDRANT_URL")
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("telegram_ai_agent", alias="QDRANT_COLLECTION")

    pinecone_api_key: str = Field("", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("", alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field("default", alias="PINECONE_NAMESPACE")


class HealthSettings(_Base):
    host: str = Field("0.0.0.0", alias="HEALTH_HOST")
    port: int = Field(8080, alias="HEALTH_PORT")


class LoggingSettings(_Base):
    level: str = Field("INFO", alias="LOG_LEVEL")
    file: str = Field("logs/bot.log", alias="LOG_FILE")
    max_bytes: int = Field(5 * 1024 * 1024, alias="LOG_MAX_BYTES")
    backup_count: int = Field(5, alias="LOG_BACKUP_COUNT")


class Settings(_Base):
    """Root settings aggregate."""

    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    ai: AISettings = Field(default_factory=AISettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    health: HealthSettings = Field(default_factory=HealthSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    groq: GroqSettings = Field(default_factory=GroqSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    cerebras: CerebrasSettings = Field(default_factory=CerebrasSettings)
    sambanova: SambaNovaSettings = Field(default_factory=SambaNovaSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    nvidia: NvidiaSettings = Field(default_factory=NvidiaSettings)

    @model_validator(mode="after")
    def _at_least_one_provider(self) -> "Settings":
        providers = [
            self.groq,
            self.gemini,
            self.openrouter,
            self.cerebras,
            self.sambanova,
            self.huggingface,
            self.nvidia,
        ]
        usable = [p for p in providers if p.enabled and p.api_key]
        if not usable:
            raise ValueError(
                "No AI provider is configured. "
                "Enable at least one provider and provide its API key."
            )
        for p in providers:
            if p.enabled and not p.api_key:
                raise ValueError(
                    f"Provider is enabled but API key is missing: {type(p).__name__}"
                )
        return self

    def enabled_providers(self) -> dict[str, ProviderSettings]:
        return {
            "groq": self.groq,
            "gemini": self.gemini,
            "openrouter": self.openrouter,
            "cerebras": self.cerebras,
            "sambanova": self.sambanova,
            "huggingface": self.huggingface,
            "nvidia": self.nvidia,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached, validated settings. Raises on invalid config."""
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests to re-load settings."""
    get_settings.cache_clear()
