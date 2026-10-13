"""Central settings. load_dotenv() runs before anything reads the environment.

Lesson carried over from the support-agent project: the .env file was silently
ignored because load_dotenv() was never called. It is the first thing here.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Provider = Literal["gemini", "groq"]
SearchMode = Literal["tavily", "ddg", "offline"]
LLMMode = Literal["live", "record", "replay"]

PROVIDER_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
}


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower() or default
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {allowed}, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    llm_provider: Provider
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    llm_requests_per_minute: int
    llm_max_concurrent: int
    llm_max_retries: int
    llm_timeout_seconds: int
    llm_mode: LLMMode
    search_mode: SearchMode
    tavily_api_key: str
    openai_api_key: str
    trace_export_openai: bool
    ollama_base_url: str
    embed_model: str
    data_dir: Path
    # Pipeline limits (code-orchestrated, not configurable by the LLM)
    max_critic_rounds: int = 3
    worker_timeout_seconds: int = 120
    review_confidence_threshold: float = 0.6
    sensitive_domains: tuple[str, ...] = field(
        default=("financial", "medical", "legal")
    )

    # ---- derived helpers ----
    @property
    def model_name(self) -> str:
        return self.gemini_model if self.llm_provider == "gemini" else self.groq_model

    @property
    def llm_api_key(self) -> str:
        return self.gemini_api_key if self.llm_provider == "gemini" else self.groq_api_key

    @property
    def llm_base_url(self) -> str:
        return PROVIDER_BASE_URLS[self.llm_provider]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "research.db"

    @property
    def sessions_db_path(self) -> Path:
        return self.data_dir / "sessions.db"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def corpus_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "corpus"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "llm"

    def require_llm_key(self) -> None:
        if self.llm_mode == "replay":
            return  # replay never calls the API
        if not self.llm_api_key:
            key_name = "GEMINI_API_KEY" if self.llm_provider == "gemini" else "GROQ_API_KEY"
            raise ConfigError(
                f"LLM_PROVIDER={self.llm_provider} but {key_name} is empty. "
                "Add it to your .env file (see .env.example)."
            )


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        llm_provider=_choice("LLM_PROVIDER", "gemini", ("gemini", "groq")),  # type: ignore[arg-type]
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        llm_requests_per_minute=_int("LLM_REQUESTS_PER_MINUTE", 10),
        llm_max_concurrent=_int("LLM_MAX_CONCURRENT", 3),
        llm_max_retries=_int("LLM_MAX_RETRIES", 4),
        llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 90),
        llm_mode=_choice("LLM_MODE", "live", ("live", "record", "replay")),  # type: ignore[arg-type]
        search_mode=_choice("SEARCH_MODE", "tavily", ("tavily", "ddg", "offline")),  # type: ignore[arg-type]
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        trace_export_openai=os.getenv("TRACE_EXPORT_OPENAI", "0").strip() == "1",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        embed_model=os.getenv("EMBED_MODEL", "bge-m3").strip(),
        data_dir=data_dir,
    )
