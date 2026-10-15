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


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float(name: str, default: float, low: float = 0.0, high: float = 1.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
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
    llm_reasoning_effort: str  # off | none | minimal | low | medium | high
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
    worker_timeout_seconds: int = 480
    worker_retries: int = 1  # a worker that failed fast (exception, no evidence) runs once more
    review_confidence_threshold: float = 0.6
    min_independent_sources: int = 2  # fewer distinct cited sources → human review (corroboration gate)
    min_coverage_for_verdict: float = 0.5  # below this share of covered subquestions: no recommendation
    sensitive_domains: tuple[str, ...] = field(
        default=("financial", "medical", "legal")
    )
    # Models tried in order when the main model is overloaded (503), rate-limited (429) or times out.
    llm_fallbacks: tuple[tuple[str, str], ...] = ()
    llm_fallback_cooldown_seconds: int = 60
    # Embeddings for hybrid retrieval (RAG bonus). auto = Ollama → Gemini → BM25-only.
    embed_mode: str = "auto"  # auto | ollama | gemini | off
    gemini_embed_model: str = "gemini-embedding-001"
    embed_timeout_seconds: int = 30
    # Long-term memory (Phase 6). Only completed runs write; reused evidence passes every check again.
    memory_mode: str = "on"  # on | read_only | off
    memory_ttl_days: tuple[tuple[str, int], ...] = (
        ("pricing", 30), ("market", 90), ("risk", 90), ("technical", 180))
    memory_similarity_threshold: float = 0.4   # related past reports (shown, never sent to an agent)
    memory_min_match: float = 0.4              # IDF-weighted share of a subquestion's words an item must contain
    memory_max_reuse_per_subquestion: int = 4
    memory_coverage_min_items: int = 3
    memory_coverage_min_sources: int = 2
    memory_max_reverify: int = 10

    def memory_ttl(self, claim_type: str) -> int:
        table = dict(self.memory_ttl_days)
        return table.get(claim_type, min(table.values()))

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

    def api_key_for(self, provider: str) -> str:
        return self.gemini_api_key if provider == "gemini" else self.groq_api_key

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

    @property
    def embed_cache_path(self) -> Path:
        return self.data_dir / "cache" / "embeddings.db"

    def require_llm_key(self) -> None:
        if self.llm_mode == "replay":
            return  # replay never calls the API
        if not self.llm_api_key:
            key_name = "GEMINI_API_KEY" if self.llm_provider == "gemini" else "GROQ_API_KEY"
            raise ConfigError(
                f"LLM_PROVIDER={self.llm_provider} but {key_name} is empty. "
                "Add it to your .env file (see .env.example)."
            )


def _fallbacks(primary: str, gemini_model: str, groq_model: str,
               gemini_key: str, groq_key: str) -> tuple[tuple[str, str], ...]:
    """LLM_FALLBACKS="gemini:gemini-3.5-flash-lite,groq:qwen/qwen3.8-27b", or "none".

    Default: the other provider, if its key is set. Entries without a key are skipped.
    """
    raw = os.getenv("LLM_FALLBACKS", "").strip()
    if raw.lower() == "none":
        return ()
    if raw:
        out = []
        for item in raw.split(","):
            provider, sep, model = item.strip().partition(":")
            provider = provider.strip().lower()
            if not sep or provider not in PROVIDER_BASE_URLS or not model.strip():
                raise ConfigError(f"LLM_FALLBACKS entry must look like provider:model, got {item.strip()!r}")
            out.append((provider, model.strip()))
    else:
        out = [("groq", groq_model)] if primary == "gemini" else [("gemini", gemini_model)]
    keys = {"gemini": gemini_key, "groq": groq_key}
    return tuple((p, m) for p, m in out if keys[p] and (p, m) != (primary, gemini_model if primary == "gemini" else groq_model))


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        llm_provider=_choice("LLM_PROVIDER", "gemini", ("gemini", "groq")),  # type: ignore[arg-type]
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip(),
        llm_requests_per_minute=_int("LLM_REQUESTS_PER_MINUTE", 10),
        llm_max_concurrent=_int("LLM_MAX_CONCURRENT", 3),
        llm_max_retries=_int("LLM_MAX_RETRIES", 4, minimum=0),
        llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 120),
        worker_timeout_seconds=_int("WORKER_TIMEOUT_SECONDS", 480, minimum=30),
        worker_retries=_int("WORKER_RETRIES", 1, minimum=0),
        min_independent_sources=_int("MIN_INDEPENDENT_SOURCES", 2, minimum=1),
        llm_reasoning_effort=_choice(
            "LLM_REASONING_EFFORT", "low", ("off", "none", "minimal", "low", "medium", "high")
        ),
        llm_mode=_choice("LLM_MODE", "live", ("live", "record", "replay")),  # type: ignore[arg-type]
        search_mode=_choice("SEARCH_MODE", "tavily", ("tavily", "ddg", "offline")),  # type: ignore[arg-type]
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        trace_export_openai=os.getenv("TRACE_EXPORT_OPENAI", "0").strip() == "1",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        embed_model=os.getenv("EMBED_MODEL", "bge-m3").strip(),
        data_dir=data_dir,
        llm_fallbacks=_fallbacks(
            os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini",
            os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
            os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip(),
            os.getenv("GEMINI_API_KEY", "").strip(), os.getenv("GROQ_API_KEY", "").strip(),
        ),
        llm_fallback_cooldown_seconds=_int("LLM_FALLBACK_COOLDOWN_SECONDS", 60, minimum=0),
        max_critic_rounds=_int("MAX_CRITIC_ROUNDS", 3),
        embed_mode=_choice("EMBED_MODE", "auto", ("auto", "ollama", "gemini", "off")),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001").strip(),
        embed_timeout_seconds=_int("EMBED_TIMEOUT_SECONDS", 30),
        memory_mode=_choice("MEMORY_MODE", "on", ("on", "read_only", "off")),
        memory_ttl_days=(
            ("pricing", _int("MEMORY_TTL_PRICING", 30)), ("market", _int("MEMORY_TTL_MARKET", 90)),
            ("risk", _int("MEMORY_TTL_RISK", 90)), ("technical", _int("MEMORY_TTL_TECHNICAL", 180)),
        ),
        memory_similarity_threshold=_float("MEMORY_SIMILARITY_THRESHOLD", 0.4),
        memory_min_match=_float("MEMORY_MIN_MATCH", 0.4),
        memory_max_reuse_per_subquestion=_int("MEMORY_MAX_REUSE_PER_SUBQUESTION", 4),
        memory_coverage_min_items=_int("MEMORY_COVERAGE_MIN_ITEMS", 3),
        memory_coverage_min_sources=_int("MEMORY_COVERAGE_MIN_SOURCES", 2),
        memory_max_reverify=_int("MEMORY_MAX_REVERIFY", 10, minimum=0),
    )
