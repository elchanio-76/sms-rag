"""Shared configuration system for SMS RAG pipelines.

Loads configuration from a YAML file at the project root, with environment
variable overrides using the SMS_RAG_ prefix. Validates all required fields
at startup and fails with named missing/invalid parameters.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import os
import yaml


class ConfigValidationError(Exception):
    """Raised when configuration validation fails at startup."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


@dataclass
class AppConfig:
    """Application configuration for both preprocessing and query pipelines."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    vector_store_path: Path = field(default_factory=lambda: Path("storage/chroma"))
    bm25_index_path: Path = field(
        default_factory=lambda: Path("storage/bm25_index.pkl")
    )
    session_db_path: Path = field(default_factory=lambda: Path("storage/sessions.db"))

    # Embedding
    embedding_model_name: str = "intfloat/multilingual-e5-large"

    # Chunking
    chunk_size: int = 20
    chunk_overlap: int = 5

    # Retrieval
    semantic_weight: float = 0.5
    top_k: int = 5
    max_context_chunks: int = 10

    # LLM
    llm_provider: str = "ollama"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    bedrock_model_id: str | None = None
    bedrock_region: str = "us-east-1"

    # Vector Store
    vector_store_provider: str = "chromadb"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 7860

    # Timeouts
    llm_timeout: int = 60
    vector_store_timeout: int = 30


def _resolve_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or config.yaml."""
    current = Path.cwd()
    # Walk up until we find a marker file
    for directory in [current, *current.parents]:
        if (directory / "pyproject.toml").exists() or (
            directory / "config.yaml"
        ).exists():
            return directory
    return current


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Returns an empty dict if the file does not exist.
    """
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides with SMS_RAG_ prefix.

    Environment variable names are uppercased field names with SMS_RAG_ prefix.
    Example: SMS_RAG_DATA_DIR overrides data_dir.
    """
    result = dict(config_dict)
    prefix = "SMS_RAG_"

    for key, value in os.environ.items():
        if key.startswith(prefix):
            field_name = key[len(prefix) :].lower()
            result[field_name] = value

    return result


def _coerce_value(field_name: str, raw_value: Any, field_type: type) -> Any:
    """Coerce a raw value to the expected field type."""
    if raw_value is None:
        return None

    # Handle Path fields
    if field_type is Path:
        return Path(str(raw_value))

    # Handle int fields
    if field_type is int:
        try:
            return int(raw_value)
        except (ValueError, TypeError) as exc:
            raise ConfigValidationError(
                [f"'{field_name}' must be an integer, got: {raw_value!r}"]
            ) from exc

    # Handle float fields
    if field_type is float:
        try:
            return float(raw_value)
        except (ValueError, TypeError) as exc:
            raise ConfigValidationError(
                [f"'{field_name}' must be a float, got: {raw_value!r}"]
            ) from exc

    # Handle optional string (str | None)
    if field_type is str:
        return str(raw_value)

    return raw_value


def _get_field_type(f) -> type:
    """Extract the base type from a dataclass field, handling Optional types."""
    type_hint = f.type
    # Handle "str | None" style annotations
    if hasattr(type_hint, "__origin__"):
        # For Union types (str | None)
        args = getattr(type_hint, "__args__", ())
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return non_none_args[0]
    return type_hint


def validate_config(config: AppConfig) -> None:
    """Validate configuration values at startup.

    Raises ConfigValidationError with all named invalid parameters.
    """
    errors: list[str] = []

    # Validate chunk_size is positive
    if config.chunk_size <= 0:
        errors.append(f"'chunk_size' must be positive, got: {config.chunk_size}")

    # Validate chunk_overlap is non-negative and less than chunk_size
    if config.chunk_overlap < 0:
        errors.append(
            f"'chunk_overlap' must be non-negative, got: {config.chunk_overlap}"
        )
    elif config.chunk_overlap >= config.chunk_size:
        errors.append(
            f"'chunk_overlap' ({config.chunk_overlap}) must be less than "
            f"'chunk_size' ({config.chunk_size})"
        )

    # Validate semantic_weight in [0.0, 1.0]
    if not (0.0 <= config.semantic_weight <= 1.0):
        errors.append(
            f"'semantic_weight' must be between 0.0 and 1.0, got: {config.semantic_weight}"
        )

    # Validate top_k is positive
    if config.top_k <= 0:
        errors.append(f"'top_k' must be positive, got: {config.top_k}")

    # Validate max_context_chunks is positive
    if config.max_context_chunks <= 0:
        errors.append(
            f"'max_context_chunks' must be positive, got: {config.max_context_chunks}"
        )

    # Validate llm_provider is a supported value
    valid_providers = {"ollama", "bedrock"}
    if config.llm_provider not in valid_providers:
        errors.append(
            f"'llm_provider' must be one of {sorted(valid_providers)}, "
            f"got: {config.llm_provider!r}"
        )

    # Validate vector_store_provider is a supported value
    valid_vector_stores = {"chromadb", "qdrant", "pinecone"}
    if config.vector_store_provider not in valid_vector_stores:
        errors.append(
            f"'vector_store_provider' must be one of {sorted(valid_vector_stores)}, "
            f"got: {config.vector_store_provider!r}"
        )

    # Validate server_port is in valid range
    if not (1 <= config.server_port <= 65535):
        errors.append(
            f"'server_port' must be between 1 and 65535, got: {config.server_port}"
        )

    # Validate timeouts are positive
    if config.llm_timeout <= 0:
        errors.append(f"'llm_timeout' must be positive, got: {config.llm_timeout}")

    if config.vector_store_timeout <= 0:
        errors.append(
            f"'vector_store_timeout' must be positive, got: {config.vector_store_timeout}"
        )

    # Validate bedrock_model_id is set when provider is bedrock
    if config.llm_provider == "bedrock" and not config.bedrock_model_id:
        errors.append("'bedrock_model_id' is required when 'llm_provider' is 'bedrock'")

    # Validate embedding_model_name is non-empty
    if not config.embedding_model_name or not config.embedding_model_name.strip():
        errors.append("'embedding_model_name' must not be empty")

    if errors:
        raise ConfigValidationError(errors)


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate application configuration.

    Configuration is loaded from:
    1. Default values defined in the AppConfig dataclass
    2. YAML config file (config.yaml at project root, or specified path)
    3. Environment variables with SMS_RAG_ prefix (highest priority)

    Raises:
        ConfigValidationError: If required fields are missing or invalid.
    """
    if config_path is None:
        project_root = _resolve_project_root()
        config_path = project_root / "config.yaml"

    # Load YAML config
    yaml_config = _load_yaml_config(config_path)

    # Apply environment variable overrides
    merged_config = _apply_env_overrides(yaml_config)

    # Build the AppConfig from merged values
    config_fields = fields(AppConfig)
    kwargs: dict[str, Any] = {}

    for f in config_fields:
        if f.name in merged_config:
            field_type = _get_field_type(f)
            kwargs[f.name] = _coerce_value(f.name, merged_config[f.name], field_type)

    config = AppConfig(**kwargs)

    # Validate the configuration
    validate_config(config)

    return config
