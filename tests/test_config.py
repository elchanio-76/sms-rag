"""Unit tests for the configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from sms_rag.shared.config import (
    AppConfig,
    ConfigValidationError,
    load_config,
    validate_config,
    _apply_env_overrides,
    _load_yaml_config,
)


class TestAppConfigDefaults:
    """Test that AppConfig has correct default values."""

    def test_default_paths(self):
        config = AppConfig()
        assert config.data_dir == Path("data")
        assert config.vector_store_path == Path("storage/chroma")
        assert config.bm25_index_path == Path("storage/bm25_index.pkl")
        assert config.session_db_path == Path("storage/sessions.db")

    def test_default_embedding(self):
        config = AppConfig()
        assert config.embedding_model_name == "intfloat/multilingual-e5-large"

    def test_default_chunking(self):
        config = AppConfig()
        assert config.chunk_size == 20
        assert config.chunk_overlap == 5

    def test_default_retrieval(self):
        config = AppConfig()
        assert config.semantic_weight == 0.5
        assert config.top_k == 5
        assert config.max_context_chunks == 10

    def test_default_llm(self):
        config = AppConfig()
        assert config.llm_provider == "ollama"
        assert config.ollama_model == "llama3.1"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.bedrock_model_id is None
        assert config.bedrock_region == "us-east-1"

    def test_default_vector_store(self):
        config = AppConfig()
        assert config.vector_store_provider == "chromadb"

    def test_default_server(self):
        config = AppConfig()
        assert config.server_host == "0.0.0.0"
        assert config.server_port == 7860

    def test_default_timeouts(self):
        config = AppConfig()
        assert config.llm_timeout == 60
        assert config.vector_store_timeout == 30


class TestYamlLoading:
    """Test YAML config file loading."""

    def test_load_existing_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"data_dir": "/custom/data", "chunk_size": 30})
        )
        result = _load_yaml_config(config_file)
        assert result["data_dir"] == "/custom/data"
        assert result["chunk_size"] == 30

    def test_load_nonexistent_yaml(self, tmp_path):
        config_file = tmp_path / "nonexistent.yaml"
        result = _load_yaml_config(config_file)
        assert result == {}

    def test_load_empty_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        result = _load_yaml_config(config_file)
        assert result == {}

    def test_load_yaml_with_all_fields(self, tmp_path):
        config_data = {
            "data_dir": "/my/data",
            "vector_store_path": "/my/store",
            "bm25_index_path": "/my/bm25.pkl",
            "session_db_path": "/my/sessions.db",
            "embedding_model_name": "custom-model",
            "chunk_size": 50,
            "chunk_overlap": 10,
            "semantic_weight": 0.7,
            "top_k": 10,
            "max_context_chunks": 15,
            "llm_provider": "bedrock",
            "ollama_model": "mistral",
            "ollama_base_url": "http://remote:11434",
            "bedrock_model_id": "anthropic.claude-v2",
            "bedrock_region": "eu-west-1",
            "vector_store_provider": "qdrant",
            "server_host": "127.0.0.1",
            "server_port": 8080,
            "llm_timeout": 120,
            "vector_store_timeout": 45,
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))
        result = _load_yaml_config(config_file)
        assert result == config_data


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_env_override_string(self, monkeypatch):
        monkeypatch.setenv("SMS_RAG_DATA_DIR", "/env/data")
        result = _apply_env_overrides({})
        assert result["data_dir"] == "/env/data"

    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("SMS_RAG_CHUNK_SIZE", "50")
        result = _apply_env_overrides({})
        assert result["chunk_size"] == "50"

    def test_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("SMS_RAG_TOP_K", "20")
        result = _apply_env_overrides({"top_k": 5})
        assert result["top_k"] == "20"  # env takes priority

    def test_unrelated_env_vars_ignored(self, monkeypatch):
        monkeypatch.setenv("OTHER_VAR", "value")
        result = _apply_env_overrides({"top_k": 5})
        assert "other_var" not in result
        assert result == {"top_k": 5}


class TestLoadConfig:
    """Test end-to-end config loading."""

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"chunk_size": 40, "chunk_overlap": 8}))
        config = load_config(config_path=config_file)
        assert config.chunk_size == 40
        assert config.chunk_overlap == 8

    def test_load_with_env_override(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"chunk_size": 40}))
        monkeypatch.setenv("SMS_RAG_CHUNK_SIZE", "60")
        config = load_config(config_path=config_file)
        assert config.chunk_size == 60

    def test_load_nonexistent_uses_defaults(self, tmp_path):
        config_file = tmp_path / "nonexistent.yaml"
        config = load_config(config_path=config_file)
        assert config.chunk_size == 20
        assert config.data_dir == Path("data")

    def test_path_coercion(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"data_dir": "/custom/path"}))
        config = load_config(config_path=config_file)
        assert isinstance(config.data_dir, Path)
        assert config.data_dir == Path("/custom/path")

    def test_int_coercion_from_env(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        monkeypatch.setenv("SMS_RAG_SERVER_PORT", "9090")
        config = load_config(config_path=config_file)
        assert config.server_port == 9090

    def test_float_coercion_from_env(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        monkeypatch.setenv("SMS_RAG_SEMANTIC_WEIGHT", "0.8")
        config = load_config(config_path=config_file)
        assert config.semantic_weight == 0.8


class TestValidation:
    """Test configuration validation."""

    def test_valid_defaults_pass(self):
        config = AppConfig()
        validate_config(config)  # Should not raise

    def test_negative_chunk_size(self):
        config = AppConfig(chunk_size=-1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'chunk_size'" in str(exc_info.value)

    def test_zero_chunk_size(self):
        config = AppConfig(chunk_size=0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'chunk_size'" in str(exc_info.value)

    def test_negative_chunk_overlap(self):
        config = AppConfig(chunk_overlap=-1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'chunk_overlap'" in str(exc_info.value)

    def test_overlap_greater_than_size(self):
        config = AppConfig(chunk_size=10, chunk_overlap=10)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'chunk_overlap'" in str(exc_info.value)

    def test_semantic_weight_below_zero(self):
        config = AppConfig(semantic_weight=-0.1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'semantic_weight'" in str(exc_info.value)

    def test_semantic_weight_above_one(self):
        config = AppConfig(semantic_weight=1.1)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'semantic_weight'" in str(exc_info.value)

    def test_zero_top_k(self):
        config = AppConfig(top_k=0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'top_k'" in str(exc_info.value)

    def test_zero_max_context_chunks(self):
        config = AppConfig(max_context_chunks=0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'max_context_chunks'" in str(exc_info.value)

    def test_invalid_llm_provider(self):
        config = AppConfig(llm_provider="openai")
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'llm_provider'" in str(exc_info.value)

    def test_invalid_vector_store_provider(self):
        config = AppConfig(vector_store_provider="faiss")
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'vector_store_provider'" in str(exc_info.value)

    def test_invalid_server_port_zero(self):
        config = AppConfig(server_port=0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'server_port'" in str(exc_info.value)

    def test_invalid_server_port_too_high(self):
        config = AppConfig(server_port=70000)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'server_port'" in str(exc_info.value)

    def test_zero_llm_timeout(self):
        config = AppConfig(llm_timeout=0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'llm_timeout'" in str(exc_info.value)

    def test_zero_vector_store_timeout(self):
        config = AppConfig(vector_store_timeout=0)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'vector_store_timeout'" in str(exc_info.value)

    def test_bedrock_provider_without_model_id(self):
        config = AppConfig(llm_provider="bedrock", bedrock_model_id=None)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'bedrock_model_id'" in str(exc_info.value)

    def test_bedrock_provider_with_model_id_passes(self):
        config = AppConfig(
            llm_provider="bedrock", bedrock_model_id="anthropic.claude-v2"
        )
        validate_config(config)  # Should not raise

    def test_empty_embedding_model_name(self):
        config = AppConfig(embedding_model_name="")
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'embedding_model_name'" in str(exc_info.value)

    def test_multiple_errors_reported(self):
        config = AppConfig(chunk_size=-1, top_k=0, llm_provider="invalid")
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "'chunk_size'" in str(exc_info.value)
        assert "'top_k'" in str(exc_info.value)
        assert "'llm_provider'" in str(exc_info.value)

    def test_invalid_int_coercion(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        monkeypatch.setenv("SMS_RAG_CHUNK_SIZE", "not_a_number")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(config_path=config_file)
        assert "'chunk_size'" in str(exc_info.value)

    def test_invalid_float_coercion(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        monkeypatch.setenv("SMS_RAG_SEMANTIC_WEIGHT", "not_a_float")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(config_path=config_file)
        assert "'semantic_weight'" in str(exc_info.value)
