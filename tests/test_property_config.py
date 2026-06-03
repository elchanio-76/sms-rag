"""Property-based tests for configuration validation.

# Feature: sms-rag, Property 17: Configuration Validation Errors

Validates: Requirements 9.6

For any configuration with one or more required fields missing or invalid,
the system SHALL fail at startup with an error message that identifies the
specific missing or invalid parameter by name.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sms_rag.shared.config import AppConfig, ConfigValidationError, validate_config


# --- Strategies for generating invalid config values ---

# Invalid chunk_size: zero or negative
invalid_chunk_size_st = st.integers(max_value=0)

# Invalid chunk_overlap: negative values
invalid_chunk_overlap_st = st.integers(max_value=-1)

# Invalid semantic_weight: outside [0.0, 1.0]
invalid_semantic_weight_st = st.one_of(
    st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.01, allow_nan=False, allow_infinity=False),
)

# Invalid top_k: zero or negative
invalid_top_k_st = st.integers(max_value=0)

# Invalid max_context_chunks: zero or negative
invalid_max_context_chunks_st = st.integers(max_value=0)

# Invalid llm_provider: not in {"ollama", "bedrock"}
invalid_llm_provider_st = st.text(min_size=1, max_size=20).filter(
    lambda x: x not in {"ollama", "bedrock"}
)

# Invalid vector_store_provider: not in {"chromadb", "qdrant", "pinecone"}
invalid_vector_store_provider_st = st.text(min_size=1, max_size=20).filter(
    lambda x: x not in {"chromadb", "qdrant", "pinecone"}
)

# Invalid server_port: outside [1, 65535]
invalid_server_port_st = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=65536),
)

# Invalid llm_timeout: zero or negative
invalid_llm_timeout_st = st.integers(max_value=0)

# Invalid vector_store_timeout: zero or negative
invalid_vector_store_timeout_st = st.integers(max_value=0)

# Invalid embedding_model_name: empty or whitespace-only
invalid_embedding_model_name_st = st.one_of(
    st.just(""),
    st.text(alphabet=" \t\n\r", min_size=1, max_size=10),
)


# --- config_generator strategy ---

# Map of field name to (invalid strategy, parameter name expected in error)
INVALID_FIELD_STRATEGIES = {
    "chunk_size": (invalid_chunk_size_st, "chunk_size"),
    "semantic_weight": (invalid_semantic_weight_st, "semantic_weight"),
    "top_k": (invalid_top_k_st, "top_k"),
    "max_context_chunks": (invalid_max_context_chunks_st, "max_context_chunks"),
    "llm_provider": (invalid_llm_provider_st, "llm_provider"),
    "vector_store_provider": (
        invalid_vector_store_provider_st,
        "vector_store_provider",
    ),
    "server_port": (invalid_server_port_st, "server_port"),
    "llm_timeout": (invalid_llm_timeout_st, "llm_timeout"),
    "vector_store_timeout": (invalid_vector_store_timeout_st, "vector_store_timeout"),
    "embedding_model_name": (invalid_embedding_model_name_st, "embedding_model_name"),
}


@st.composite
def config_generator(draw):
    """Generate AppConfig instances with at least one invalid field.

    Randomly selects 1 or more fields to make invalid, draws invalid values
    for those fields, and constructs an AppConfig with defaults for the rest.
    """
    # Choose which fields to invalidate (at least 1)
    field_names = list(INVALID_FIELD_STRATEGIES.keys())
    num_invalid = draw(st.integers(min_value=1, max_value=len(field_names)))
    chosen_fields = draw(
        st.lists(
            st.sampled_from(field_names),
            min_size=num_invalid,
            max_size=num_invalid,
            unique=True,
        )
    )

    # Build kwargs with invalid values for chosen fields
    kwargs = {}
    expected_params = []

    for field_name in chosen_fields:
        strategy, param_name = INVALID_FIELD_STRATEGIES[field_name]
        invalid_value = draw(strategy)
        kwargs[field_name] = invalid_value
        expected_params.append(param_name)

    # Handle special case: chunk_overlap >= chunk_size is also invalid
    # If chunk_size is invalid (<=0), chunk_overlap validation may differ
    # Keep chunk_overlap valid relative to chunk_size to focus on the chosen field
    if "chunk_size" in kwargs and "chunk_overlap" not in kwargs:
        # Set overlap to 0 so it doesn't trigger its own error
        kwargs["chunk_overlap"] = 0

    config = AppConfig(**kwargs)
    return config, expected_params


class TestConfigValidationProperty:
    """Property-based test: invalid configs produce named error messages."""

    # Feature: sms-rag, Property 17: Configuration Validation Errors

    @given(data=config_generator())
    @settings(max_examples=25)
    def test_invalid_config_raises_with_named_parameters(self, data):
        """For any configuration with invalid fields, validate_config SHALL
        raise ConfigValidationError with an error message that identifies
        each specific invalid parameter by name.

        Validates: Requirements 9.6
        """
        config, expected_params = data

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        error_message = str(exc_info.value)

        # Each invalid parameter must be named in the error message
        for param in expected_params:
            assert f"'{param}'" in error_message, (
                f"Expected parameter '{param}' to be named in error message, "
                f"but got: {error_message}"
            )

    @given(
        chunk_size=invalid_chunk_size_st,
    )
    @settings(max_examples=25)
    def test_negative_chunk_size_named_in_error(self, chunk_size):
        """Negative or zero chunk_size produces error naming 'chunk_size'.

        Validates: Requirements 9.6
        """
        config = AppConfig(chunk_size=chunk_size, chunk_overlap=0)

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "'chunk_size'" in str(exc_info.value)

    @given(
        semantic_weight=invalid_semantic_weight_st,
    )
    @settings(max_examples=25)
    def test_out_of_range_semantic_weight_named_in_error(self, semantic_weight):
        """Out-of-range semantic_weight produces error naming 'semantic_weight'.

        Validates: Requirements 9.6
        """
        config = AppConfig(semantic_weight=semantic_weight)

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "'semantic_weight'" in str(exc_info.value)

    @given(
        llm_provider=invalid_llm_provider_st,
    )
    @settings(max_examples=25)
    def test_invalid_llm_provider_named_in_error(self, llm_provider):
        """Invalid llm_provider produces error naming 'llm_provider'.

        Validates: Requirements 9.6
        """
        config = AppConfig(llm_provider=llm_provider)

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "'llm_provider'" in str(exc_info.value)

    @given(
        timeout=invalid_llm_timeout_st,
    )
    @settings(max_examples=25)
    def test_zero_or_negative_timeout_named_in_error(self, timeout):
        """Zero or negative llm_timeout produces error naming 'llm_timeout'.

        Validates: Requirements 9.6
        """
        config = AppConfig(llm_timeout=timeout)

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "'llm_timeout'" in str(exc_info.value)

    @given(
        port=invalid_server_port_st,
    )
    @settings(max_examples=25)
    def test_invalid_server_port_named_in_error(self, port):
        """Invalid server_port produces error naming 'server_port'.

        Validates: Requirements 9.6
        """
        config = AppConfig(server_port=port)

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "'server_port'" in str(exc_info.value)
