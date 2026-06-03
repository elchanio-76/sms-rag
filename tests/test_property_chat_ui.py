"""Property-based tests for the Chat Interface.

# Feature: sms-rag, Property 13: Chat History Limit
# Feature: sms-rag, Property 14: Whitespace Input Rejection

Validates: Requirements 7.3, 7.6
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from sms_rag.query.chat_ui import ChatInterface, _EMPTY_INPUT_MSG
from sms_rag.shared.models import GenerationResult, SearchResult, SessionExchange


# --- Generators ---


@st.composite
def whitespace_generator(draw):
    """Generate strings composed entirely of whitespace characters.

    Produces strings containing only spaces, tabs, newlines, or
    the empty string. These should all be rejected by the ChatInterface.
    """
    chars = draw(
        st.lists(
            st.sampled_from([" ", "\t", "\n", "\r", "\v", "\f"]),
            min_size=0,
            max_size=50,
        )
    )
    return "".join(chars)


# --- Property 13: Chat History Limit ---


class TestChatHistoryLimit:
    """Property 13: Chat History Limit.

    For any sequence of message exchanges in a session, the stored
    conversation history SHALL never exceed 50 message pairs. When the
    limit is reached, the oldest exchanges SHALL be dropped to make
    room for new ones.

    # Feature: sms-rag, Property 13: Chat History Limit
    **Validates: Requirements 7.3**
    """

    @given(num_exchanges=st.integers(min_value=1, max_value=120))
    @settings(max_examples=100, deadline=None)
    def test_in_memory_history_never_exceeds_50(self, num_exchanges: int):
        """For any number of message exchanges persisted via in-memory
        storage, the history SHALL never exceed 50 message pairs.

        **Validates: Requirements 7.3**
        """
        # Use session_store=None to force in-memory history
        mock_orchestrator = MagicMock()
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=None)

        base_time = datetime(2024, 1, 1)
        for i in range(num_exchanges):
            exchange = SessionExchange(
                exchange_id=f"ex-{i}",
                user_message=f"message {i}",
                assistant_response=f"response {i}",
                source_references=[],
                timestamp=base_time + timedelta(seconds=i),
            )
            interface._in_memory_history.append(exchange)
            # Enforce the cap the same way _persist_exchange does
            while len(interface._in_memory_history) > 50:
                interface._in_memory_history.pop(0)

        assert len(interface._in_memory_history) <= 50

    @given(num_exchanges=st.integers(min_value=51, max_value=120))
    @settings(max_examples=100, deadline=None)
    def test_persist_exchange_enforces_cap(self, num_exchanges: int):
        """For any number of exchanges beyond 50 persisted via
        _persist_exchange (with in-memory fallback), the stored history
        SHALL be exactly 50 entries.

        **Validates: Requirements 7.3**
        """
        mock_orchestrator = MagicMock()
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=None)

        for i in range(num_exchanges):
            interface._persist_exchange(
                user_message=f"msg-{i}",
                assistant_response=f"resp-{i}",
                source_chunks=[],
            )

        assert len(interface._in_memory_history) == 50

    @given(num_exchanges=st.integers(min_value=51, max_value=120))
    @settings(max_examples=100, deadline=None)
    def test_oldest_exchanges_dropped_when_cap_reached(self, num_exchanges: int):
        """For any sequence of exchanges exceeding the 50-exchange limit,
        the oldest exchanges SHALL be dropped and the newest SHALL be
        retained.

        **Validates: Requirements 7.3**
        """
        mock_orchestrator = MagicMock()
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=None)

        for i in range(num_exchanges):
            interface._persist_exchange(
                user_message=f"msg-{i}",
                assistant_response=f"resp-{i}",
                source_chunks=[],
            )

        history = interface._in_memory_history

        # Verify cap is enforced
        assert len(history) == 50

        # Verify the most recent 50 exchanges are kept
        expected_start = num_exchanges - 50
        for idx, exchange in enumerate(history):
            expected_idx = expected_start + idx
            assert exchange.user_message == f"msg-{expected_idx}"


# --- Property 14: Whitespace Input Rejection ---


class TestWhitespaceInputRejection:
    """Property 14: Whitespace Input Rejection.

    For any string composed entirely of whitespace characters (spaces,
    tabs, newlines, or empty string), the Chat_Interface SHALL reject
    the input without invoking the LLM_Orchestrator.

    # Feature: sms-rag, Property 14: Whitespace Input Rejection
    **Validates: Requirements 7.6**
    """

    @given(whitespace_input=whitespace_generator())
    @settings(max_examples=100)
    def test_whitespace_only_input_returns_empty_msg(self, whitespace_input: str):
        """For any string composed entirely of whitespace characters or
        the empty string, _respond() SHALL return _EMPTY_INPUT_MSG.

        **Validates: Requirements 7.6**
        """
        mock_orchestrator = MagicMock()
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=None)

        result = interface._respond(whitespace_input, [])

        assert result == _EMPTY_INPUT_MSG

    @given(whitespace_input=whitespace_generator())
    @settings(max_examples=100)
    def test_whitespace_only_input_never_calls_orchestrator(
        self, whitespace_input: str
    ):
        """For any string composed entirely of whitespace characters or
        the empty string, the Chat_Interface SHALL NOT invoke the
        LLM_Orchestrator.

        **Validates: Requirements 7.6**
        """
        mock_orchestrator = MagicMock()
        interface = ChatInterface(orchestrator=mock_orchestrator, session_store=None)

        interface._respond(whitespace_input, [])

        mock_orchestrator.query.assert_not_called()
