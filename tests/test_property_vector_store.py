"""Property-based tests for the ChromaDB vector store metadata round-trip.

# Feature: sms-rag, Property 7: Vector Store Metadata Round-Trip

Validates: Requirements 4.3

Property 7: For any set of embeddings stored with metadata dictionaries,
querying those embeddings back SHALL return metadata dictionaries that are
equivalent to the originals, and filtering by any stored metadata field SHALL
correctly include/exclude matching documents.
"""

import tempfile
import uuid
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from sms_rag.shared.chroma_store import ChromaStore


# --- Generators ---

EMBEDDING_DIM = 10


@st.composite
def metadata_generator(draw):
    """Generate random metadata dicts with participant_name, source_filename,
    message_types, and phone_numbers."""
    participant_name = draw(
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "Z"),
                blacklist_characters="\x00",
            ),
        )
    )

    source_filename = draw(
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                blacklist_characters="\x00",
            ),
        ).map(lambda s: s + ".pdf")
    )

    message_types = draw(
        st.lists(
            st.sampled_from(["SMS", "iMessage", "RCS"]),
            min_size=0,
            max_size=3,
        )
    )

    phone_numbers = draw(
        st.lists(
            st.from_regex(r"\+\d{10,12}", fullmatch=True),
            min_size=0,
            max_size=3,
        )
    )

    return {
        "participant_name": participant_name,
        "source_filename": source_filename,
        "message_types": message_types,
        "phone_numbers": phone_numbers,
    }


@st.composite
def embedding_generator(draw, dim=EMBEDDING_DIM):
    """Generate a random embedding vector of fixed dimensionality."""
    components = draw(
        st.lists(
            st.floats(
                min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            min_size=dim,
            max_size=dim,
        )
    )
    return components


# --- Property 7: Vector Store Metadata Round-Trip ---


class TestVectorStoreMetadataRoundTrip:
    """Property 7: Vector Store Metadata Round-Trip.

    # Feature: sms-rag, Property 7: Vector Store Metadata Round-Trip
    """

    @given(
        metadata=metadata_generator(),
        embedding=embedding_generator(),
    )
    @settings(max_examples=25, deadline=None)
    def test_metadata_round_trip_equivalence(self, metadata, embedding):
        """For any embeddings stored with metadata, querying them back SHALL return
        metadata dictionaries equivalent to the originals.

        **Validates: Requirements 4.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "chroma"
            store = ChromaStore(persist_path=store_path, timeout=30)

            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            document_text = "Test document content"

            # Store embedding with metadata
            store.store_embeddings(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[metadata],
            )

            # Query it back using the same embedding
            results = store.query_by_similarity(
                query_embedding=embedding,
                top_k=1,
            )

            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            result_metadata = results[0].metadata

            # Verify metadata equivalence (None values are excluded by ChromaStore)
            for key, value in metadata.items():
                if value is None:
                    assert (
                        key not in result_metadata
                    ), f"None-valued key '{key}' should not be in stored metadata"
                else:
                    assert (
                        key in result_metadata
                    ), f"Key '{key}' missing from result metadata"
                    assert result_metadata[key] == value, (
                        f"Metadata mismatch for '{key}': "
                        f"stored={result_metadata[key]}, expected={value}"
                    )

    @given(
        metadatas=st.lists(metadata_generator(), min_size=2, max_size=5),
        filter_field=st.sampled_from(["participant_name", "source_filename"]),
    )
    @settings(max_examples=25, deadline=None)
    def test_metadata_filtering_includes_excludes_correctly(
        self, metadatas, filter_field
    ):
        """Filtering by any stored metadata field SHALL correctly include/exclude
        matching documents.

        **Validates: Requirements 4.3**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "chroma"
            store = ChromaStore(persist_path=store_path, timeout=30)

            # Generate unique embeddings per document (use distinct directions)
            num_docs = len(metadatas)
            embeddings = []
            for i in range(num_docs):
                emb = [0.0] * EMBEDDING_DIM
                emb[i % EMBEDDING_DIM] = 1.0
                embeddings.append(emb)

            ids = [f"doc_{i}_{uuid.uuid4().hex[:8]}" for i in range(num_docs)]
            documents = [f"Document content {i}" for i in range(num_docs)]

            # Store all documents
            store.store_embeddings(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            # Pick the filter value from the first metadata entry
            filter_value = metadatas[0][filter_field]

            # Determine which documents should match the filter
            expected_matching_ids = set()
            for i, m in enumerate(metadatas):
                if m[filter_field] == filter_value:
                    expected_matching_ids.add(ids[i])

            # Query with filter using a neutral embedding
            query_embedding = [1.0 / EMBEDDING_DIM**0.5] * EMBEDDING_DIM
            results = store.query_by_similarity(
                query_embedding=query_embedding,
                top_k=num_docs,
                metadata_filters={filter_field: filter_value},
            )

            # Verify only matching documents are returned
            result_ids = {r.chunk_id for r in results}
            assert result_ids == expected_matching_ids, (
                f"Filter on '{filter_field}'='{filter_value}': "
                f"expected IDs={expected_matching_ids}, got IDs={result_ids}"
            )

            # Verify all returned results have the correct filter value
            for result in results:
                assert result.metadata[filter_field] == filter_value, (
                    f"Result {result.chunk_id} has {filter_field}="
                    f"'{result.metadata[filter_field]}', expected '{filter_value}'"
                )
