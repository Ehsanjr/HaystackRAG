
from haystack_integrations.document_stores.qdrant import (
    QdrantDocumentStore,
)

from src.config import (
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_URL,
)


def create_document_store() -> QdrantDocumentStore:
    """Create a connection to the persistent Qdrant collection."""

    return QdrantDocumentStore(
        url=QDRANT_URL,
        index=QDRANT_COLLECTION,
        embedding_dim=EMBEDDING_DIM,
        similarity="cosine",
        recreate_index=False,
        wait_result_from_api=True,
    )
