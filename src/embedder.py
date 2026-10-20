
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from src.config import MODEL_NAME


def create_document_embedder() -> SentenceTransformersDocumentEmbedder:
    """Create an embedder for documents before indexing."""
    return SentenceTransformersDocumentEmbedder(
        model=MODEL_NAME,
    )


def create_text_embedder() -> SentenceTransformersTextEmbedder:
    """Create an embedder for user search queries."""
    return SentenceTransformersTextEmbedder(
        model=MODEL_NAME,
    )
