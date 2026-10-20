
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from src.chunker import split_documents
from src.embedder import create_document_embedder
from src.loader import load_documents
from src.vector_store import create_document_store


def build_indexing_pipeline():
    """Build the embedding-to-Qdrant indexing pipeline."""

    document_store = create_document_store()

    document_embedder = create_document_embedder()
    document_embedder.warm_up()

    document_writer = DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    pipeline = Pipeline()
    pipeline.add_component(
        name="embedder",
        instance=document_embedder,
    )
    pipeline.add_component(
        name="writer",
        instance=document_writer,
    )

    pipeline.connect(
        "embedder.documents",
        "writer.documents",
    )

    return pipeline, document_store


def index_documents() -> dict:
    """Load, chunk, embed, and write documents to Qdrant."""

    documents = load_documents()
    chunks = split_documents(documents)

    pipeline, document_store = build_indexing_pipeline()

    result = pipeline.run(
        {
            "embedder": {
                "documents": chunks,
            }
        }
    )

    return {
        "source_documents": len(documents),
        "chunks_created": len(chunks),
        "chunks_written": result["writer"]["documents_written"],
        "total_documents_in_qdrant": (
            document_store.count_documents()
        ),
    }
