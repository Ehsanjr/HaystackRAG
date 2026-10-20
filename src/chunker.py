
from haystack import Document

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_document(document: Document) -> list[Document]:
    """Split one document into overlapping character-based chunks."""

    if not document.content:
        return []

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE."
        )

    content = document.content
    chunks: list[Document] = []
    start = 0
    chunk_index = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))
        chunk_text = content[start:end].strip()

        if chunk_text:
            metadata = dict(document.meta or {})
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end,
                }
            )

            chunks.append(
                Document(
                    content=chunk_text,
                    meta=metadata,
                )
            )
            chunk_index += 1

        start += step

    return chunks


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split multiple documents while preserving their metadata."""

    chunks: list[Document] = []

    for document in documents:
        chunks.extend(split_document(document))

    return chunks
