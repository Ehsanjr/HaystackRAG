
from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
)

from src.config import TOP_K
from src.embedder import create_text_embedder
from src.vector_store import create_document_store


VALID_TEAMS = {"hamta", "karbordi"}
VALID_CONFIDENTIALITY = {"private", "public"}


def build_search_pipeline() -> Pipeline:
    """Build a semantic search pipeline over Qdrant."""

    document_store = create_document_store()

    text_embedder = create_text_embedder()
    text_embedder.warm_up()

    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=TOP_K,
    )

    pipeline = Pipeline()
    pipeline.add_component(
        name="text_embedder",
        instance=text_embedder,
    )
    pipeline.add_component(
        name="retriever",
        instance=retriever,
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding",
    )

    return pipeline


def build_filters(
    team: str | None = None,
    confidentiality: str | None = None,
) -> dict | None:
    """Build Qdrant metadata filters."""

    conditions = []

    if team is not None:
        if team not in VALID_TEAMS:
            raise ValueError(f"Invalid team: {team}")

        conditions.append({
            "field": "meta.team",
            "operator": "==",
            "value": team,
        })

    if confidentiality is not None:
        if confidentiality not in VALID_CONFIDENTIALITY:
            raise ValueError(
                f"Invalid confidentiality: {confidentiality}"
            )

        conditions.append({
            "field": "meta.confidentiality",
            "operator": "==",
            "value": confidentiality,
        })

    if not conditions:
        return None

    return {
        "operator": "AND",
        "conditions": conditions,
    }



def search_documents(
    query: str,
    team: str | None = None,
    confidentiality: str | None = None,
) -> list:
    """Search Qdrant with optional team and confidentiality filters."""

    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    filters = build_filters(team, confidentiality)
    pipeline = build_search_pipeline()

    inputs = {
        "text_embedder": {
            "text": query.strip(),
        }
    }

    if filters is not None:
        inputs["retriever"] = {
            "filters": filters,
        }

    result = pipeline.run(inputs)

    return result["retriever"]["documents"]
