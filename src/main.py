
import argparse

from src.config import TOP_K
from src.indexing import index_documents
from src.retriever import build_search_pipeline


def run_index() -> None:
    print("Starting indexing...\n")

    summary = index_documents()

    print("\nIndexing completed:")
    for key, value in summary.items():
        print(f"{key}: {value}")


def run_search() -> None:
    print("Initializing semantic search...")
    pipeline = build_search_pipeline()

    print("\nSemantic search is ready.")
    print("Enter a Persian or English query.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSearch stopped.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Search stopped.")
            break

        if not query:
            continue

        try:
            result = pipeline.run(
                {
                    "text_embedder": {
                        "text": query,
                    }
                }
            )

            documents = result["retriever"]["documents"]

            if not documents:
                print("No results found.\n")
                continue

            print(f"\nTop {len(documents)} results:\n")

            for rank, document in enumerate(documents, start=1):
                source = document.meta.get("source", "unknown")
                score = (
                    f"{document.score:.4f}"
                    if document.score is not None
                    else "N/A"
                )

                print(f"--- Result {rank} | score={score} ---")
                print(f"Source: {source}")
                print(document.content)
                print()

        except Exception as exc:
            print(f"Search failed: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persian RAG project using Haystack and Qdrant"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "index",
        help="Load, chunk, embed, and store documents",
    )

    subparsers.add_parser(
        "search",
        help=f"Search indexed documents (top_k={TOP_K})",
    )

    args = parser.parse_args()

    if args.command == "index":
        run_index()
    elif args.command == "search":
        run_search()


if __name__ == "__main__":
    main()
