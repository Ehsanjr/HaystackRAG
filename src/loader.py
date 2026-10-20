
from pathlib import Path

from haystack import Document

from src.config import BASE_DIR, DOCUMENTS_DIR


VALID_TEAMS = {"hamta", "karbordi"}
VALID_CONFIDENTIALITY = {"private", "public"}


def load_documents(folder: Path = DOCUMENTS_DIR) -> list[Document]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {folder}"
        )

    documents: list[Document] = []

    for file_path in sorted(folder.rglob("*.txt")):
        try:
            content = file_path.read_text(
                encoding="utf-8-sig"
            ).strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"File is not valid UTF-8: {file_path}"
            ) from exc

        if not content:
            print(f"Skipping empty file: {file_path.name}")
            continue

        # Extract team and confidentiality from the filename.
        parts = file_path.stem.lower().split("-")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid filename format: {file_path.name}. "
                "Expected: <team>-<confidentiality>.txt"
            )

        team, confidentiality = parts

        if team not in VALID_TEAMS:
            raise ValueError(
                f"Unknown team '{team}' in {file_path.name}"
            )

        if confidentiality not in VALID_CONFIDENTIALITY:
            raise ValueError(
                f"Unknown confidentiality '{confidentiality}' "
                f"in {file_path.name}"
            )

        source = file_path.relative_to(BASE_DIR).as_posix()

        documents.append(
            Document(
                content=content,
                meta={
                    "source": source,
                    "team": team,
                    "confidentiality": confidentiality,
                },
            )
        )

        print(
            f"Loaded: {source} | "
            f"team={team} | "
            f"confidentiality={confidentiality}"
        )

    if not documents:
        raise ValueError(
            f"No non-empty .txt files found in {folder}"
        )

    return documents
