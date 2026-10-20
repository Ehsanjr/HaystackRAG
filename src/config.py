
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"

# Qdrant configuration
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "persian_documents_v1",
)

# Embedding configuration
MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

EMBEDDING_DIM = 384

# Chunking configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval configuration
TOP_K = 3
