"""Chunk the synthetic Northstar policy documents, embed them, and upload to Azure AI Search.

Usage:
    python -m src.ingestion.ingest_policies
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from src.utils.azure_clients import get_azure_openai_client
from src.utils.config import settings, validate_settings

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"
EMBEDDING_DIMENSIONS = 1536
MAX_CHUNK_CHARS = 1500
CHUNK_OVERLAP_CHARS = 150
EMBEDDING_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 50

HEADER_PATTERN = re.compile(r"^##\s+(.*)$", re.MULTILINE)
TITLE_PATTERN = re.compile(r"^#\s+(.*)$", re.MULTILINE)
DOC_ID_PATTERN = re.compile(r"\*\*Document ID:\*\*\s*(\S+)")


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section_title: str
    chunk_index: int
    content: str
    source_path: str


def _sanitize_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-=]", "_", value)


def _split_long_section(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def load_and_chunk_documents() -> list[Chunk]:
    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {KNOWLEDGE_BASE_DIR}")

    all_chunks: list[Chunk] = []

    for path in md_files:
        text = path.read_text(encoding="utf-8")

        title_match = TITLE_PATTERN.search(text)
        doc_title = title_match.group(1).strip() if title_match else path.stem

        doc_id_match = DOC_ID_PATTERN.search(text)
        doc_id = doc_id_match.group(1).strip() if doc_id_match else path.stem

        header_matches = list(HEADER_PATTERN.finditer(text))
        chunk_index = 0

        if not header_matches:
            sections = [(doc_title, text)]
        else:
            sections = []
            for i, m in enumerate(header_matches):
                section_title = m.group(1).strip()
                start = m.end()
                end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
                sections.append((section_title, text[start:end].strip()))

        for section_title, body in sections:
            if not body:
                continue
            for piece in _split_long_section(body, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS):
                content = f"Document: {doc_title}\nSection: {section_title}\n\n{piece}"
                all_chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        section_title=section_title,
                        chunk_index=chunk_index,
                        content=content,
                        source_path=path.name,
                    )
                )
                chunk_index += 1

        print(f"  {path.name}: {chunk_index} chunks")

    return all_chunks


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    client = get_azure_openai_client()
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=[c.content for c in batch],
        )
        embeddings.extend([item.embedding for item in response.data])

    return embeddings


def ensure_index_exists() -> None:
    index_client = SearchIndexClient(
        endpoint=settings.SEARCH_ENDPOINT,
        credential=AzureKeyCredential(settings.SEARCH_KEY),
    )

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="northstar-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="northstar-vector-profile",
                algorithm_configuration_name="northstar-hnsw",
            )
        ],
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="doc_title", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="section_title", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="source_path", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="northstar-vector-profile",
        ),
    ]

    index = SearchIndex(name=settings.SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Index '{settings.SEARCH_INDEX_NAME}' ready.")


def upload_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    search_client = SearchClient(
        endpoint=settings.SEARCH_ENDPOINT,
        index_name=settings.SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.SEARCH_KEY),
    )

    documents = []
    for chunk, vector in zip(chunks, embeddings):
        key = _sanitize_key(f"{chunk.doc_id}_{chunk.chunk_index}")
        documents.append(
            {
                "id": key,
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "source_path": chunk.source_path,
                "content_vector": vector,
            }
        )

    for i in range(0, len(documents), UPLOAD_BATCH_SIZE):
        batch = documents[i : i + UPLOAD_BATCH_SIZE]
        result = search_client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        if failed:
            raise RuntimeError(f"Failed to upload {len(failed)} documents: {failed}")

    print(f"Uploaded {len(documents)} chunks to '{settings.SEARCH_INDEX_NAME}'.")


def main() -> None:
    validate_settings()

    print(f"Reading policy documents from {KNOWLEDGE_BASE_DIR}")
    chunks = load_and_chunk_documents()
    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("Ensuring search index exists...")
    ensure_index_exists()

    print("Uploading chunks...")
    upload_chunks(chunks, embeddings)

    print("Ingestion complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise