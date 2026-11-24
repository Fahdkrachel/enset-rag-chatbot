"""Ingestion : lecture du PDF, découpage en chunks et indexation dans Chroma."""
import os
import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf(data: bytes, filename: str):
    """Lit un PDF (octets) et renvoie une liste de pages (Documents LangChain).

    Le nom réel du fichier remplace le chemin temporaire dans les métadonnées,
    pour que les sources affichées soient lisibles.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.remove(path)
    for page in pages:
        page.metadata["source"] = filename
    return pages


def split_pages(pages, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Découpe les pages en chunks et supprime les chunks vides."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    return [c for c in chunks if c.page_content.strip()]


def build_index(chunks, embeddings):
    """Vectorise les chunks et les stocke dans une collection Chroma (en mémoire)."""
    if not chunks:
        raise ValueError(
            "Aucun texte extractible dans ce PDF (document scanné ou composé "
            "d'images ?). Un OCR serait nécessaire."
        )
    return Chroma.from_documents(
        chunks, embeddings, collection_name=f"pdf_{uuid.uuid4().hex[:8]}"
    )


def index_pdf(data: bytes, filename: str, embeddings, chunk_size: int = CHUNK_SIZE):
    """Pipeline complet d'ingestion. Renvoie (base_vectorielle, nombre_de_chunks)."""
    pages = load_pdf(data, filename)
    chunks = split_pages(pages, chunk_size=chunk_size)
    return build_index(chunks, embeddings), len(chunks)
