"""Paramètres par défaut du projet.

Les valeurs sensibles à l'environnement peuvent être surchargées par des
variables d'environnement (voir le README, section « Configuration »).
"""
import os

# --- Ollama (LLM local) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))  # taille du contexte du LLM

# --- Embeddings (modèle multilingue : le document est en français) ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# --- Découpage du PDF ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Recherche ---
TOP_K = 5             # chunks envoyés au LLM pour écrire la réponse
K_SEARCH = 5          # chunks cherchés par requête (RAG amélioré)
N_VARIANTS = 3        # reformulations générées par la Query Expansion
MAX_CANDIDATES = 8    # chunks maximum à noter pendant le reranking
