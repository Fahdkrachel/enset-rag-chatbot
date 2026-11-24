"""Briques du RAG avancé : Query Expansion, fusion/dédoublonnage et Reranking.

Ce module n'a aucune dépendance externe : il fonctionne avec n'importe quel
objet `llm` exposant une méthode `invoke(prompt) -> str`, ce qui permet de le
tester sans Ollama.
"""
import hashlib
import re
from itertools import zip_longest

from .config import MAX_CANDIDATES, N_VARIANTS

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def expand_query(query: str, llm, n: int = N_VARIANTS) -> list:
    """Query Expansion : demande au LLM `n` reformulations de la question.

    Renvoie [question d'origine] + reformulations. En cas d'erreur du LLM, la
    question d'origine est renvoyée seule : le RAG continue de fonctionner.
    """
    prompt = (
        f"Génère {n} reformulations différentes de la question ci-dessous. "
        "Garde le même sens mais utilise d'autres mots (synonymes, formulation "
        "plus précise). Réponds uniquement avec les questions, une par ligne, "
        "sans numérotation ni commentaire.\n\n"
        f"Question : {query}\n"
    )
    try:
        raw = llm.invoke(prompt)
    except Exception:
        return [query]

    variants = []
    for line in raw.splitlines():
        line = _BULLET.sub("", line).strip()  # retire puces et numérotation
        if (
            len(line) > 5
            and not line.endswith(":")  # ex. « Voici 3 questions : »
            and line.lower() != query.lower()
            and line not in variants
        ):
            variants.append(line)
    return [query] + variants[:n]


def merge_results(result_lists: list, max_docs: int = MAX_CANDIDATES) -> list:
    """Fusionne les résultats de plusieurs recherches.

    - mélange à tour de rôle (pour ne pas favoriser une seule requête) ;
    - supprime les doublons (même source, même page, même texte) ;
    - limite le nombre de candidats (chaque candidat coûte un appel au LLM).
    """
    mixed = [d for group in zip_longest(*result_lists) for d in group if d is not None]
    seen, unique = set(), []
    for doc in mixed:
        signature = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        key = (doc.metadata.get("source"), doc.metadata.get("page"), signature)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique[:max_docs]


def parse_score(raw: str) -> float:
    """Extrait une note entre 0 et 10 de la réponse du LLM (0 si illisible)."""
    match = _NUMBER.search(raw or "")
    if not match:
        return 0.0
    return min(max(float(match.group().replace(",", ".")), 0.0), 10.0)


def rerank_chunks(query: str, docs: list, llm, top_k: int = 5) -> list:
    """Reranking : le LLM note chaque chunk de 0 à 10, on garde les `top_k` meilleurs.

    La note est conservée dans `doc.metadata["rerank_score"]`.
    """
    scored = []
    for doc in docs:
        prompt = (
            "Tu évalues la pertinence d'un extrait de document pour répondre "
            "à une question.\n"
            "Donne une note entière de 0 (aucun rapport) à 10 "
            "(contient clairement la réponse).\n"
            "Réponds uniquement par le nombre.\n\n"
            f"Question : {query}\n\n"
            f"Extrait :\n{doc.page_content}\n\n"
            "Note :"
        )
        try:
            score = parse_score(llm.invoke(prompt))
        except Exception:
            score = 0.0
        doc.metadata["rerank_score"] = score
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)  # tri stable
    return [doc for _, doc in scored[:top_k]]
