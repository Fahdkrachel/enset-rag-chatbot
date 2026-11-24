"""Pipelines RAG : classique et amélioré (Query Expansion + Reranking)."""
import time
from dataclasses import dataclass

from .config import K_SEARCH, MAX_CANDIDATES, N_VARIANTS, TOP_K
from .retrieval import expand_query, merge_results, rerank_chunks

ANSWER_PROMPT = """Tu dois répondre UNIQUEMENT à partir du contexte ci-dessous, en français.
Si la réponse n'est pas dans le contexte, réponds exactement :
"Je ne sais pas d'après le document fourni."

Contexte :
{context}

Question :
{question}

Réponse :
"""


@dataclass
class RAGResult:
    """Résultat d'un pipeline, prêt à être affiché par l'interface."""

    title: str
    answer: str
    sources: list
    elapsed: float   # temps total en secondes
    details: str     # explication des étapes (markdown)


def format_sources(docs: list) -> list:
    """Liste triée et sans doublons : « fichier.pdf (page N) »."""
    out = set()
    for doc in docs:
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page")
        out.add(f"{source} (page {page + 1})" if isinstance(page, int) else str(source))
    return sorted(out)


def generate_answer(query: str, docs: list, llm) -> str:
    """Étape « Generation » : le LLM répond à partir des chunks fournis."""
    context = "\n\n".join(d.page_content for d in docs)
    return llm.invoke(ANSWER_PROMPT.format(context=context, question=query)).strip()


def run_classic(query: str, db, llm, top_k: int = TOP_K) -> RAGResult:
    """RAG classique : recherche vectorielle simple, puis génération."""
    t0 = time.perf_counter()
    docs = db.as_retriever(search_kwargs={"k": top_k}).invoke(query)
    t_search = time.perf_counter() - t0

    t1 = time.perf_counter()
    answer = generate_answer(query, docs, llm)
    t_gen = time.perf_counter() - t1

    return RAGResult(
        title="RAG classique",
        answer=answer,
        sources=format_sources(docs),
        elapsed=time.perf_counter() - t0,
        details=(
            f"Recherche vectorielle avec la question d'origine ; {top_k} chunks "
            f"envoyés au LLM.\n\nRecherche : {t_search:.1f} s · "
            f"Génération : {t_gen:.1f} s"
        ),
    )


def run_advanced(
    query: str,
    db,
    llm,
    top_k: int = TOP_K,
    k_search: int = K_SEARCH,
    n_variants: int = N_VARIANTS,
    max_candidates: int = MAX_CANDIDATES,
) -> RAGResult:
    """RAG amélioré : Query Expansion -> multi-recherche -> Reranking -> génération."""
    t0 = time.perf_counter()

    queries = expand_query(query, llm, n=n_variants)                  # 1. expansion
    t_exp = time.perf_counter() - t0

    t1 = time.perf_counter()
    retriever = db.as_retriever(search_kwargs={"k": k_search})
    candidates = merge_results([retriever.invoke(q) for q in queries], max_candidates)
    t_search = time.perf_counter() - t1                               # 2. multi-recherche

    t2 = time.perf_counter()
    selected = rerank_chunks(query, candidates, llm, top_k=top_k)     # 3. reranking
    t_rerank = time.perf_counter() - t2

    t3 = time.perf_counter()
    answer = generate_answer(query, selected, llm)                    # 4. génération
    t_gen = time.perf_counter() - t3

    kept = {id(d) for d in selected}
    lines = ["**Requêtes utilisées :**"] + [f"- {q}" for q in queries]
    lines += ["", "**Chunks candidats et notes de reranking :**"]
    ranked = sorted(
        candidates, key=lambda d: d.metadata.get("rerank_score", 0), reverse=True
    )
    for doc in ranked:
        page = doc.metadata.get("page")
        where = f"page {page + 1}" if isinstance(page, int) else "page ?"
        excerpt = doc.page_content[:90].replace("\n", " ")
        status = "retenu" if id(doc) in kept else "écarté"
        score = doc.metadata.get("rerank_score", 0)
        lines.append(f"- {where} · note {score:.0f}/10 · {status} · « {excerpt}… »")
    lines += [
        "",
        f"Expansion : {t_exp:.1f} s · Recherche : {t_search:.1f} s · "
        f"Reranking : {t_rerank:.1f} s · Génération : {t_gen:.1f} s",
    ]

    return RAGResult(
        title="RAG amélioré (Query Expansion + Reranking)",
        answer=answer,
        sources=format_sources(selected),
        elapsed=time.perf_counter() - t0,
        details="\n".join(lines),
    )
