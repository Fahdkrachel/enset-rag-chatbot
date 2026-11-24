"""Tests unitaires de la Query Expansion, de la fusion et du Reranking.

Aucun appel à Ollama : on utilise un faux LLM qui renvoie des réponses prévues.
"""
from dataclasses import dataclass, field

import pytest

from rag.retrieval import expand_query, merge_results, parse_score, rerank_chunks


@dataclass
class Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeLLM:
    """Renvoie les réponses dans l'ordre ; lève une erreur si `fail=True`."""

    def __init__(self, outputs=(), fail=False):
        self.outputs = list(outputs)
        self.fail = fail
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("Ollama indisponible")
        return self.outputs.pop(0)


def doc(text, page=0):
    return Doc(text, {"source": "x.pdf", "page": page})


# ---------- expand_query ----------
def test_expand_query_nettoie_la_sortie_du_llm():
    llm = FakeLLM([
        "Voici 3 questions :\n"
        "1. En quelle année l'ENSET a-t-elle été fondée ?\n"
        "- Quelle est la date de création de l'école ?\n"
        "* Depuis quand l'ENSET existe-t-elle ?\n"
        "4) Une quatrième question en trop ?"
    ])
    result = expand_query("Quand l'école a-t-elle été créée ?", llm, n=3)
    assert result[0] == "Quand l'école a-t-elle été créée ?"
    assert len(result) == 4
    assert not any("Voici" in q for q in result)
    assert all(not q.startswith(("1.", "-", "*")) for q in result)


def test_expand_query_ignore_la_question_d_origine_repetee():
    llm = FakeLLM(["Quelle est l'adresse ?\nOù se trouve l'école exactement ?"])
    result = expand_query("Quelle est l'adresse ?", llm)
    assert result == ["Quelle est l'adresse ?", "Où se trouve l'école exactement ?"]


def test_expand_query_repli_si_le_llm_echoue():
    assert expand_query("Une question ?", FakeLLM(fail=True)) == ["Une question ?"]


# ---------- merge_results ----------
def test_merge_results_melange_dedoublonne_et_limite():
    a = [doc("A"), doc("B"), doc("C", 1)]
    b = [doc("A"), doc("D", 1), doc("E", 1)]
    merged = merge_results([a, b], max_docs=4)
    assert [d.page_content for d in merged] == ["A", "B", "D", "C"]


def test_merge_results_listes_de_tailles_differentes():
    merged = merge_results([[doc("A")], [doc("B"), doc("C")]], max_docs=10)
    assert [d.page_content for d in merged] == ["A", "B", "C"]


# ---------- parse_score ----------
@pytest.mark.parametrize(
    "raw, expected",
    [("7", 7.0), ("Note : 9", 9.0), ("10/10", 10.0), ("8,5", 8.5),
     ("15", 10.0), ("aucun nombre", 0.0), ("", 0.0)],
)
def test_parse_score(raw, expected):
    assert parse_score(raw) == expected


# ---------- rerank_chunks ----------
def test_rerank_garde_les_meilleurs_et_stocke_la_note():
    docs = [doc("A"), doc("B"), doc("C", 1), doc("D", 1)]
    llm = FakeLLM(["3", "Note : 9", "10/10", "pas de nombre"])
    top = rerank_chunks("question", docs, llm, top_k=2)
    assert [d.page_content for d in top] == ["C", "B"]
    assert top[0].metadata["rerank_score"] == 10.0
    assert docs[3].metadata["rerank_score"] == 0.0


def test_rerank_tri_stable_en_cas_d_egalite():
    docs = [doc("A"), doc("B"), doc("C")]
    top = rerank_chunks("q", docs, FakeLLM(["5", "5", "5"]), top_k=2)
    assert [d.page_content for d in top] == ["A", "B"]


def test_rerank_note_zero_si_le_llm_echoue():
    docs = [doc("A")]
    top = rerank_chunks("q", docs, FakeLLM(fail=True), top_k=5)
    assert top[0].metadata["rerank_score"] == 0.0
