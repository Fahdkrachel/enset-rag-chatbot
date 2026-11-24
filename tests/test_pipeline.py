"""Tests des pipelines classique et amélioré avec une base vectorielle simulée."""
from dataclasses import dataclass, field

from rag.pipeline import ANSWER_PROMPT, format_sources, run_advanced, run_classic


@dataclass
class Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeRetriever:
    def __init__(self, table):
        self.table = table

    def invoke(self, query):
        return list(self.table.get(query, []))


class FakeDB:
    """Imite `db.as_retriever(search_kwargs=...)` d'un vector store."""

    def __init__(self, table):
        self.table = table
        self.last_kwargs = None

    def as_retriever(self, search_kwargs=None):
        self.last_kwargs = search_kwargs
        return FakeRetriever(self.table)


class ScriptedLLM:
    """Répond selon le type de prompt (expansion, note, réponse)."""

    def __init__(self):
        self.calls = {"expansion": 0, "note": 0, "reponse": 0}

    def invoke(self, prompt):
        if prompt.startswith("Génère"):
            self.calls["expansion"] += 1
            return "En quelle année l'école a-t-elle été fondée ?"
        if prompt.startswith("Tu évalues"):
            self.calls["note"] += 1
            return "9" if "1985" in prompt else "1"
        self.calls["reponse"] += 1
        return "  Fondée en 1985.  "


def make_db():
    fondation = Doc("L'école a été fondée en septembre 1985.", {"source": "x.pdf", "page": 0})
    adresse = Doc("Adresse : BP 159, Mohammedia.", {"source": "x.pdf", "page": 1})
    return FakeDB({
        "Quand l'école a-t-elle été créée ?": [adresse],
        "En quelle année l'école a-t-elle été fondée ?": [fondation, adresse],
    })


def test_format_sources_page_commence_a_1_sans_doublons():
    docs = [Doc("a", {"source": "x.pdf", "page": 0}), Doc("b", {"source": "x.pdf", "page": 0}),
            Doc("c", {"source": "x.pdf", "page": 1})]
    assert format_sources(docs) == ["x.pdf (page 1)", "x.pdf (page 2)"]


def test_run_classic():
    db, llm = make_db(), ScriptedLLM()
    res = run_classic("Quand l'école a-t-elle été créée ?", db, llm, top_k=3)
    assert db.last_kwargs == {"k": 3}
    assert res.answer == "Fondée en 1985."           # espaces retirés
    assert res.sources == ["x.pdf (page 2)"]
    assert llm.calls == {"expansion": 0, "note": 0, "reponse": 1}


def test_run_advanced_expansion_puis_reranking():
    db, llm = make_db(), ScriptedLLM()
    res = run_advanced("Quand l'école a-t-elle été créée ?", db, llm, top_k=1)
    # 1 expansion, 2 candidats notés (dédoublonnés), 1 réponse
    assert llm.calls == {"expansion": 1, "note": 2, "reponse": 1}
    assert res.sources == ["x.pdf (page 1)"]          # le chunk « 1985 » est retenu
    assert "retenu" in res.details and "écarté" in res.details
    assert "En quelle année l'école a-t-elle été fondée ?" in res.details


def test_le_prompt_impose_le_refus_hors_document():
    assert "Je ne sais pas d'après le document fourni." in ANSWER_PROMPT
