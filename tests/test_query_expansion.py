import src.query_expansion as qe
import src.rag as rag
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from tests.conftest import make_config


class StubRetriever(BaseRetriever):
    """Records every query it was asked for, so fan-out can be asserted."""

    seen: list

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        self.seen.append(query)
        return [Document(page_content=f"doc for {query}", metadata={"source": "a.pdf", "page": 1})]


def test_clean_query_strips_quotes_bullets_and_numbering():
    assert qe._clean_query('  "diabetes symptoms"  ') == "diabetes symptoms"
    assert qe._clean_query("- diabetes symptoms") == "diabetes symptoms"
    assert qe._clean_query("1. diabetes symptoms") == "diabetes symptoms"


def test_rewrite_query_returns_rewritten_text(monkeypatch):
    monkeypatch.setattr(rag, "complete_text", lambda *a, **k: "diabetes mellitus symptoms")
    assert qe.rewrite_query("sugar ki problem", make_config()) == "diabetes mellitus symptoms"


def test_rewrite_query_falls_back_to_original_on_failure(monkeypatch):
    """A degraded query must beat a failed request."""
    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(rag, "complete_text", boom)
    assert qe.rewrite_query("sugar ki problem", make_config()) == "sugar ki problem"


def test_rewrite_query_falls_back_when_llm_returns_blank(monkeypatch):
    monkeypatch.setattr(rag, "complete_text", lambda *a, **k: "   ")
    assert qe.rewrite_query("chest pain", make_config()) == "chest pain"


def test_generate_query_variations_parses_lines(monkeypatch):
    monkeypatch.setattr(
        rag, "complete_text",
        lambda *a, **k: "causes of anemia\n- treatment for anemia\n2. anemia prevention",
    )
    variations = qe.generate_query_variations("anemia", make_config(), n=3)
    assert variations == ["causes of anemia", "treatment for anemia", "anemia prevention"]


def test_generate_query_variations_drops_duplicates_of_the_original(monkeypatch):
    monkeypatch.setattr(rag, "complete_text", lambda *a, **k: "anemia\ncauses of anemia")
    assert qe.generate_query_variations("anemia", make_config(), n=3) == ["causes of anemia"]


def test_generate_query_variations_respects_n(monkeypatch):
    monkeypatch.setattr(rag, "complete_text", lambda *a, **k: "one\ntwo\nthree\nfour")
    assert len(qe.generate_query_variations("q", make_config(), n=2)) == 2


def test_generate_query_variations_returns_empty_on_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(rag, "complete_text", boom)
    assert qe.generate_query_variations("q", make_config(), n=2) == []


def test_expanding_retriever_always_includes_the_original_query(monkeypatch):
    """A bad rewrite must never be able to discard what the user actually asked."""
    monkeypatch.setattr(rag, "complete_text", lambda *a, **k: "rewritten form")

    inner = StubRetriever(seen=[])
    retriever = rag.QueryExpandingRetriever(
        inner_retriever=inner, config=make_config(), rewrite=True, variation_count=0
    )
    retriever.invoke("original question")

    assert "original question" in inner.seen
    assert "rewritten form" in inner.seen


def test_expanding_retriever_skips_fanout_when_disabled():
    inner = StubRetriever(seen=[])
    retriever = rag.QueryExpandingRetriever(
        inner_retriever=inner, config=make_config(), rewrite=False, variation_count=0
    )
    retriever.invoke("just this")

    assert inner.seen == ["just this"], "no LLM calls or extra retrievals should happen"


def test_expanding_retriever_fans_out_to_variations(monkeypatch):
    monkeypatch.setattr(qe, "rewrite_query", lambda q, c: q)
    monkeypatch.setattr(qe, "generate_query_variations", lambda q, c, n: ["alt one", "alt two"])

    inner = StubRetriever(seen=[])
    retriever = rag.QueryExpandingRetriever(
        inner_retriever=inner, config=make_config(), rewrite=False, variation_count=2
    )
    docs = retriever.invoke("base query")

    assert set(inner.seen) == {"base query", "alt one", "alt two"}
    assert len(docs) == 3, "results from every query should be fused"
