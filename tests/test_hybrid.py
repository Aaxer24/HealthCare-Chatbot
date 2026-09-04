from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag import HybridRetriever, document_identity, reciprocal_rank_fusion


class StubRetriever(BaseRetriever):
    """Returns a fixed list, so fusion behaviour can be tested without models."""

    docs: list

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.docs


class ExplodingRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        raise RuntimeError("bm25 exploded")


def doc(text: str, source: str = "a.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_document_identity_is_stable_for_same_content():
    assert document_identity(doc("same text")) == document_identity(doc("same text"))


def test_document_identity_differs_by_content_and_page():
    assert document_identity(doc("one")) != document_identity(doc("two"))
    assert document_identity(doc("same", page=1)) != document_identity(doc("same", page=2))


def test_fusion_deduplicates_documents_found_by_both_retrievers():
    shared = doc("shared chunk")
    fused = reciprocal_rank_fusion([[shared, doc("only-a")], [shared, doc("only-b")]])
    contents = [d.page_content for d in fused]
    assert contents.count("shared chunk") == 1, "duplicate chunk should appear once"
    assert len(fused) == 3


def test_fusion_ranks_documents_found_by_both_retrievers_first():
    """The core value of RRF: agreement between retrievers outranks a single
    retriever's top hit."""
    both = doc("found by both")
    fused = reciprocal_rank_fusion(
        [
            [doc("semantic top"), both],
            [doc("keyword top"), both],
        ]
    )
    assert fused[0].page_content == "found by both"


def test_fusion_respects_top_n():
    docs = [doc(f"chunk {i}") for i in range(10)]
    assert len(reciprocal_rank_fusion([docs], top_n=4)) == 4


def test_fusion_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_hybrid_retriever_merges_both_sources():
    hybrid = HybridRetriever(
        semantic_retriever=StubRetriever(docs=[doc("semantic hit")]),
        keyword_retriever=StubRetriever(docs=[doc("keyword hit")]),
    )
    contents = [d.page_content for d in hybrid.invoke("query")]
    assert "semantic hit" in contents
    assert "keyword hit" in contents


def test_hybrid_retriever_survives_keyword_failure():
    """BM25 must never be able to break retrieval outright -- semantic results
    alone are still a usable answer."""
    hybrid = HybridRetriever(
        semantic_retriever=StubRetriever(docs=[doc("semantic hit")]),
        keyword_retriever=ExplodingRetriever(),
    )
    contents = [d.page_content for d in hybrid.invoke("query")]
    assert contents == ["semantic hit"]
