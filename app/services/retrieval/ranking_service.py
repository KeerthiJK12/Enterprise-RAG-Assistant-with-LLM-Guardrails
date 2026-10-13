import time
import logfire

from flashrank import Ranker, RerankRequest


# Lazy initialization - Ranker is loaded on first use to ensure
# logfire.configure() has already run.
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.

    FlashRank uses a local ONNX cross-encoder model for
    lightweight and fast semantic reranking.
    """
    global _ranker

    if _ranker is None:
        logfire.info(
            "🧠 Initializing FlashRank Model locally..."
        )

        try:
            # Use a specific cache directory to avoid permission
            # issues in production environments.
            _ranker = Ranker(
                cache_dir="/tmp/flashrank"
            )
        except Exception:
            _ranker = Ranker()

    return _ranker


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5
) -> list[str]:
    """
    Re-ranks retrieved documents based on semantic relevance.

    Qdrant performs the initial vector search. FlashRank then
    uses a local cross-encoder model to refine the ordering.
    """
    if not documents:
        return []

    start_time = time.time()

    logfire.info(
        f"📡 [Reranker] Sending {len(documents)} docs "
        "to FlashRank Cross-Encoder..."
    )

    try:
        ranker = _get_ranker()

        # FlashRank expects passages as dictionaries containing
        # an id and the document text.
        passages = [
            {
                "id": i,
                "text": doc
            }
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(
            query=query,
            passages=passages
        )

        results = ranker.rerank(request)

        # FlashRank returns results ordered by semantic relevance.
        reranked_docs = [
            result["text"]
            for result in results[:top_n]
        ]

        duration = time.time() - start_time

        top_score = (
            results[0]["score"]
            if results
            else "N/A"
        )

        logfire.info(
            f"✅ [Reranker] Done in {duration:.2f}s. "
            f"Top semantic score: {top_score}"
        )

        return reranked_docs

    except Exception as e:
        logfire.error(
            f"❌ [Reranker] Semantic Reranking Failed: {e}"
        )

        # Fallback to the original Qdrant ordering so that
        # reranking failure does not stop the RAG pipeline.
        return documents[:top_n]