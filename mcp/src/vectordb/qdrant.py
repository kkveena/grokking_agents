import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
import cohere
from src.vectordb.base import BaseVectorDB
from src.embeddings.base import BaseEmbedding

class QdrantSearch(BaseVectorDB):
    """
    Qdrant implementation for Vector Search with Cohere Reranking.
    Follows best practices for dependency injection and environment configuration.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        collection_name: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        rerank_model: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
    ):
        """
        Initializes the QdrantSearch class.
        
        Args:
            embedding_model: An instance of a class inheriting from BaseEmbedding.
            collection_name: The name of the Qdrant collection.
            cohere_api_key: API key for Cohere (used for reranking).
            rerank_model: The Cohere rerank model to use.
            qdrant_url: URL for Qdrant Cloud.
            qdrant_api_key: API key for Qdrant Cloud.
            qdrant_host: Host for local Qdrant.
            qdrant_port: Port for local Qdrant.
        """
        self.embedding_model = embedding_model
        
        # Configuration from arguments or environment variables
        self.collection_name = collection_name or os.environ.get("QDRANT_COLLECTION", "docs")
        self.rerank_model = rerank_model or os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")
        
        # Initialize Cohere client for reranking
        # Following the pattern from CohereMultimodalEmbedding of requiring an API key
        api_key = cohere_api_key or os.environ.get("COHERE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("COHERE_API_KEY must be provided or set in environment variables")
        self.cohere_client = cohere.ClientV2(api_key=api_key)

        # Initialize Qdrant client
        # Support both Cloud and Local modes based on environment or parameters
        q_mode = os.environ.get("QDRANT_MODE", "local").lower()
        q_url = qdrant_url or os.environ.get("QDRANT_URL", "").strip()
        q_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY", "").strip()

        if q_mode == "cloud" or (q_url and q_api_key):
            self.qdrant_client = QdrantClient(
                url=q_url,
                api_key=q_api_key,
            )
        else:
            host = qdrant_host or os.environ.get("QDRANT_HOST", "qdrant")
            port = int(qdrant_port or os.environ.get("QDRANT_PORT", "6333"))
            self.qdrant_client = QdrantClient(host=host, port=port)

    def search(self, query: str, limit: int = 10, rerank_top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Single search method that:
        1. Generates embedding using the custom embedding class.
        2. Queries Qdrant using dense vector search.
        3. Reranks results using Cohere.
        
        Args:
            query: The natural-language search query.
            limit: Number of candidates to retrieve from Qdrant before reranking.
            rerank_top_n: Number of final results to return after reranking.
            
        Returns:
            List of dictionaries with 'text', 'score', and 'metadata'.
        """
        # 1. Generate embedding (using the query-specific method if available)
        if hasattr(self.embedding_model, "embed_text_query"):
            query_vector = self.embedding_model.embed_text_query(query)
        else:
            query_vector = self.embedding_model.embed_text(query)

        # 2. Dense search in Qdrant
        search_response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        
        candidates = search_response.points
        if not candidates:
            return []

        # 3. Rerank using Cohere
        documents = [c.payload.get("text", "") for c in candidates if c.payload]
        
        rerank_results = self.cohere_client.rerank(
            model=self.rerank_model,
            query=query,
            documents=documents,
            top_n=rerank_top_n,
        )

        # 4. Format final results
        final_results = []
        for r in rerank_results.results:
            # Match rerank index back to original candidate
            candidate = candidates[r.index]
            final_results.append({
                "text": candidate.payload.get("text", ""),
                "score": r.relevance_score,
                "metadata": candidate.payload
            })

        return final_results
