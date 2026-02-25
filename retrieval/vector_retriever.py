"""
Vector (semantic) retriever for Kazakh legal corpus.
Uses FAISS index with OpenAI embeddings.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR, CHUNKS_DIR, TOP_K
from indexing.chunker import load_chunks
from indexing.faiss_index import FAISSIndex, load_faiss_index, EmbeddingClient


class VectorRetriever:
    """Vector-based document retriever using FAISS."""
    
    def __init__(
        self,
        index: FAISSIndex,
        chunks: List[Dict],
        embedding_client: EmbeddingClient
    ):
        """
        Initialize vector retriever.
        
        Args:
            index: FAISSIndex object
            chunks: List of chunk dictionaries
            embedding_client: Client for generating query embeddings
        """
        self.index = index
        self.chunks = chunks
        self.embedding_client = embedding_client
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk_idx, score, chunk_dict) tuples, sorted by score desc
        """
        # Generate query embedding
        query_embedding = self.embedding_client.embed_query(query)
        
        # Search index
        results = self.index.search(query_embedding, top_k)
        
        # Add chunk data
        full_results = []
        for chunk_idx, score in results:
            chunk = self.chunks[chunk_idx]
            full_results.append((chunk_idx, score, chunk))
        
        return full_results
    
    def get_rankings(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Get ranking positions for chunks.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk_idx, rank) tuples where rank starts from 1
        """
        # For efficiency, we limit the search to a reasonable top_k
        search_k = top_k if top_k else min(1000, len(self.chunks))
        
        query_embedding = self.embedding_client.embed_query(query)
        results = self.index.search(query_embedding, search_k)
        
        rankings = []
        for rank, (chunk_idx, score) in enumerate(results, start=1):
            rankings.append((chunk_idx, rank))
        
        return rankings


def load_vector_retriever(
    index_dir: Path = INDEX_DIR,
    chunks_dir: Path = CHUNKS_DIR
) -> VectorRetriever:
    """
    Load vector retriever from saved index.
    
    Args:
        index_dir: Directory containing index files
        chunks_dir: Directory containing chunks
        
    Returns:
        VectorRetriever instance
    """
    # Load index
    index = load_faiss_index(index_dir)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    
    # Create embedding client
    embedding_client = EmbeddingClient()
    
    return VectorRetriever(index, chunks, embedding_client)
