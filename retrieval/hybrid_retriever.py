"""
Hybrid retriever using Weighted Reciprocal Rank Fusion (RRF).
Combines BM25 and vector retrieval results.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR, CHUNKS_DIR, TOP_K, RRF_K_VALUES, RRF_ALPHA_VALUES
from indexing.chunker import load_chunks
from retrieval.bm25_retriever import BM25Retriever, load_bm25_retriever
from retrieval.vector_retriever import VectorRetriever, load_vector_retriever


class HybridRetriever:
    """
    Hybrid retriever using Weighted RRF.
    
    Weighted RRF formula:
        score(d) = α / (k + rank_bm25(d)) + (1-α) / (k + rank_vector(d))
    
    Where:
        - α: weight for BM25 (0 = pure vector, 1 = pure BM25)
        - k: RRF smoothing constant
    """
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        chunks: List[Dict],
        rrf_k: int = 60,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            vector_retriever: Vector retriever instance
            chunks: List of chunk dictionaries
            rrf_k: RRF smoothing constant
            alpha: Weight for BM25 (0-1)
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.alpha = alpha
    
    def set_params(self, rrf_k: int, alpha: float):
        """Update RRF parameters."""
        self.rrf_k = rrf_k
        self.alpha = alpha
    
    def _compute_rrf_score(
        self,
        bm25_rank: Optional[int],
        vector_rank: Optional[int]
    ) -> float:
        """
        Compute weighted RRF score.
        
        Args:
            bm25_rank: Rank from BM25 (1-based, None if not in results)
            vector_rank: Rank from vector search (1-based, None if not in results)
            
        Returns:
            Combined RRF score
        """
        score = 0.0
        
        if bm25_rank is not None:
            score += self.alpha / (self.rrf_k + bm25_rank)
        
        if vector_rank is not None:
            score += (1 - self.alpha) / (self.rrf_k + vector_rank)
        
        return score
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        retrieval_depth: int = 100
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve top-k relevant chunks using hybrid approach.
        
        Args:
            query: Query string
            top_k: Number of results to return
            retrieval_depth: How many results to fetch from each retriever
            
        Returns:
            List of (chunk_idx, score, chunk_dict) tuples, sorted by score desc
        """
        # Get rankings from both retrievers
        bm25_rankings = dict(self.bm25_retriever.get_rankings(query, retrieval_depth))
        vector_rankings = dict(self.vector_retriever.get_rankings(query, retrieval_depth))
        
        # Collect all unique chunk IDs
        all_chunk_ids = set(bm25_rankings.keys()) | set(vector_rankings.keys())
        
        # Compute RRF scores
        scores = []
        for chunk_idx in all_chunk_ids:
            bm25_rank = bm25_rankings.get(chunk_idx)
            vector_rank = vector_rankings.get(chunk_idx)
            rrf_score = self._compute_rrf_score(bm25_rank, vector_rank)
            scores.append((chunk_idx, rrf_score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with chunk data
        results = []
        for chunk_idx, score in scores[:top_k]:
            chunk = self.chunks[chunk_idx]
            results.append((chunk_idx, score, chunk))
        
        return results
    
    def get_rankings(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieval_depth: int = 100
    ) -> List[Tuple[int, int]]:
        """
        Get ranking positions for chunks using hybrid approach.
        
        Args:
            query: Query string
            top_k: Limit to top-k results (None for all)
            retrieval_depth: How many results to fetch from each retriever
            
        Returns:
            List of (chunk_idx, rank) tuples where rank starts from 1
        """
        # Get rankings from both retrievers
        bm25_rankings = dict(self.bm25_retriever.get_rankings(query, retrieval_depth))
        vector_rankings = dict(self.vector_retriever.get_rankings(query, retrieval_depth))
        
        # Collect all unique chunk IDs
        all_chunk_ids = set(bm25_rankings.keys()) | set(vector_rankings.keys())
        
        # Compute RRF scores
        scores = []
        for chunk_idx in all_chunk_ids:
            bm25_rank = bm25_rankings.get(chunk_idx)
            vector_rank = vector_rankings.get(chunk_idx)
            rrf_score = self._compute_rrf_score(bm25_rank, vector_rank)
            scores.append((chunk_idx, rrf_score))
        
        # Sort by score descending and assign ranks
        scores.sort(key=lambda x: x[1], reverse=True)
        
        rankings = []
        for rank, (chunk_idx, score) in enumerate(scores, start=1):
            if top_k and rank > top_k:
                break
            rankings.append((chunk_idx, rank))
        
        return rankings


def load_hybrid_retriever(
    bm25_tokenizer_type: str = 'kaznlp',
    rrf_k: int = 60,
    alpha: float = 0.5,
    index_dir: Path = INDEX_DIR,
    chunks_dir: Path = CHUNKS_DIR
) -> HybridRetriever:
    """
    Load hybrid retriever with both BM25 and vector components.
    
    Args:
        bm25_tokenizer_type: 'kaznlp' or 'whitespace'
        rrf_k: RRF smoothing constant
        alpha: Weight for BM25
        index_dir: Directory containing index files
        chunks_dir: Directory containing chunks
        
    Returns:
        HybridRetriever instance
    """
    # Load retrievers
    bm25_retriever = load_bm25_retriever(bm25_tokenizer_type, index_dir, chunks_dir)
    vector_retriever = load_vector_retriever(index_dir, chunks_dir)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        chunks=chunks,
        rrf_k=rrf_k,
        alpha=alpha
    )


# Grid search parameter combinations
def get_rrf_param_grid():
    """Get all parameter combinations for grid search."""
    params = []
    for k in RRF_K_VALUES:
        for alpha in RRF_ALPHA_VALUES:
            params.append({'rrf_k': k, 'alpha': alpha})
    return params
