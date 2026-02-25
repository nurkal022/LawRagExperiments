"""
BM25 retriever for Kazakh legal corpus.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDEX_DIR, CHUNKS_DIR, TOP_K
from indexing.chunker import load_chunks
from indexing.bm25_index import BM25Index, load_bm25_index, KazakhTokenizer, WhitespaceTokenizer


class BM25Retriever:
    """BM25-based document retriever."""
    
    def __init__(
        self,
        index: BM25Index,
        chunks: List[Dict],
        tokenizer
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            index: BM25Index object
            chunks: List of chunk dictionaries
            tokenizer: Tokenizer instance (KazakhTokenizer or WhitespaceTokenizer)
        """
        self.index = index
        self.chunks = chunks
        self.tokenizer = tokenizer
    
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
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.index.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_idx = self.index.chunk_ids[idx]
            score = float(scores[idx])
            chunk = self.chunks[chunk_idx]
            results.append((chunk_idx, score, chunk))
        
        return results
    
    def get_rankings(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Get ranking positions for all chunks.
        
        Args:
            query: Query string
            top_k: Limit to top-k results (None for all)
            
        Returns:
            List of (chunk_idx, rank) tuples where rank starts from 1
        """
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            return []
        
        scores = self.index.bm25.get_scores(query_tokens)
        sorted_indices = scores.argsort()[::-1]
        
        rankings = []
        for rank, idx in enumerate(sorted_indices, start=1):
            if top_k and rank > top_k:
                break
            chunk_idx = self.index.chunk_ids[idx]
            rankings.append((chunk_idx, rank))
        
        return rankings


def load_bm25_retriever(
    tokenizer_type: str = 'kaznlp',
    index_dir: Path = INDEX_DIR,
    chunks_dir: Path = CHUNKS_DIR
) -> BM25Retriever:
    """
    Load BM25 retriever from saved index.
    
    Args:
        tokenizer_type: 'kaznlp' or 'whitespace'
        index_dir: Directory containing index files
        chunks_dir: Directory containing chunks
        
    Returns:
        BM25Retriever instance
    """
    # Load index
    index_file = index_dir / f"bm25_{tokenizer_type}.pkl"
    index = load_bm25_index(index_file)
    
    # Load chunks
    chunks = load_chunks(chunks_dir)
    
    # Create fresh tokenizer (avoid pickle issues)
    if tokenizer_type == 'kaznlp':
        tokenizer = KazakhTokenizer(use_kaznlp=True)
    else:
        tokenizer = WhitespaceTokenizer()
    
    return BM25Retriever(index, chunks, tokenizer)
