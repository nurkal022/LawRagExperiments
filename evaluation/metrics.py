"""
Retrieval metrics for RAG evaluation.
Implements Recall@k, MRR, NDCG@k, Precision@k.
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    precision_at_k: float
    k: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            f'recall@{self.k}': self.recall_at_k,
            'mrr': self.mrr,
            f'ndcg@{self.k}': self.ndcg_at_k,
            f'precision@{self.k}': self.precision_at_k
        }


def recall_at_k(
    retrieved: List[int],
    relevant: Set[int],
    k: int
) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = |retrieved@k ∩ relevant| / |relevant|
    
    Args:
        retrieved: List of retrieved chunk indices (ordered by rank)
        relevant: Set of relevant chunk indices (gold passages)
        k: Cutoff for evaluation
        
    Returns:
        Recall@k score (0-1)
    """
    if not relevant:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & relevant)
    
    return hits / len(relevant)


def precision_at_k(
    retrieved: List[int],
    relevant: Set[int],
    k: int
) -> float:
    """
    Calculate Precision@k.
    
    Precision@k = |retrieved@k ∩ relevant| / k
    
    Args:
        retrieved: List of retrieved chunk indices (ordered by rank)
        relevant: Set of relevant chunk indices (gold passages)
        k: Cutoff for evaluation
        
    Returns:
        Precision@k score (0-1)
    """
    if k == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & relevant)
    
    return hits / k


def reciprocal_rank(
    retrieved: List[int],
    relevant: Set[int]
) -> float:
    """
    Calculate Reciprocal Rank (RR).
    
    RR = 1 / rank of first relevant result
    
    Args:
        retrieved: List of retrieved chunk indices (ordered by rank)
        relevant: Set of relevant chunk indices (gold passages)
        
    Returns:
        Reciprocal Rank score (0-1)
    """
    for rank, chunk_idx in enumerate(retrieved, start=1):
        if chunk_idx in relevant:
            return 1.0 / rank
    
    return 0.0


def dcg_at_k(
    retrieved: List[int],
    relevant: Set[int],
    k: int
) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG@k).
    
    DCG@k = Σ rel_i / log2(i + 1) for i in 1..k
    
    Args:
        retrieved: List of retrieved chunk indices (ordered by rank)
        relevant: Set of relevant chunk indices (gold passages)
        k: Cutoff for evaluation
        
    Returns:
        DCG@k score
    """
    dcg = 0.0
    
    for i, chunk_idx in enumerate(retrieved[:k], start=1):
        if chunk_idx in relevant:
            # Binary relevance: rel = 1 if relevant, 0 otherwise
            dcg += 1.0 / np.log2(i + 1)
    
    return dcg


def ndcg_at_k(
    retrieved: List[int],
    relevant: Set[int],
    k: int
) -> float:
    """
    Calculate Normalized DCG@k.
    
    NDCG@k = DCG@k / IDCG@k
    
    Args:
        retrieved: List of retrieved chunk indices (ordered by rank)
        relevant: Set of relevant chunk indices (gold passages)
        k: Cutoff for evaluation
        
    Returns:
        NDCG@k score (0-1)
    """
    dcg = dcg_at_k(retrieved, relevant, k)
    
    # Ideal DCG: all relevant documents at top positions
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_k + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_retrieval_metrics(
    retrieved: List[int],
    relevant: Set[int],
    k: int
) -> RetrievalMetrics:
    """
    Calculate all retrieval metrics.
    
    Args:
        retrieved: List of retrieved chunk indices (ordered by rank)
        relevant: Set of relevant chunk indices (gold passages)
        k: Cutoff for evaluation
        
    Returns:
        RetrievalMetrics object
    """
    return RetrievalMetrics(
        recall_at_k=recall_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant),
        ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
        precision_at_k=precision_at_k(retrieved, relevant, k),
        k=k
    )


def aggregate_metrics(
    metrics_list: List[RetrievalMetrics]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple queries.
    
    Args:
        metrics_list: List of RetrievalMetrics objects
        
    Returns:
        Dictionary with mean, std, and 95% CI for each metric
    """
    if not metrics_list:
        return {}
    
    k = metrics_list[0].k
    
    # Collect values for each metric
    recall_values = [m.recall_at_k for m in metrics_list]
    mrr_values = [m.mrr for m in metrics_list]
    ndcg_values = [m.ndcg_at_k for m in metrics_list]
    precision_values = [m.precision_at_k for m in metrics_list]
    
    def compute_stats(values: List[float], name: str) -> Dict[str, float]:
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        n = len(arr)
        
        # 95% CI using t-distribution approximation
        ci_margin = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        
        return {
            'mean': float(mean),
            'std': float(std),
            'ci_lower': float(mean - ci_margin),
            'ci_upper': float(mean + ci_margin),
            'n': n
        }
    
    return {
        f'recall@{k}': compute_stats(recall_values, f'recall@{k}'),
        'mrr': compute_stats(mrr_values, 'mrr'),
        f'ndcg@{k}': compute_stats(ndcg_values, f'ndcg@{k}'),
        f'precision@{k}': compute_stats(precision_values, f'precision@{k}')
    }


def get_gold_chunk_indices(
    question: Dict,
    chunk_index: Dict[str, List[int]]
) -> Set[int]:
    """
    Get global chunk indices for gold passages.
    
    Args:
        question: Question dictionary with gold_passages
        chunk_index: Mapping from doc_id to chunk indices
        
    Returns:
        Set of global chunk indices
    """
    gold_indices = set()
    
    for passage in question.get('gold_passages', []):
        doc_id = passage['doc_id']
        chunk_id = passage['chunk_id']
        
        if doc_id in chunk_index:
            # chunk_index[doc_id] contains global indices of all chunks for this doc
            doc_chunks = chunk_index[doc_id]
            if chunk_id < len(doc_chunks):
                gold_indices.add(doc_chunks[chunk_id])
    
    return gold_indices
