"""
Grid search for Weighted RRF hyperparameters.
Finds optimal k and alpha values using retrieval metrics.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RRF_K_VALUES, RRF_ALPHA_VALUES, TOP_K
from retrieval.hybrid_retriever import HybridRetriever, get_rrf_param_grid
from evaluation.metrics import (
    calculate_retrieval_metrics,
    aggregate_metrics,
    RetrievalMetrics
)


@dataclass
class GridSearchResult:
    """Result from grid search."""
    best_k: int
    best_alpha: float
    best_score: float
    best_metric: str
    all_results: Dict[Tuple[int, float], Dict[str, float]]


def evaluate_params(
    hybrid_retriever: HybridRetriever,
    questions: List[Dict],
    gold_indices: List[Set[int]],
    rrf_k: int,
    alpha: float,
    top_k: int = TOP_K
) -> Dict[str, float]:
    """
    Evaluate a specific parameter combination.
    
    Args:
        hybrid_retriever: Hybrid retriever instance
        questions: List of question dictionaries
        gold_indices: List of gold passage index sets (one per question)
        rrf_k: RRF smoothing constant
        alpha: BM25 weight
        top_k: Retrieval cutoff
        
    Returns:
        Dictionary with mean metrics
    """
    # Set parameters
    hybrid_retriever.set_params(rrf_k, alpha)
    
    # Evaluate on all questions
    metrics_list = []
    
    for question, gold in zip(questions, gold_indices):
        # Get rankings
        rankings = hybrid_retriever.get_rankings(
            question['question'],
            top_k=top_k
        )
        
        # Extract chunk indices
        retrieved = [chunk_idx for chunk_idx, rank in rankings]
        
        # Calculate metrics
        metrics = calculate_retrieval_metrics(retrieved, gold, top_k)
        metrics_list.append(metrics)
    
    # Aggregate
    aggregated = aggregate_metrics(metrics_list)
    
    return {
        name: stats['mean']
        for name, stats in aggregated.items()
    }


def grid_search(
    hybrid_retriever: HybridRetriever,
    train_questions: List[Dict],
    train_gold_indices: List[Set[int]],
    optimization_metric: str = 'ndcg@6',
    top_k: int = TOP_K,
    show_progress: bool = True
) -> GridSearchResult:
    """
    Perform grid search over RRF parameters.
    
    Args:
        hybrid_retriever: Hybrid retriever instance
        train_questions: Training questions
        train_gold_indices: Gold passage indices for training questions
        optimization_metric: Metric to optimize ('ndcg@6', 'recall@6', etc.)
        top_k: Retrieval cutoff
        show_progress: Show progress bar
        
    Returns:
        GridSearchResult with best parameters
    """
    param_grid = get_rrf_param_grid()
    all_results = {}
    
    best_score = -1
    best_k = None
    best_alpha = None
    
    iterator = param_grid
    if show_progress:
        iterator = tqdm(param_grid, desc="Grid search")
    
    for params in iterator:
        rrf_k = params['rrf_k']
        alpha = params['alpha']
        
        # Evaluate this combination
        metrics = evaluate_params(
            hybrid_retriever,
            train_questions,
            train_gold_indices,
            rrf_k,
            alpha,
            top_k
        )
        
        all_results[(rrf_k, alpha)] = metrics
        
        # Check if this is the best
        score = metrics.get(optimization_metric, 0)
        if score > best_score:
            best_score = score
            best_k = rrf_k
            best_alpha = alpha
    
    return GridSearchResult(
        best_k=best_k,
        best_alpha=best_alpha,
        best_score=best_score,
        best_metric=optimization_metric,
        all_results=all_results
    )


def format_grid_search_results(result: GridSearchResult) -> str:
    """Format grid search results as a string table."""
    lines = []
    lines.append(f"Best parameters: k={result.best_k}, α={result.best_alpha}")
    lines.append(f"Best {result.best_metric}: {result.best_score:.4f}")
    lines.append("")
    lines.append("All results:")
    lines.append("-" * 60)
    
    # Header
    header = f"{'k':>6} | {'alpha':>6} | "
    metrics = list(list(result.all_results.values())[0].keys())
    header += " | ".join(f"{m:>10}" for m in metrics)
    lines.append(header)
    lines.append("-" * 60)
    
    # Sort by best metric descending
    sorted_results = sorted(
        result.all_results.items(),
        key=lambda x: x[1].get(result.best_metric, 0),
        reverse=True
    )
    
    for (k, alpha), metrics_dict in sorted_results:
        row = f"{k:>6} | {alpha:>6.2f} | "
        row += " | ".join(f"{v:>10.4f}" for v in metrics_dict.values())
        lines.append(row)
    
    return "\n".join(lines)
