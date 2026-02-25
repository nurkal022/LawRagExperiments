"""
5-fold cross-validation for experiment evaluation.
Implements proper train/test splits with nested grid search.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.model_selection import KFold
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    N_FOLDS, TOP_K, QUESTIONS_FILE, INDEX_DIR,
    LLM_MODELS, LLMModel
)
from indexing.chunker import load_chunk_index
from evaluation.metrics import (
    calculate_retrieval_metrics,
    aggregate_metrics,
    get_gold_chunk_indices,
    RetrievalMetrics
)
from evaluation.judge import LLMJudge, JudgeResult, compute_answer_accuracy
from experiment.grid_search import grid_search, GridSearchResult


@dataclass
class FoldResult:
    """Results from a single fold."""
    fold_idx: int
    train_indices: List[int]
    test_indices: List[int]
    
    # Grid search results (for hybrid)
    grid_search_result: Optional[Dict]
    
    # Retrieval metrics per method
    retrieval_metrics: Dict[str, Dict[str, float]]
    
    # Answer accuracy per model per method
    answer_accuracy: Dict[str, Dict[str, Dict[str, float]]]


@dataclass  
class CVResult:
    """Aggregated cross-validation results."""
    n_folds: int
    n_questions: int
    
    # Fold-level results
    fold_results: List[FoldResult]
    
    # Aggregated retrieval metrics per method
    aggregated_retrieval: Dict[str, Dict[str, Dict[str, float]]]
    
    # Aggregated answer accuracy per model per method
    aggregated_accuracy: Dict[str, Dict[str, Dict[str, float]]]


def load_questions(questions_file: Path = QUESTIONS_FILE) -> List[Dict]:
    """Load questions from JSON file."""
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get('questions', [])
    # Filter out example/placeholder questions
    questions = [q for q in questions if not q.get('id', '').startswith('EXAMPLE')]
    
    return questions


def prepare_gold_indices(
    questions: List[Dict],
    chunk_index: Dict[str, List[int]]
) -> List[Set[int]]:
    """
    Prepare gold passage indices for all questions.
    
    Args:
        questions: List of question dictionaries
        chunk_index: Mapping from doc_id to chunk indices
        
    Returns:
        List of gold index sets (one per question)
    """
    gold_indices = []
    
    for question in questions:
        gold = get_gold_chunk_indices(question, chunk_index)
        gold_indices.append(gold)
    
    return gold_indices


class CrossValidator:
    """5-fold cross-validation executor."""
    
    def __init__(
        self,
        questions: List[Dict],
        gold_indices: List[Set[int]],
        n_folds: int = N_FOLDS,
        random_state: int = 42
    ):
        """
        Initialize cross-validator.
        
        Args:
            questions: List of question dictionaries
            gold_indices: Gold passage indices per question
            n_folds: Number of folds
            random_state: Random seed for reproducibility
        """
        self.questions = questions
        self.gold_indices = gold_indices
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Create fold splits
        self.kfold = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )
        
        self.splits = list(self.kfold.split(questions))
    
    def get_fold_data(
        self,
        fold_idx: int
    ) -> Tuple[List[Dict], List[Dict], List[Set[int]], List[Set[int]]]:
        """
        Get train/test data for a specific fold.
        
        Returns:
            Tuple of (train_questions, test_questions, train_gold, test_gold)
        """
        train_indices, test_indices = self.splits[fold_idx]
        
        train_questions = [self.questions[i] for i in train_indices]
        test_questions = [self.questions[i] for i in test_indices]
        train_gold = [self.gold_indices[i] for i in train_indices]
        test_gold = [self.gold_indices[i] for i in test_indices]
        
        return train_questions, test_questions, train_gold, test_gold
    
    def run_fold(
        self,
        fold_idx: int,
        bm25_retriever,
        vector_retriever,
        hybrid_retriever,
        rag_pipeline,
        llm_models: List[LLMModel],
        judge: LLMJudge,
        run_generation: bool = True,
        show_progress: bool = True
    ) -> FoldResult:
        """
        Run evaluation on a single fold.
        
        Args:
            fold_idx: Fold index
            bm25_retriever: BM25 retriever
            vector_retriever: Vector retriever
            hybrid_retriever: Hybrid retriever
            rag_pipeline: RAG pipeline instance
            llm_models: List of models to evaluate
            judge: LLM judge instance
            run_generation: Whether to run LLM generation (expensive)
            show_progress: Show progress bars
            
        Returns:
            FoldResult object
        """
        train_questions, test_questions, train_gold, test_gold = self.get_fold_data(fold_idx)
        train_indices, test_indices = self.splits[fold_idx]
        
        print(f"\n=== Fold {fold_idx + 1}/{self.n_folds} ===")
        print(f"Train: {len(train_questions)} questions, Test: {len(test_questions)} questions")
        
        # 1. Grid search for hybrid on train set
        print("\nRunning grid search for hybrid retriever...")
        gs_result = grid_search(
            hybrid_retriever,
            train_questions,
            train_gold,
            optimization_metric=f'ndcg@{TOP_K}',
            show_progress=show_progress
        )
        print(f"Best params: k={gs_result.best_k}, α={gs_result.best_alpha}, "
              f"NDCG@{TOP_K}={gs_result.best_score:.4f}")
        
        # Apply best params
        hybrid_retriever.set_params(gs_result.best_k, gs_result.best_alpha)
        
        # 2. Evaluate retrieval on test set
        print("\nEvaluating retrieval on test set...")
        retrieval_metrics = {}
        
        for method_name, retriever in [
            ('bm25', bm25_retriever),
            ('vector', vector_retriever),
            ('hybrid', hybrid_retriever)
        ]:
            method_metrics = []
            
            for question, gold in zip(test_questions, test_gold):
                if method_name == 'bm25':
                    results = retriever.retrieve(question['question'], TOP_K)
                elif method_name == 'vector':
                    results = retriever.retrieve(question['question'], TOP_K)
                else:  # hybrid
                    results = retriever.retrieve(question['question'], TOP_K)
                
                retrieved = [r[0] for r in results]  # chunk indices
                metrics = calculate_retrieval_metrics(retrieved, gold, TOP_K)
                method_metrics.append(metrics)
            
            retrieval_metrics[method_name] = aggregate_metrics(method_metrics)
        
        # 3. Run LLM generation and judge evaluation (if enabled)
        answer_accuracy = {}
        
        if run_generation:
            print("\nRunning LLM generation and evaluation...")
            
            for model in llm_models:
                print(f"  Model: {model.name}")
                answer_accuracy[model.name] = {}
                
                for method_name in ['bm25', 'vector', 'hybrid']:
                    judge_results = []
                    
                    iterator = test_questions
                    if show_progress:
                        iterator = tqdm(
                            test_questions,
                            desc=f"    {method_name}",
                            leave=False
                        )
                    
                    for question in iterator:
                        # Run RAG pipeline
                        rag_result = rag_pipeline.run(
                            question['question'],
                            model,
                            method_name
                        )
                        
                        # Judge evaluation
                        judge_result = judge.evaluate(
                            question_id=question['id'],
                            question=question['question'],
                            gold_answer=question['gold_answer'],
                            generated_answer=rag_result.answer,
                            model_name=model.name,
                            retrieval_method=method_name
                        )
                        judge_results.append(judge_result)
                    
                    # Compute accuracy
                    accuracy = compute_answer_accuracy(judge_results)
                    answer_accuracy[model.name][method_name] = accuracy
        
        return FoldResult(
            fold_idx=fold_idx,
            train_indices=list(train_indices),
            test_indices=list(test_indices),
            grid_search_result={
                'best_k': gs_result.best_k,
                'best_alpha': gs_result.best_alpha,
                'best_score': gs_result.best_score
            },
            retrieval_metrics=retrieval_metrics,
            answer_accuracy=answer_accuracy
        )
    
    def aggregate_results(
        self,
        fold_results: List[FoldResult]
    ) -> CVResult:
        """
        Aggregate results across all folds.
        
        Args:
            fold_results: List of FoldResult objects
            
        Returns:
            CVResult with aggregated statistics
        """
        # Aggregate retrieval metrics
        aggregated_retrieval = {}
        methods = fold_results[0].retrieval_metrics.keys()
        
        for method in methods:
            aggregated_retrieval[method] = {}
            
            # Get all metric names
            metric_names = fold_results[0].retrieval_metrics[method].keys()
            
            for metric_name in metric_names:
                values = [
                    fr.retrieval_metrics[method][metric_name]['mean']
                    for fr in fold_results
                ]
                
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                ci_margin = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0
                
                aggregated_retrieval[method][metric_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'ci_lower': float(mean_val - ci_margin),
                    'ci_upper': float(mean_val + ci_margin)
                }
        
        # Aggregate answer accuracy
        aggregated_accuracy = {}
        
        if fold_results[0].answer_accuracy:
            models = fold_results[0].answer_accuracy.keys()
            
            for model in models:
                aggregated_accuracy[model] = {}
                
                for method in methods:
                    values = [
                        fr.answer_accuracy[model][method]['accuracy']
                        for fr in fold_results
                        if model in fr.answer_accuracy
                    ]
                    
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                        ci_margin = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0
                        
                        aggregated_accuracy[model][method] = {
                            'mean': float(mean_val),
                            'std': float(std_val),
                            'ci_lower': float(mean_val - ci_margin),
                            'ci_upper': float(mean_val + ci_margin)
                        }
        
        return CVResult(
            n_folds=self.n_folds,
            n_questions=len(self.questions),
            fold_results=fold_results,
            aggregated_retrieval=aggregated_retrieval,
            aggregated_accuracy=aggregated_accuracy
        )
