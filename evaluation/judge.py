"""
LLM-as-judge evaluation for answer quality.
Uses GPT-4o to evaluate generated answers against gold answers.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JUDGE_PROMPT, JUDGE_SCORES, JUDGE_MODEL
from generation.llm_client import LLMClient, LLMResponse, get_judge_model


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    question_id: str
    question: str
    gold_answer: str
    generated_answer: str
    verdict: str  # CORRECT, PARTIALLY_CORRECT, INCORRECT
    score: float  # 1.0, 0.5, 0.0
    raw_response: str
    model_name: str
    retrieval_method: str


class LLMJudge:
    """LLM-as-judge for answer evaluation."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize judge.
        
        Args:
            llm_client: LLM client for API calls (uses shared client if None)
        """
        self.llm_client = llm_client or LLMClient()
        self.judge_model = get_judge_model()
    
    def _parse_verdict(self, response: str) -> Tuple[str, float]:
        """
        Parse verdict from judge response.
        
        Args:
            response: Raw judge response
            
        Returns:
            Tuple of (verdict, score)
        """
        # Clean response
        response = response.strip().upper()
        
        # Try to extract verdict
        for verdict, score in JUDGE_SCORES.items():
            if verdict in response:
                return verdict, score
        
        # If no clear verdict, try to infer from response
        if 'PARTIAL' in response:
            return 'PARTIALLY_CORRECT', 0.5
        elif 'CORRECT' in response or 'ПРАВИЛЬН' in response:
            return 'CORRECT', 1.0
        elif 'INCORRECT' in response or 'НЕПРАВИЛЬН' in response or 'WRONG' in response:
            return 'INCORRECT', 0.0
        
        # Default to incorrect if unparseable
        return 'INCORRECT', 0.0
    
    def evaluate(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
        generated_answer: str,
        model_name: str,
        retrieval_method: str
    ) -> JudgeResult:
        """
        Evaluate a single answer.
        
        Args:
            question_id: Question identifier
            question: Original question text
            gold_answer: Reference answer
            generated_answer: Generated answer to evaluate
            model_name: Name of model that generated the answer
            retrieval_method: Retrieval method used
            
        Returns:
            JudgeResult object
        """
        # Format prompt
        prompt = JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )
        
        # Call judge model
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(self.judge_model, messages, max_tokens=50)
        
        # Parse verdict
        verdict, score = self._parse_verdict(response.content)
        
        return JudgeResult(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            verdict=verdict,
            score=score,
            raw_response=response.content,
            model_name=model_name,
            retrieval_method=retrieval_method
        )
    
    def evaluate_batch(
        self,
        results: List[Dict],
        questions: Dict[str, Dict]
    ) -> List[JudgeResult]:
        """
        Evaluate a batch of answers.
        
        Args:
            results: List of RAG results with 'question_id', 'answer', 'model_name', 'retrieval_method'
            questions: Dictionary mapping question_id to question data with 'question' and 'gold_answer'
            
        Returns:
            List of JudgeResult objects
        """
        judge_results = []
        
        for result in results:
            question_id = result['question_id']
            question_data = questions.get(question_id, {})
            
            judge_result = self.evaluate(
                question_id=question_id,
                question=question_data.get('question', ''),
                gold_answer=question_data.get('gold_answer', ''),
                generated_answer=result['answer'],
                model_name=result['model_name'],
                retrieval_method=result['retrieval_method']
            )
            
            judge_results.append(judge_result)
        
        return judge_results


def compute_answer_accuracy(judge_results: List[JudgeResult]) -> Dict[str, float]:
    """
    Compute answer accuracy metrics from judge results.
    
    Args:
        judge_results: List of JudgeResult objects
        
    Returns:
        Dictionary with accuracy metrics
    """
    if not judge_results:
        return {'accuracy': 0.0, 'n': 0}
    
    scores = [r.score for r in judge_results]
    
    # Count verdicts
    verdicts = [r.verdict for r in judge_results]
    n_correct = sum(1 for v in verdicts if v == 'CORRECT')
    n_partial = sum(1 for v in verdicts if v == 'PARTIALLY_CORRECT')
    n_incorrect = sum(1 for v in verdicts if v == 'INCORRECT')
    
    import numpy as np
    
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
    
    # 95% CI
    n = len(scores)
    ci_margin = 1.96 * std_score / np.sqrt(n) if n > 1 else 0.0
    
    return {
        'accuracy': float(mean_score),
        'std': float(std_score),
        'ci_lower': float(mean_score - ci_margin),
        'ci_upper': float(mean_score + ci_margin),
        'n_correct': n_correct,
        'n_partial': n_partial,
        'n_incorrect': n_incorrect,
        'n': n
    }


def aggregate_by_model_and_method(
    judge_results: List[JudgeResult]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate results by model and retrieval method.
    
    Args:
        judge_results: List of JudgeResult objects
        
    Returns:
        Nested dictionary: model -> method -> metrics
    """
    from collections import defaultdict
    
    # Group by model and method
    grouped = defaultdict(lambda: defaultdict(list))
    
    for result in judge_results:
        grouped[result.model_name][result.retrieval_method].append(result)
    
    # Compute metrics for each group
    aggregated = {}
    
    for model_name, methods in grouped.items():
        aggregated[model_name] = {}
        for method_name, results in methods.items():
            aggregated[model_name][method_name] = compute_answer_accuracy(results)
    
    return aggregated
