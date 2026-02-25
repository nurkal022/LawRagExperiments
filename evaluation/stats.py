"""
Statistical tests for experiment analysis.
Implements Wilcoxon, McNemar, Friedman, and Nemenyi tests.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import warnings


@dataclass
class TestResult:
    """Result from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    comparison: Optional[str] = None


def wilcoxon_signed_rank(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
    name_a: str = "Method A",
    name_b: str = "Method B"
) -> TestResult:
    """
    Wilcoxon signed-rank test for paired samples.
    
    Tests whether there is a significant difference between two related samples.
    Non-parametric alternative to paired t-test.
    
    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B
        alpha: Significance level
        name_a: Name of method A
        name_b: Name of method B
        
    Returns:
        TestResult object
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    # Handle tied values (differences of zero)
    differences = scores_a - scores_b
    
    # Remove zero differences for Wilcoxon test
    non_zero_mask = differences != 0
    if not np.any(non_zero_mask):
        # All differences are zero - no significant difference
        return TestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            comparison=f"{name_a} vs {name_b}"
        )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value = wilcoxon(
            scores_a[non_zero_mask],
            scores_b[non_zero_mask],
            alternative='two-sided'
        )
    
    # Calculate effect size (r = Z / sqrt(N))
    n = np.sum(non_zero_mask)
    z_score = stats.norm.ppf(1 - p_value / 2)
    effect_size = abs(z_score) / np.sqrt(n) if n > 0 else 0
    
    return TestResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(effect_size),
        comparison=f"{name_a} vs {name_b}"
    )


def mcnemar_test(
    correct_a: List[bool],
    correct_b: List[bool],
    alpha: float = 0.05,
    name_a: str = "Method A",
    name_b: str = "Method B"
) -> TestResult:
    """
    McNemar's test for paired binary outcomes.
    
    Tests whether two methods have significantly different accuracy on paired samples.
    
    Args:
        correct_a: Binary outcomes for method A (True = correct)
        correct_b: Binary outcomes for method B (True = correct)
        alpha: Significance level
        name_a: Name of method A
        name_b: Name of method B
        
    Returns:
        TestResult object
    """
    correct_a = np.array(correct_a)
    correct_b = np.array(correct_b)
    
    # Build contingency table
    # n01: A wrong, B correct
    # n10: A correct, B wrong
    n01 = np.sum((~correct_a) & correct_b)
    n10 = np.sum(correct_a & (~correct_b))
    
    # McNemar's test with continuity correction
    if n01 + n10 == 0:
        return TestResult(
            test_name="McNemar",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            comparison=f"{name_a} vs {name_b}"
        )
    
    # Chi-square statistic with continuity correction
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return TestResult(
        test_name="McNemar",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        comparison=f"{name_a} vs {name_b}"
    )


def friedman_test(
    scores_dict: Dict[str, List[float]],
    alpha: float = 0.05
) -> TestResult:
    """
    Friedman test for comparing multiple related samples.
    
    Non-parametric alternative to repeated-measures ANOVA.
    Tests whether there are significant differences among K treatments.
    
    Args:
        scores_dict: Dictionary mapping method names to score lists
        alpha: Significance level
        
    Returns:
        TestResult object
    """
    method_names = list(scores_dict.keys())
    scores_matrix = np.array([scores_dict[m] for m in method_names])
    
    if len(method_names) < 3:
        raise ValueError("Friedman test requires at least 3 groups")
    
    # Friedman test
    statistic, p_value = friedmanchisquare(*scores_matrix)
    
    return TestResult(
        test_name="Friedman",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        comparison=f"Comparing: {', '.join(method_names)}"
    )


def nemenyi_posthoc(
    scores_dict: Dict[str, List[float]],
    alpha: float = 0.05
) -> List[TestResult]:
    """
    Nemenyi post-hoc test for pairwise comparisons after Friedman.
    
    Uses critical difference to determine significant differences.
    
    Args:
        scores_dict: Dictionary mapping method names to score lists
        alpha: Significance level
        
    Returns:
        List of TestResult objects for each pairwise comparison
    """
    method_names = list(scores_dict.keys())
    k = len(method_names)  # number of methods
    n = len(list(scores_dict.values())[0])  # number of samples
    
    # Calculate average ranks for each method
    scores_matrix = np.array([scores_dict[m] for m in method_names]).T  # samples x methods
    ranks = np.zeros_like(scores_matrix, dtype=float)
    
    for i in range(n):
        ranks[i] = stats.rankdata(-scores_matrix[i])  # negative for descending rank
    
    avg_ranks = {
        method: np.mean(ranks[:, j])
        for j, method in enumerate(method_names)
    }
    
    # Critical difference for Nemenyi test
    # q_alpha values for k groups at alpha=0.05
    # These are from Demsar (2006) critical values table
    q_alpha_table = {
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164
    }
    
    q_alpha = q_alpha_table.get(k, 3.0)  # default to 3.0 if k not in table
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    
    # Pairwise comparisons
    results = []
    
    for i, method_a in enumerate(method_names):
        for j, method_b in enumerate(method_names):
            if i >= j:
                continue
            
            rank_diff = abs(avg_ranks[method_a] - avg_ranks[method_b])
            significant = rank_diff > cd
            
            results.append(TestResult(
                test_name="Nemenyi",
                statistic=float(rank_diff),
                p_value=np.nan,  # Nemenyi doesn't give p-value directly
                significant=significant,
                effect_size=float(cd),  # Store critical difference as reference
                comparison=f"{method_a} vs {method_b} (ranks: {avg_ranks[method_a]:.2f} vs {avg_ranks[method_b]:.2f})"
            ))
    
    return results


def cochrans_q_test(
    correct_dict: Dict[str, List[bool]],
    alpha: float = 0.05
) -> TestResult:
    """
    Cochran's Q test for comparing multiple binary outcomes.
    
    Extension of McNemar's test to K groups.
    
    Args:
        correct_dict: Dictionary mapping method names to binary outcome lists
        alpha: Significance level
        
    Returns:
        TestResult object
    """
    method_names = list(correct_dict.keys())
    k = len(method_names)
    n = len(list(correct_dict.values())[0])
    
    # Build matrix: samples x methods
    matrix = np.array([correct_dict[m] for m in method_names], dtype=int).T
    
    # Row sums (number of correct per sample)
    L = np.sum(matrix, axis=1)
    
    # Column sums (number of correct per method)
    T = np.sum(matrix, axis=0)
    
    # Cochran's Q statistic
    numerator = (k - 1) * (k * np.sum(T ** 2) - np.sum(T) ** 2)
    denominator = k * np.sum(L) - np.sum(L ** 2)
    
    if denominator == 0:
        return TestResult(
            test_name="Cochran's Q",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            comparison=f"Comparing: {', '.join(method_names)}"
        )
    
    Q = numerator / denominator
    p_value = 1 - stats.chi2.cdf(Q, df=k - 1)
    
    return TestResult(
        test_name="Cochran's Q",
        statistic=float(Q),
        p_value=float(p_value),
        significant=p_value < alpha,
        comparison=f"Comparing: {', '.join(method_names)}"
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Family-wise significance level
        
    Returns:
        List of (adjusted_alpha, significant) tuples
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    return [(adjusted_alpha, p < adjusted_alpha) for p in p_values]


def format_test_results(results: List[TestResult]) -> str:
    """Format test results as a readable string."""
    lines = []
    
    for result in results:
        lines.append(f"\n{result.test_name} Test")
        lines.append("-" * 40)
        
        if result.comparison:
            lines.append(f"Comparison: {result.comparison}")
        
        lines.append(f"Statistic: {result.statistic:.4f}")
        
        if not np.isnan(result.p_value):
            lines.append(f"P-value: {result.p_value:.4f}")
        
        if result.effect_size is not None:
            lines.append(f"Effect size/CD: {result.effect_size:.4f}")
        
        sig_str = "YES (reject H0)" if result.significant else "NO (fail to reject H0)"
        lines.append(f"Significant: {sig_str}")
    
    return "\n".join(lines)


def run_all_statistical_tests(
    retrieval_scores: Dict[str, List[float]],
    accuracy_scores: Dict[str, Dict[str, List[float]]],
    binary_outcomes: Dict[str, Dict[str, List[bool]]],
    alpha: float = 0.05
) -> Dict[str, List[TestResult]]:
    """
    Run all statistical tests on experiment results.
    
    Args:
        retrieval_scores: method -> list of metric scores
        accuracy_scores: model -> method -> list of accuracy scores
        binary_outcomes: model -> method -> list of correct/incorrect
        alpha: Significance level
        
    Returns:
        Dictionary with test results by category
    """
    results = {
        'retrieval_pairwise': [],
        'retrieval_overall': [],
        'accuracy_pairwise': [],
        'accuracy_overall': [],
        'binary_tests': []
    }
    
    methods = list(retrieval_scores.keys())
    
    # 1. Retrieval metrics - Friedman test
    if len(methods) >= 3:
        friedman_result = friedman_test(retrieval_scores, alpha)
        results['retrieval_overall'].append(friedman_result)
        
        # If significant, do post-hoc
        if friedman_result.significant:
            nemenyi_results = nemenyi_posthoc(retrieval_scores, alpha)
            results['retrieval_pairwise'].extend(nemenyi_results)
    
    # 2. Pairwise Wilcoxon tests
    for i, method_a in enumerate(methods):
        for j, method_b in enumerate(methods):
            if i >= j:
                continue
            
            wilcoxon_result = wilcoxon_signed_rank(
                retrieval_scores[method_a],
                retrieval_scores[method_b],
                alpha,
                method_a,
                method_b
            )
            results['retrieval_pairwise'].append(wilcoxon_result)
    
    # 3. Accuracy tests per model
    for model, method_scores in accuracy_scores.items():
        if len(method_scores) >= 3:
            friedman_result = friedman_test(method_scores, alpha)
            friedman_result.comparison = f"{model}: " + friedman_result.comparison
            results['accuracy_overall'].append(friedman_result)
    
    # 4. Binary tests (McNemar, Cochran's Q)
    for model, method_outcomes in binary_outcomes.items():
        methods = list(method_outcomes.keys())
        
        # Cochran's Q for all methods
        if len(methods) >= 3:
            cochran_result = cochrans_q_test(method_outcomes, alpha)
            cochran_result.comparison = f"{model}: " + cochran_result.comparison
            results['binary_tests'].append(cochran_result)
        
        # McNemar pairwise
        for i, method_a in enumerate(methods):
            for j, method_b in enumerate(methods):
                if i >= j:
                    continue
                
                mcnemar_result = mcnemar_test(
                    method_outcomes[method_a],
                    method_outcomes[method_b],
                    alpha,
                    f"{model}/{method_a}",
                    f"{model}/{method_b}"
                )
                results['binary_tests'].append(mcnemar_result)
    
    return results
