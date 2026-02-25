"""
Main experiment runner for Kazakh Legal RAG experiment.
Orchestrates indexing, retrieval, generation, and evaluation.
With detailed logging for analysis.
"""

import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CORPUS_DIR, CHUNKS_DIR, INDEX_DIR, RESULTS_DIR,
    QUESTIONS_FILE, LLM_MODELS, TOP_K, N_FOLDS
)
from experiment.run_metadata import get_run_metadata


def setup_logging(run_dir: Path) -> logging.Logger:
    """Setup detailed logging."""
    logger = logging.getLogger('rag_experiment')
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logs
    fh = logging.FileHandler(run_dir / 'experiment.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler - info and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def setup_directories():
    """Create necessary directories."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_indexing(force_reindex: bool = False, logger=None):
    """Run corpus indexing (chunking, BM25, FAISS)."""
    from indexing.chunker import process_corpus, load_chunks
    from indexing.bm25_index import build_all_indices
    from indexing.faiss_index import build_index as build_faiss_index
    
    log = logger.info if logger else print
    
    chunks_file = CHUNKS_DIR / "all_chunks.json"
    bm25_file = INDEX_DIR / "bm25_kaznlp.pkl"
    faiss_file = INDEX_DIR / "faiss_index.faiss"
    
    if force_reindex or not chunks_file.exists():
        log("STEP 1: Chunking corpus...")
        process_corpus()
    else:
        log(f"Chunks already exist at {chunks_file}")
    
    if force_reindex or not bm25_file.exists():
        log("STEP 2: Building BM25 indices...")
        build_all_indices()
    else:
        log(f"BM25 index already exists at {bm25_file}")
    
    if force_reindex or not faiss_file.exists():
        log("STEP 3: Building FAISS index...")
        build_faiss_index()
    else:
        log(f"FAISS index already exists at {faiss_file}")


def run_experiment(
    run_generation: bool = True,
    models: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    save_intermediate: bool = True,
    logger: logging.Logger = None
):
    """
    Run the full experiment with detailed logging.
    """
    from indexing.chunker import load_chunks, load_chunk_index
    from retrieval.bm25_retriever import load_bm25_retriever
    from retrieval.vector_retriever import load_vector_retriever
    from retrieval.hybrid_retriever import load_hybrid_retriever
    from generation.rag_pipeline import RAGPipeline
    from generation.llm_client import LLMClient, get_model_by_name
    from evaluation.judge import LLMJudge
    from evaluation.metrics import calculate_retrieval_metrics
    import numpy as np
    
    def aggregate_metrics_dict(metrics_list):
        """Aggregate list of metric dicts."""
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        result = {}
        for key in keys:
            values = [m[key] for m in metrics_list]
            arr = np.array(values)
            result[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                'n': len(arr)
            }
        return result
    
    log = logger
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging if not provided
    if not logger:
        logger = setup_logging(run_dir)
        log = logger
    
    log.info("=" * 70)
    log.info(f"EXPERIMENT RUN: {timestamp}")
    log.info("=" * 70)
    
    # Load questions
    log.info("\n[1/6] Loading questions...")
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data.get('questions', [])
    
    if len(questions) == 0:
        log.error("No questions found!")
        return None
    
    log.info(f"  Loaded {len(questions)} questions")
    
    # Load chunk index
    log.info("\n[2/6] Loading chunk index...")
    chunk_index = load_chunk_index()
    log.info(f"  Index has {len(chunk_index)} documents")
    
    # Load chunks
    log.info("\n[3/6] Loading chunks...")
    start = time.time()
    chunks = load_chunks()
    log.info(f"  Loaded {len(chunks)} chunks in {time.time()-start:.1f}s")
    
    # Load retrievers
    log.info("\n[4/6] Loading retrievers...")
    
    log.info("  Loading BM25 retriever...")
    start = time.time()
    bm25_retriever = load_bm25_retriever('kaznlp')
    log.info(f"    BM25 loaded in {time.time()-start:.1f}s")
    
    log.info("  Loading Vector retriever...")
    start = time.time()
    vector_retriever = load_vector_retriever()
    log.info(f"    Vector loaded in {time.time()-start:.1f}s")
    
    log.info("  Loading Hybrid retriever...")
    start = time.time()
    hybrid_retriever = load_hybrid_retriever()
    log.info(f"    Hybrid loaded in {time.time()-start:.1f}s")
    
    # Select models
    if models:
        selected_models = [get_model_by_name(m) for m in models if get_model_by_name(m)]
    else:
        selected_models = LLM_MODELS
    
    # Retrieval methods
    retrieval_methods = methods or ['bm25', 'vector', 'hybrid']
    
    # Exact model versions and run metadata (for reproducibility)
    run_metadata = get_run_metadata(
        run_id=timestamp,
        selected_models=selected_models,
        retrieval_methods=retrieval_methods,
    )
    
    # Save run metadata (exact model versions) for reproducibility
    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)
    log.debug(f"Run metadata saved to {metadata_file}")
    
    log.info(f"\n[5/6] Models to evaluate: {[m.name for m in selected_models]}")
    log.info(f"  API IDs: {[m.api_id for m in selected_models]}")
    log.info(f"  Retrieval methods: {retrieval_methods}")
    
    # ========================================
    # RETRIEVAL EVALUATION
    # ========================================
    log.info("\n" + "=" * 70)
    log.info("RETRIEVAL EVALUATION")
    log.info("=" * 70)
    
    retrieval_results = {method: [] for method in retrieval_methods}
    
    for q_idx, q in enumerate(questions):
        query = q['question']
        # Get gold passage indices
        gold_passages = q.get('gold_passages', [])
        gold_indices = set()
        for gp in gold_passages:
            if 'global_idx' in gp:
                gold_indices.add(gp['global_idx'])
            elif 'doc_id' in gp and 'chunk_id' in gp:
                doc_id = gp['doc_id']
                if doc_id in chunk_index:
                    doc_chunks = chunk_index[doc_id]
                    chunk_id = gp['chunk_id']
                    if chunk_id < len(doc_chunks):
                        gold_indices.add(doc_chunks[chunk_id])
        
        log.debug(f"Q{q_idx+1}: {query[:50]}... | Gold: {gold_indices}")
        
        # Evaluate each method
        for method in retrieval_methods:
            if method == 'bm25':
                results = bm25_retriever.retrieve(query, TOP_K)
            elif method == 'vector':
                results = vector_retriever.retrieve(query, TOP_K)
            else:  # hybrid
                results = hybrid_retriever.retrieve(query, TOP_K)
            
            retrieved_ids = [r[0] for r in results]
            
            # Calculate metrics
            metrics = calculate_retrieval_metrics(retrieved_ids, gold_indices, TOP_K)
            # Convert to dict for easier handling
            metrics_dict = {
                'recall_at_k': metrics.recall_at_k,
                'precision_at_k': metrics.precision_at_k,
                'mrr': metrics.mrr,
                'ndcg_at_k': metrics.ndcg_at_k
            }
            retrieval_results[method].append(metrics_dict)
            
            log.debug(f"  {method}: Retrieved {retrieved_ids[:3]}... | R@{TOP_K}={metrics.recall_at_k:.2f}")
    
    # Aggregate retrieval metrics
    log.info("\n--- Retrieval Results ---")
    aggregated_retrieval = {}
    for method in retrieval_methods:
        agg = aggregate_metrics_dict(retrieval_results[method])
        aggregated_retrieval[method] = agg
        log.info(f"\n{method.upper()}:")
        log.info(f"  Recall@{TOP_K}:    {agg['recall_at_k']['mean']:.4f} ± {agg['recall_at_k']['std']:.4f}")
        log.info(f"  Precision@{TOP_K}: {agg['precision_at_k']['mean']:.4f} ± {agg['precision_at_k']['std']:.4f}")
        log.info(f"  MRR:              {agg['mrr']['mean']:.4f} ± {agg['mrr']['std']:.4f}")
        log.info(f"  NDCG@{TOP_K}:      {agg['ndcg_at_k']['mean']:.4f} ± {agg['ndcg_at_k']['std']:.4f}")
    
    # Save retrieval results (with exact model versions and metadata)
    retrieval_file = run_dir / "retrieval_results.json"
    with open(retrieval_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': run_metadata,
            'aggregated': aggregated_retrieval,
            'per_question': {m: retrieval_results[m] for m in retrieval_methods}
        }, f, indent=2, ensure_ascii=False)
    log.info(f"\nRetrieval results saved to {retrieval_file}")
    
    # ========================================
    # GENERATION EVALUATION
    # ========================================
    if run_generation:
        log.info("\n" + "=" * 70)
        log.info("[6/6] GENERATION EVALUATION")
        log.info("=" * 70)
        
        llm_client = LLMClient()
        judge = LLMJudge()
        
        generation_results = {}
        
        for model in selected_models:
            log.info(f"\n--- Model: {model.name} ({model.api_id}) ---")
            model_results = {method: {'scores': [], 'details': []} for method in retrieval_methods}
            
            for q_idx, q in enumerate(questions):
                query = q['question']
                gold_answer = q.get('gold_answer', '')
                
                for method in retrieval_methods:
                    # Retrieve context
                    if method == 'bm25':
                        results = bm25_retriever.retrieve(query, TOP_K)
                    elif method == 'vector':
                        results = vector_retriever.retrieve(query, TOP_K)
                    else:
                        results = hybrid_retriever.retrieve(query, TOP_K)
                    
                    # Format context
                    context_parts = []
                    for i, (idx, score, chunk) in enumerate(results):
                        doc_id = chunk.get('doc_id', 'unknown')
                        chunk_id = chunk.get('chunk_idx', 0)
                        text = chunk.get('text', '')
                        context_parts.append(f"[{doc_id}:{chunk_id}]\n{text}")
                    
                    context = "\n\n---\n\n".join(context_parts)
                    
                    # Generate answer
                    from config import SYSTEM_PROMPT
                    prompt = SYSTEM_PROMPT.format(context=context, question=query)
                    
                    try:
                        log.debug(f"Q{q_idx+1}/{method}: Generating with {model.name}...")
                        start = time.time()
                        
                        response = llm_client.generate(
                            model=model,
                            prompt=prompt,
                            system_prompt=None  # Already in prompt
                        )
                        
                        generated_answer = response.content
                        gen_time = time.time() - start
                        
                        log.debug(f"  Generated in {gen_time:.1f}s, {len(generated_answer)} chars")
                        
                        # Judge answer
                        judge_result = judge.evaluate(
                            question=query,
                            gold_answer=gold_answer,
                            generated_answer=generated_answer
                        )
                        
                        score = judge_result.score
                        model_results[method]['scores'].append(score)
                        model_results[method]['details'].append({
                            'q_id': q['id'],
                            'question': query[:100],
                            'generated': generated_answer[:200],
                            'verdict': judge_result.verdict,
                            'score': score,
                            'time': gen_time
                        })
                        
                        log.info(f"  Q{q_idx+1}/{method}: {judge_result.verdict} (score={score})")
                        
                    except Exception as e:
                        log.error(f"  Q{q_idx+1}/{method}: ERROR - {str(e)}")
                        log.debug(traceback.format_exc())
                        model_results[method]['scores'].append(0.0)
                        model_results[method]['details'].append({
                            'q_id': q['id'],
                            'error': str(e)
                        })
            
            # Calculate model stats
            for method in retrieval_methods:
                scores = model_results[method]['scores']
                if scores:
                    mean_score = sum(scores) / len(scores)
                    model_results[method]['mean'] = mean_score
                    model_results[method]['total'] = len(scores)
                    log.info(f"\n{model.name} + {method}: Mean Accuracy = {mean_score:.4f}")
            
            generation_results[model.name] = model_results
        
        # Save generation results (with exact model versions and metadata)
        generation_file = run_dir / "generation_results.json"
        with open(generation_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': run_metadata,
                'generation_results': generation_results,
            }, f, indent=2, ensure_ascii=False)
        log.info(f"\nGeneration results saved to {generation_file}")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 70)
    log.info(f"\nResults directory: {run_dir}")
    
    # Summary table
    log.info("\n=== RETRIEVAL SUMMARY ===")
    log.info(f"{'Method':<12} {'Recall@6':<12} {'MRR':<12} {'NDCG@6':<12}")
    log.info("-" * 48)
    for method in retrieval_methods:
        agg = aggregated_retrieval[method]
        log.info(f"{method:<12} {agg['recall_at_k']['mean']:.4f}       {agg['mrr']['mean']:.4f}       {agg['ndcg_at_k']['mean']:.4f}")
    
    if run_generation and generation_results:
        log.info("\n=== GENERATION SUMMARY ===")
        log.info(f"{'Model':<16} {'BM25':<10} {'Vector':<10} {'Hybrid':<10}")
        log.info("-" * 46)
        for model_name, results in generation_results.items():
            bm25_score = results.get('bm25', {}).get('mean', 0)
            vector_score = results.get('vector', {}).get('mean', 0)
            hybrid_score = results.get('hybrid', {}).get('mean', 0)
            log.info(f"{model_name:<16} {bm25_score:.4f}     {vector_score:.4f}     {hybrid_score:.4f}")
    
    return {
        'retrieval': aggregated_retrieval,
        'generation': generation_results if run_generation else None,
        'run_dir': str(run_dir)
    }


def main():
    parser = argparse.ArgumentParser(description='Kazakh Legal RAG Experiment')
    parser.add_argument('--index', action='store_true', help='Run indexing only')
    parser.add_argument('--force-reindex', action='store_true', help='Force re-indexing')
    parser.add_argument('--no-generation', action='store_true', help='Skip LLM generation')
    parser.add_argument('--models', nargs='+', help='Models to evaluate')
    parser.add_argument('--retrieval-only', action='store_true', help='Evaluate retrieval only')
    
    args = parser.parse_args()
    
    setup_directories()
    
    # Create run directory early for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir)
    
    if args.index:
        run_indexing(force_reindex=args.force_reindex, logger=logger)
        return
    
    # Ensure indices exist
    run_indexing(force_reindex=args.force_reindex, logger=logger)
    
    # Run experiment
    run_experiment(
        run_generation=not args.no_generation and not args.retrieval_only,
        models=args.models,
        logger=logger
    )


if __name__ == "__main__":
    main()
