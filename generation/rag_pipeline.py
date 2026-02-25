"""
RAG Pipeline for Kazakh Legal QA.
Combines retrieval and generation with proper context formatting.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYSTEM_PROMPT, TOP_K, LLMModel
from generation.llm_client import LLMClient, LLMResponse
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    question: str
    answer: str
    retrieved_chunks: List[Dict]
    model_name: str
    retrieval_method: str
    usage: Dict[str, int]
    citations: List[str]


class RAGPipeline:
    """
    RAG Pipeline for Kazakh Legal QA.
    
    Supports three retrieval methods:
    - BM25 (lexical)
    - Vector (semantic)
    - Hybrid (Weighted RRF)
    """
    
    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm_client: Optional[LLMClient] = None,
        top_k: int = TOP_K
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            bm25_retriever: BM25 retriever (optional)
            vector_retriever: Vector retriever (optional)
            hybrid_retriever: Hybrid retriever (optional)
            llm_client: LLM client for generation
            top_k: Number of chunks to retrieve
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client or LLMClient()
        self.top_k = top_k
    
    def _format_context(self, chunks: List[Tuple[int, float, Dict]]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks: List of (chunk_idx, score, chunk_dict) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for chunk_idx, score, chunk in chunks:
            doc_id = chunk['doc_id']
            chunk_id = chunk['chunk_id']
            text = chunk['text']
            
            # Format with citation marker
            context_part = f"[{doc_id}:{chunk_id}]\n{text}"
            context_parts.append(context_part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_citations(self, answer: str) -> List[str]:
        """
        Extract citations from answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            List of citation strings
        """
        import re
        # Match [doc_id:chunk_id] pattern
        pattern = r'\[([^\]]+):(\d+)\]'
        matches = re.findall(pattern, answer)
        return [f"{doc_id}:{chunk_id}" for doc_id, chunk_id in matches]
    
    def retrieve(
        self,
        query: str,
        method: str = 'hybrid'
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve relevant chunks.
        
        Args:
            query: Query string
            method: 'bm25', 'vector', or 'hybrid'
            
        Returns:
            List of (chunk_idx, score, chunk_dict) tuples
        """
        if method == 'bm25':
            if not self.bm25_retriever:
                raise ValueError("BM25 retriever not initialized")
            return self.bm25_retriever.retrieve(query, self.top_k)
        
        elif method == 'vector':
            if not self.vector_retriever:
                raise ValueError("Vector retriever not initialized")
            return self.vector_retriever.retrieve(query, self.top_k)
        
        elif method == 'hybrid':
            if not self.hybrid_retriever:
                raise ValueError("Hybrid retriever not initialized")
            return self.hybrid_retriever.retrieve(query, self.top_k)
        
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def generate(
        self,
        question: str,
        context: str,
        model: LLMModel
    ) -> LLMResponse:
        """
        Generate answer using LLM.
        
        Args:
            question: Question text
            context: Formatted context string
            model: LLM model configuration
            
        Returns:
            LLMResponse object
        """
        # Format prompt using template
        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question
        )
        
        # Generate response
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.generate(model, messages)
    
    def run(
        self,
        question: str,
        model: LLMModel,
        retrieval_method: str = 'hybrid'
    ) -> RAGResult:
        """
        Run full RAG pipeline.
        
        Args:
            question: Question text
            model: LLM model configuration
            retrieval_method: 'bm25', 'vector', or 'hybrid'
            
        Returns:
            RAGResult object
        """
        # Retrieve chunks
        chunks = self.retrieve(question, retrieval_method)
        
        # Format context
        context = self._format_context(chunks)
        
        # Generate answer
        response = self.generate(question, context, model)
        
        # Extract citations
        citations = self._extract_citations(response.content)
        
        # Build result
        return RAGResult(
            question=question,
            answer=response.content,
            retrieved_chunks=[
                {
                    'chunk_idx': idx,
                    'score': score,
                    'doc_id': chunk['doc_id'],
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'][:500] + '...' if len(chunk['text']) > 500 else chunk['text']
                }
                for idx, score, chunk in chunks
            ],
            model_name=model.name,
            retrieval_method=retrieval_method,
            usage=response.usage,
            citations=citations
        )


def create_pipeline(
    retrieval_methods: List[str] = ['bm25', 'vector', 'hybrid'],
    bm25_tokenizer_type: str = 'kaznlp',
    rrf_k: int = 60,
    alpha: float = 0.5
) -> RAGPipeline:
    """
    Create RAG pipeline with specified retrievers.
    
    Args:
        retrieval_methods: List of methods to enable
        bm25_tokenizer_type: 'kaznlp' or 'whitespace'
        rrf_k: RRF smoothing constant for hybrid
        alpha: BM25 weight for hybrid
        
    Returns:
        RAGPipeline instance
    """
    from retrieval.bm25_retriever import load_bm25_retriever
    from retrieval.vector_retriever import load_vector_retriever
    from retrieval.hybrid_retriever import load_hybrid_retriever
    
    bm25_retriever = None
    vector_retriever = None
    hybrid_retriever = None
    
    if 'bm25' in retrieval_methods:
        print("Loading BM25 retriever...")
        bm25_retriever = load_bm25_retriever(bm25_tokenizer_type)
    
    if 'vector' in retrieval_methods:
        print("Loading Vector retriever...")
        vector_retriever = load_vector_retriever()
    
    if 'hybrid' in retrieval_methods:
        print("Loading Hybrid retriever...")
        hybrid_retriever = load_hybrid_retriever(
            bm25_tokenizer_type=bm25_tokenizer_type,
            rrf_k=rrf_k,
            alpha=alpha
        )
    
    return RAGPipeline(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        hybrid_retriever=hybrid_retriever
    )
