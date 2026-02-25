"""
Configuration for Kazakh Legal RAG Experiment.

Paper: "Development of the Retrieval-Augmented Generation (RAG) System
for the Kazakh Language Using Hybrid Information Methods"
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# Paths
BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "kaz"
DATA_DIR = BASE_DIR / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = BASE_DIR / "results"
QUESTIONS_FILE = DATA_DIR / "questions.json"

# Chunking parameters
CHUNK_SIZE = 900  # tokens
CHUNK_OVERLAP = 150  # tokens
TOKENIZER_MODEL = "cl100k_base"  # tiktoken model for OpenAI compatibility

# Embedding parameters
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 100  # documents per API call

# Retrieval parameters
TOP_K = 6  # number of chunks to retrieve

# Hybrid RRF Grid Search parameters
RRF_K_VALUES = [10, 20, 40, 60, 100]
RRF_ALPHA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]

# Cross-validation parameters
N_FOLDS = 5
N_QUESTIONS = 100

# LLM Models configuration
@dataclass
class LLMModel:
    name: str
    api_id: str
    provider: str  # "openai" or "openrouter"
    temperature: float = 0.0

LLM_MODELS: List[LLMModel] = [
    # GPT-4o / GPT-5 family: temperature 0.3 (Table 1 in paper)
    LLMModel("GPT-4o", "gpt-4o", "openai", temperature=0.3),
    LLMModel("GPT-4o-mini", "gpt-4o-mini", "openai", temperature=0.3),
    LLMModel("GPT-5", "gpt-5", "openai", temperature=0.3),
    LLMModel("GPT-5-mini", "gpt-5-mini", "openai", temperature=0.3),
    # o1 family: temperature 1.0 (reasoning models, Table 1 in paper)
    LLMModel("o1", "o1", "openai", temperature=1.0),
    LLMModel("o1-mini", "o1-mini", "openai", temperature=1.0),
]

# Judge model (for LLM-as-judge evaluation)
JUDGE_MODEL = LLMModel("GPT-4o-Judge", "gpt-4o", "openai", temperature=0.0)

# Generation parameters
MAX_OUTPUT_TOKENS = 500  # as specified in Table 1 of the paper
TOP_P = 1.0

# System prompt template (Figure 1 in the paper)
SYSTEM_PROMPT = """You are a legal QA system operating over a fixed corpus of Kazakhstani laws.

Your task is to answer the question using only the information contained in the provided context excerpts.
You must not use any external knowledge or assumptions.

Rules:
1. Use only facts that appear verbatim or can be directly inferred from the context.
2. Do not add any legal interpretations, opinions, or explanations beyond what is stated.
3. Do not paraphrase in a way that changes legal meaning.
4. If multiple excerpts are relevant, combine them faithfully.
5. Do not include any information not grounded in the context.
6. Do not mention the word "context" in your answer.

At the end of your answer, provide a list of citations in the format:
Source:
- [doc_id:chunk_id]

Context:
{context}

Question:
{question}

Answer:"""

# Judge prompt template (from todo.md)
JUDGE_PROMPT = """You are evaluating the correctness of an AI-generated answer to a legal question about Kazakhstani law.

Question (Kazakh): {question}
Reference Answer: {gold_answer}
Generated Answer: {generated_answer}

Evaluate the generated answer against the reference:

CORRECT — The answer is factually accurate and fully addresses the question. Minor wording differences are acceptable if the legal meaning is preserved.

PARTIALLY_CORRECT — The answer contains relevant correct information but is incomplete, missing key details, or includes minor inaccuracies that don't change the core legal meaning.

INCORRECT — The answer is factually wrong, contradicts the reference, fails to answer the question, or states "Answer not found" when the answer exists in the reference.

Output exactly one word: CORRECT, PARTIALLY_CORRECT, or INCORRECT"""

# Score mapping for judge responses
JUDGE_SCORES = {
    "CORRECT": 1.0,
    "PARTIALLY_CORRECT": 0.5,
    "INCORRECT": 0.0
}

# Statistical test parameters
SIGNIFICANCE_LEVEL = 0.05

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
