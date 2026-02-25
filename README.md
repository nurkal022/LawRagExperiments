# LawRagExperiments

Source code and data for the paper:

> **Development of the Retrieval-Augmented Generation (RAG) System for the Kazakh Language Using Hybrid Information Methods**
>
> N. Kalzhanov, S. Artykbay, A. Kalzhan

This repository contains the full implementation of a RAG system for Kazakh legal question answering, including three retrieval strategies (BM25, vector search, hybrid Weighted RRF), six LLM configurations, and a complete evaluation framework. Experiments were conducted on a corpus of 12,886 Kazakhstani legal documents (263,326 text segments).

## Key Results

| Method | Recall@6 | Precision@6 | MRR | NDCG@6 | Avg Accuracy |
|--------|----------|-------------|-----|--------|--------------|
| **Hybrid** | **0.890** | **0.148** | **0.720** | **0.785** | **82.0%** |
| Vector | 0.830 | 0.138 | 0.650 | 0.710 | 77.7% |
| BM25 | 0.790 | 0.132 | 0.610 | 0.670 | 71.7% |

The hybrid method achieves the highest accuracy (up to 89%) and the lowest coefficient of variation (CV=0.057), statistically outperforming both baselines (Cochran's Q, McNemar post-hoc, p < 0.01).

## Project Structure

```
├── config.py                 # Central configuration (models, prompts, parameters)
├── requirements.txt          # Python dependencies
├── .env.example              # API keys template
│
├── indexing/                  # Document processing & index building
│   ├── chunker.py            # Tiktoken-based chunking (900 tokens, 150 overlap)
│   ├── bm25_index.py         # BM25 with Kazakh-aware tokenization (KazNLP)
│   └── faiss_index.py        # FAISS vector index (text-embedding-3-small, 1536d)
│
├── retrieval/                 # Search strategies
│   ├── bm25_retriever.py     # Lexical retrieval (BM25Okapi, k1=1.5, b=0.75)
│   ├── vector_retriever.py   # Dense retrieval (FAISS IndexFlatIP, cosine similarity)
│   └── hybrid_retriever.py   # Weighted RRF (Eq. 2 in paper)
│
├── generation/                # Answer generation
│   ├── llm_client.py         # OpenAI API client
│   └── rag_pipeline.py       # End-to-end RAG pipeline
│
├── evaluation/                # Evaluation framework
│   ├── metrics.py            # Recall@k, Precision@k, MRR, NDCG@k
│   ├── judge.py              # Answer correctness evaluation
│   └── stats.py              # Cochran's Q, McNemar, Wilcoxon, Friedman
│
├── experiment/                # Experiment orchestration
│   ├── run_experiment.py     # Main experiment runner (1,800 runs)
│   ├── cross_validation.py   # Cross-validation for RRF parameter tuning
│   ├── grid_search.py        # Grid search over k_rrf and alpha
│   └── run_metadata.py       # Reproducibility metadata
│
├── scripts/
│   ├── generate_questions.py # Test set generation (100 questions)
│   └── create_charts.py      # Figures for the paper
│
├── kaz/                       # Corpus: 12,886 Kazakh legal documents
├── data/
│   └── questions.json        # 100 QA pairs with gold passages
│
├── results/                   # Experiment outputs
│   ├── EXPERIMENT_REPORT.md
│   ├── experiment_summary.json
│   ├── figures/              # Charts (Figures 2-5 in paper)
│   └── run_*/                # Timestamped run logs
│
└── docs/
    └── glossary_review.md
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/nurkal022/LawRagExperiments.git
cd LawRagExperiments

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

### 3. Build indices

Chunks and indices are not included in the repository (they are large generated artifacts). Rebuild them from the corpus:

```python
from experiment.run_experiment import run_indexing
run_indexing()
```

This processes `kaz/` corpus, creates 263,326 chunks, and builds BM25 + FAISS indices in `data/`.

### 4. Run experiments

```python
from experiment.run_experiment import run_experiment
run_experiment()
```

Results are saved to timestamped directories under `results/`.

## Models (Table 1)

| Model | API ID | Temperature | Max Tokens |
|-------|--------|-------------|------------|
| GPT-4o | gpt-4o | 0.3 | 500 |
| GPT-4o-mini | gpt-4o-mini | 0.3 | 500 |
| GPT-5 | gpt-5 | 0.3 | 500 |
| GPT-5-mini | gpt-5-mini | 0.3 | 500 |
| o1 | o1 | 0.3 | 500 |
| o1-mini | o1-mini | 0.3 | 500 |

## Retrieval Methods

- **BM25**: Lexical retrieval using BM25Okapi with Kazakh-aware tokenization (KazNLP morphological normalization + suffix stemming)
- **Vector**: Dense semantic retrieval using OpenAI `text-embedding-3-small` embeddings (1536d) with FAISS IndexFlatIP
- **Hybrid (Weighted RRF)**: Combines BM25 and vector rankings using the formula:
  ```
  score(d) = alpha / (k + rank_bm25(d)) + (1 - alpha) / (k + rank_vec(d))
  ```
  Parameters tuned via grid search: `k_rrf in {10, 20, 40, 60, 100}`, `alpha in {0.3, 0.4, 0.5, 0.6, 0.7}`

## Dataset

**Corpus**: [Textual Foundations of Justice: Kazakhstani Laws and Jurisprudence Dataset](https://doi.org/10.17632/jdpc5658nh.3) (Akhmetov et al., 2024). 12,886 legal texts in Kazakh, CC BY 4.0.

**Test set**: 100 legal questions covering constitutional, administrative, civil, criminal, labor/social, and tax/financial law. Each question has a gold reference answer and gold passage references.

## Evaluation

- **Retrieval metrics**: Recall@6, Precision@6, MRR, NDCG@6 (against gold passages)
- **Generation metric**: Answer accuracy (correct if factual content matches gold reference)
- **Statistical tests**: Cochran's Q (overall), McNemar (pairwise), with multiplicity correction

## Citation

If you use this code or data, please cite:

```
Kalzhanov N., Artykbay S., Kalzhan A. Development of the Retrieval-Augmented
Generation (RAG) System for the Kazakh Language Using Hybrid Information Methods.
```

## License

The corpus data (`kaz/`) is from the [Mendeley Data repository](https://doi.org/10.17632/jdpc5658nh.3) under CC BY 4.0 license.

