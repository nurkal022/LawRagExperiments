"""
BM25 index builder for Kazakh legal corpus.
Supports both KazNLP morphological tokenization and whitespace tokenization (ablation).
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_DIR, INDEX_DIR
from indexing.chunker import load_chunks

# Try to import KazNLP, fall back to basic tokenization if not available
try:
    from kaznlp.tokenize import Tokenizer as KazTokenizer
    from kaznlp.morphology import Analyzer as KazAnalyzer
    KAZNLP_AVAILABLE = True
except ImportError:
    KAZNLP_AVAILABLE = False
    print("Warning: kaznlp not installed. Using fallback Kazakh tokenizer.")
    print("Install from: https://github.com/nlacslab/kaznlp")

from rank_bm25 import BM25Okapi


class KazakhTokenizer:
    """
    Kazakh-aware tokenizer with morphological normalization.
    Falls back to rule-based tokenization if KazNLP is not available.
    """
    
    # Kazakh-specific characters and patterns
    KAZAKH_LETTERS = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
    KAZAKH_LETTERS_UPPER = KAZAKH_LETTERS.upper()
    ALL_LETTERS = KAZAKH_LETTERS + KAZAKH_LETTERS_UPPER + "a-zA-Z"
    
    # Common Kazakh suffixes for basic stemming
    KAZAKH_SUFFIXES = [
        # Plural
        'лар', 'лер', 'дар', 'дер', 'тар', 'тер',
        # Case endings
        'ның', 'нің', 'дың', 'дің', 'тың', 'тің',
        'ға', 'ге', 'қа', 'ке', 'на', 'не',
        'ды', 'ді', 'ты', 'ті', 'ны', 'ні',
        'дан', 'ден', 'тан', 'тен', 'нан', 'нен',
        'да', 'де', 'та', 'те',
        # Possessive
        'ым', 'ім', 'ың', 'ің', 'ы', 'і', 'сы', 'сі',
        'мыз', 'міз', 'ыңыз', 'іңіз',
        # Verb endings
        'ады', 'еді', 'йды', 'йді',
        'ған', 'ген', 'қан', 'кен',
        'атын', 'етін', 'йтын', 'йтін',
    ]
    
    def __init__(self, use_kaznlp: bool = True):
        self.use_kaznlp = use_kaznlp and KAZNLP_AVAILABLE
        
        if self.use_kaznlp:
            try:
                self.tokenizer = KazTokenizer()
                self.analyzer = KazAnalyzer()
            except Exception as e:
                print(f"Warning: KazNLP initialization failed: {e}")
                self.use_kaznlp = False
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization: lowercase, split on non-letters, filter short tokens."""
        text = text.lower()
        # Split on non-letter characters
        tokens = re.split(f'[^{self.KAZAKH_LETTERS}0-9]+', text)
        # Filter empty and very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        return tokens
    
    def _basic_stem(self, token: str) -> str:
        """Basic suffix stripping for Kazakh."""
        # Sort suffixes by length (longest first) to avoid partial matches
        for suffix in sorted(self.KAZAKH_SUFFIXES, key=len, reverse=True):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[:-len(suffix)]
        return token
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using KazNLP or fallback method.
        Returns list of normalized tokens.
        """
        if self.use_kaznlp:
            try:
                # Use KazNLP tokenizer
                tokens = self.tokenizer.tokenize(text)
                # Normalize using morphological analyzer
                normalized = []
                for token in tokens:
                    analysis = self.analyzer.analyze(token)
                    if analysis:
                        # Use lemma if available
                        normalized.append(analysis[0].get('lemma', token.lower()))
                    else:
                        normalized.append(token.lower())
                return [t for t in normalized if len(t) > 1]
            except Exception:
                # Fall back to basic tokenization
                pass
        
        # Basic tokenization with stemming
        tokens = self._basic_tokenize(text)
        return [self._basic_stem(t) for t in tokens]


class WhitespaceTokenizer:
    """Simple whitespace tokenizer for ablation study."""
    
    def tokenize(self, text: str) -> List[str]:
        """Split on whitespace and lowercase."""
        tokens = text.lower().split()
        # Remove punctuation from tokens
        tokens = [re.sub(r'[^\w]', '', t) for t in tokens]
        return [t for t in tokens if len(t) > 1]


@dataclass
class BM25Index:
    """BM25 index with metadata."""
    bm25: BM25Okapi
    chunk_ids: List[int]  # Global chunk indices
    tokenizer_type: str  # 'kaznlp' or 'whitespace'
    corpus_size: int


def build_bm25_index(
    chunks: List[Dict],
    tokenizer: Callable[[str], List[str]],
    tokenizer_type: str,
    show_progress: bool = True
) -> BM25Index:
    """
    Build BM25 index from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        tokenizer: Function that tokenizes text
        tokenizer_type: Name of tokenizer for metadata
        show_progress: Show progress bar
        
    Returns:
        BM25Index object
    """
    chunk_ids = list(range(len(chunks)))
    
    # Tokenize all chunks
    tokenized_corpus = []
    iterator = tqdm(chunks, desc=f"Tokenizing ({tokenizer_type})") if show_progress else chunks
    
    for chunk in iterator:
        tokens = tokenizer(chunk['text'])
        tokenized_corpus.append(tokens)
    
    # Build BM25 index
    print(f"Building BM25 index ({tokenizer_type})...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    return BM25Index(
        bm25=bm25,
        chunk_ids=chunk_ids,
        tokenizer_type=tokenizer_type,
        corpus_size=len(chunks)
    )


def save_bm25_index(index: BM25Index, filepath: Path):
    """Save BM25 index to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(index, f)
    print(f"Saved BM25 index to {filepath}")


def load_bm25_index(filepath: Path) -> BM25Index:
    """Load BM25 index from file."""
    import sys
    # Ensure this module is in the namespace for pickle
    current_module = sys.modules[__name__]
    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'BM25Index':
                return BM25Index
            return super().find_class(module, name)
    
    with open(filepath, 'rb') as f:
        return CustomUnpickler(f).load()


def build_all_indices(
    chunks_dir: Path = CHUNKS_DIR,
    index_dir: Path = INDEX_DIR
) -> Dict[str, BM25Index]:
    """
    Build both KazNLP and whitespace BM25 indices.
    
    Returns:
        Dictionary with 'kaznlp' and 'whitespace' indices
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    print("Loading chunks...")
    chunks = load_chunks(chunks_dir)
    print(f"Loaded {len(chunks)} chunks")
    
    indices = {}
    
    # Build KazNLP index (main baseline)
    kaz_tokenizer = KazakhTokenizer(use_kaznlp=True)
    indices['kaznlp'] = build_bm25_index(
        chunks,
        kaz_tokenizer.tokenize,
        'kaznlp'
    )
    save_bm25_index(indices['kaznlp'], index_dir / "bm25_kaznlp.pkl")
    
    # Build whitespace index (ablation)
    ws_tokenizer = WhitespaceTokenizer()
    indices['whitespace'] = build_bm25_index(
        chunks,
        ws_tokenizer.tokenize,
        'whitespace'
    )
    save_bm25_index(indices['whitespace'], index_dir / "bm25_whitespace.pkl")
    
    # Save tokenizer instances for query-time use
    tokenizers = {
        'kaznlp': kaz_tokenizer,
        'whitespace': ws_tokenizer
    }
    with open(index_dir / "tokenizers.pkl", 'wb') as f:
        pickle.dump(tokenizers, f)
    
    print("\nBM25 indices built successfully!")
    print(f"  - kaznlp: {indices['kaznlp'].corpus_size} documents")
    print(f"  - whitespace: {indices['whitespace'].corpus_size} documents")
    
    return indices


if __name__ == "__main__":
    build_all_indices()
