"""
Document chunker for Kazakh legal corpus.
Segments documents into fixed-size chunks with overlap.
"""

import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Generator
from dataclasses import dataclass, asdict
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CORPUS_DIR, CHUNKS_DIR, INDEX_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOKENIZER_MODEL
)


@dataclass
class Chunk:
    """Represents a document chunk."""
    doc_id: str
    chunk_id: int
    text: str
    token_count: int
    char_start: int
    char_end: int


class DocumentChunker:
    """Chunks documents using tiktoken tokenizer."""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        tokenizer_model: str = TOKENIZER_MODEL
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_model)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        return self.tokenizer.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(tokens)
    
    def chunk_text(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Document text
            doc_id: Document identifier
            
        Returns:
            List of Chunk objects
        """
        tokens = self.tokenize(text)
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return []
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < total_tokens:
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.detokenize(chunk_tokens)
            
            # Calculate character positions (approximate)
            char_start = len(self.detokenize(tokens[:start_idx]))
            char_end = char_start + len(chunk_text)
            
            chunk = Chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                char_start=char_start,
                char_end=char_end
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap
            chunk_id += 1
            
            # Avoid infinite loop for very small documents
            if end_idx == total_tokens:
                break
        
        return chunks
    
    def chunk_document(self, filepath: Path) -> List[Chunk]:
        """
        Read and chunk a single document.
        
        Args:
            filepath: Path to document file
            
        Returns:
            List of Chunk objects
        """
        try:
            text = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['utf-8-sig', 'cp1251', 'latin-1']:
                try:
                    text = filepath.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"Warning: Could not decode {filepath}")
                return []
        
        doc_id = filepath.name
        return self.chunk_text(text, doc_id)


def process_corpus(
    corpus_dir: Path = CORPUS_DIR,
    output_dir: Path = CHUNKS_DIR,
    batch_size: int = 1000
) -> Dict[str, any]:
    """
    Process entire corpus and save chunks.
    
    Args:
        corpus_dir: Directory containing documents
        output_dir: Directory to save chunks
        batch_size: Number of documents per batch file
        
    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chunker = DocumentChunker()
    
    # Get all document files
    doc_files = list(corpus_dir.glob("*.txt"))
    print(f"Found {len(doc_files)} documents")
    
    all_chunks: List[Dict] = []
    stats = {
        "total_documents": len(doc_files),
        "total_chunks": 0,
        "total_tokens": 0,
        "empty_documents": 0,
        "failed_documents": 0
    }
    
    batch_num = 0
    
    for doc_file in tqdm(doc_files, desc="Chunking documents"):
        chunks = chunker.chunk_document(doc_file)
        
        if not chunks:
            stats["empty_documents"] += 1
            continue
        
        for chunk in chunks:
            all_chunks.append(asdict(chunk))
            stats["total_tokens"] += chunk.token_count
        
        stats["total_chunks"] += len(chunks)
        
        # Save batch if reached batch size
        if len(all_chunks) >= batch_size * 10:  # ~10 chunks per doc average
            batch_file = output_dir / f"chunks_batch_{batch_num:04d}.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            print(f"Saved batch {batch_num} with {len(all_chunks)} chunks")
            all_chunks = []
            batch_num += 1
    
    # Save remaining chunks
    if all_chunks:
        batch_file = output_dir / f"chunks_batch_{batch_num:04d}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved final batch {batch_num} with {len(all_chunks)} chunks")
    
    # Save consolidated chunks file
    print("Creating consolidated chunks file...")
    all_chunks_consolidated = []
    for batch_file in sorted(output_dir.glob("chunks_batch_*.json")):
        with open(batch_file, 'r', encoding='utf-8') as f:
            all_chunks_consolidated.extend(json.load(f))
    
    consolidated_file = output_dir / "all_chunks.json"
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks_consolidated, f, ensure_ascii=False)
    
    # Create chunk index (doc_id -> chunk_ids mapping)
    chunk_index = {}
    for i, chunk in enumerate(all_chunks_consolidated):
        doc_id = chunk["doc_id"]
        if doc_id not in chunk_index:
            chunk_index[doc_id] = []
        chunk_index[doc_id].append(i)
    
    index_dir = INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = index_dir / "chunk_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_index, f, ensure_ascii=False)
    
    # Save statistics
    stats_file = output_dir / "chunking_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nChunking complete!")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average chunks per document: {stats['total_chunks'] / max(1, stats['total_documents'] - stats['empty_documents']):.2f}")
    print(f"Average tokens per chunk: {stats['total_tokens'] / max(1, stats['total_chunks']):.2f}")
    
    return stats


def load_chunks(chunks_dir: Path = CHUNKS_DIR) -> List[Dict]:
    """Load all chunks from consolidated file."""
    chunks_file = chunks_dir / "all_chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}. Run chunker first.")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_chunk_index(index_dir: Path = INDEX_DIR) -> Dict[str, List[int]]:
    """Load chunk index (doc_id -> chunk indices)."""
    index_file = index_dir / "chunk_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}. Run chunker first.")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    process_corpus()
