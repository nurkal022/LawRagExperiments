"""
FAISS vector index builder for Kazakh legal corpus.
Uses OpenAI text-embedding-3-small for embeddings.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHUNKS_DIR, INDEX_DIR,
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE
)
from indexing.chunker import load_chunks

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingClient:
    """Client for generating embeddings using OpenAI API."""
    
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE
    ):
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Generating embeddings")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            # Handle empty texts
            batch = [t if t.strip() else " " for t in batch]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            numpy array of embedding (embedding_dim,)
        """
        if not query.strip():
            query = " "
        
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        
        return np.array(response.data[0].embedding, dtype=np.float32)


class FAISSIndex:
    """FAISS index wrapper with metadata."""
    
    def __init__(
        self,
        index: faiss.Index,
        chunk_ids: List[int],
        embedding_dim: int = EMBEDDING_DIM
    ):
        self.index = index
        self.chunk_ids = chunk_ids
        self.embedding_dim = embedding_dim
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[tuple]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Normalize query for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for not enough results
                results.append((self.chunk_ids[idx], float(score)))
        
        return results
    
    def save(self, filepath: Path):
        """Save index to file."""
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
        
        # Save metadata
        metadata = {
            'chunk_ids': self.chunk_ids,
            'embedding_dim': self.embedding_dim
        }
        with open(filepath.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"Saved FAISS index to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'FAISSIndex':
        """Load index from file."""
        # Load FAISS index
        index = faiss.read_index(str(filepath.with_suffix('.faiss')))
        
        # Load metadata
        with open(filepath.with_suffix('.meta.json'), 'r') as f:
            metadata = json.load(f)
        
        return cls(
            index=index,
            chunk_ids=metadata['chunk_ids'],
            embedding_dim=metadata['embedding_dim']
        )


def build_faiss_index(
    chunks: List[Dict],
    embedding_client: EmbeddingClient,
    embedding_dim: int = EMBEDDING_DIM
) -> FAISSIndex:
    """
    Build FAISS index from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        embedding_client: Client for generating embeddings
        embedding_dim: Dimension of embeddings
        
    Returns:
        FAISSIndex object
    """
    chunk_ids = list(range(len(chunks)))
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_client.embed_texts(texts)
    
    # Normalize embeddings for cosine similarity (using inner product)
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index (Inner Product = Cosine Similarity after normalization)
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    
    return FAISSIndex(
        index=index,
        chunk_ids=chunk_ids,
        embedding_dim=embedding_dim
    )


def build_index(
    chunks_dir: Path = CHUNKS_DIR,
    index_dir: Path = INDEX_DIR,
    save_embeddings: bool = True
) -> FAISSIndex:
    """
    Build FAISS index from corpus chunks.
    
    Args:
        chunks_dir: Directory containing chunks
        index_dir: Directory to save index
        save_embeddings: Whether to save raw embeddings
        
    Returns:
        FAISSIndex object
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    print("Loading chunks...")
    chunks = load_chunks(chunks_dir)
    print(f"Loaded {len(chunks)} chunks")
    
    # Initialize embedding client
    embedding_client = EmbeddingClient()
    
    # Build index
    faiss_index = build_faiss_index(chunks, embedding_client)
    
    # Save index
    index_path = index_dir / "faiss_index"
    faiss_index.save(index_path)
    
    # Optionally save raw embeddings for analysis
    if save_embeddings:
        print("Saving raw embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_client.embed_texts(texts, show_progress=False)
        np.save(index_dir / "embeddings.npy", embeddings)
    
    print("\nFAISS index built successfully!")
    print(f"  - Vectors: {faiss_index.index.ntotal}")
    print(f"  - Dimension: {faiss_index.embedding_dim}")
    
    return faiss_index


def load_faiss_index(index_dir: Path = INDEX_DIR) -> FAISSIndex:
    """Load FAISS index from directory."""
    index_path = index_dir / "faiss_index"
    return FAISSIndex.load(index_path)


if __name__ == "__main__":
    build_index()
