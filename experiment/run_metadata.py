"""
Collect exact model versions and environment metadata for reproducibility.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries."""
    versions = {}
    for pkg in ("openai", "numpy", "faiss", "tiktoken", "httpx"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def get_run_metadata(
    run_id: str,
    selected_models: Optional[List[Any]] = None,
    retrieval_methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build metadata dict for a run: exact model IDs, timestamp, library versions.
    For reproducibility and reviewer clarity.
    """
    from config import (
        LLM_MODELS,
        EMBEDDING_MODEL,
        TOP_K,
        N_FOLDS,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    # Exact LLM model identifiers (API IDs)
    if selected_models is not None:
        models_spec = [
            {"display_name": m.name, "api_id": m.api_id, "provider": m.provider}
            for m in selected_models
        ]
    else:
        models_spec = [
            {"display_name": m.name, "api_id": m.api_id, "provider": m.provider}
            for m in LLM_MODELS
        ]

    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
        "library_versions": get_library_versions(),
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "n_folds": N_FOLDS,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "llm_models": models_spec,
        "retrieval_methods": retrieval_methods or ["bm25", "vector", "hybrid"],
    }
    return metadata
