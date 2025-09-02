from typing import List
import numpy as np
# at top of file
from functools import lru_cache

@lru_cache(maxsize=2)
def get_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from .embeddings import EmbeddingModel  # avoid circular import
    return EmbeddingModel(name)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer=None
class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        if SentenceTransformer is None: raise RuntimeError('sentence-transformers not installed')
        self.model=SentenceTransformer(model_name)
    def encode(self, texts:List[str]):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
