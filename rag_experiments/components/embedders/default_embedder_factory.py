from rag_experiments.core.embedder_factory import EmbedderFactory
from langchain_core.embeddings import Embeddings
from typing import List

class DummyEmbeddings(Embeddings):
    def __init__(self, size: int = 384):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * self.size for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * self.size

class DefaultEmbedderFactory(EmbedderFactory):
    def __init__(self):
        pass
    def create_embedder(self) -> Embeddings:
        return DummyEmbeddings()
