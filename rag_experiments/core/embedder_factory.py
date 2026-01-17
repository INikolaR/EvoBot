from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

class EmbedderFactory(ABC):
    @abstractmethod
    def create_embedder(self, **kwargs) -> Embeddings:
        pass
