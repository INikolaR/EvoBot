from abc import ABC, abstractmethod
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

class RetrieverFactory(ABC):
    @abstractmethod
    def create_retriever(self, texts: List[str], embedder: Embeddings, **kwargs) -> VectorStoreRetriever:
        pass