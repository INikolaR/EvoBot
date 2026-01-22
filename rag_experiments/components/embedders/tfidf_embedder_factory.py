from rag_experiments.core.embedder_factory import EmbedderFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.embeddings import Embeddings
import numpy as np
from typing import List


class TfidfEmbeddings(Embeddings):
    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(texts)
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return embedding.tolist()


class TfidfEmbedderFactory(EmbedderFactory):
    def __init__(self, **tfidf_kwargs):
        self.tfidf_kwargs = tfidf_kwargs

    def create_embedder(self, **additional_params) -> Embeddings:
        params = {**self.tfidf_kwargs, **additional_params}
        vectorizer = TfidfVectorizer(**params)
        return TfidfEmbeddings(vectorizer)