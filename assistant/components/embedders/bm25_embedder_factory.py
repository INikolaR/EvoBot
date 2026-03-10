from rag_experiments.core.embedder_factory import EmbedderFactory
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.embeddings import Embeddings
import numpy as np
from typing import List


class BM25Embeddings(Embeddings):
    def __init__(self, vectorizer: CountVectorizer, k1: float = 1.5, b: float = 0.75):
        self.vectorizer = vectorizer
        self.k1 = k1
        self.b = b
        self.idf_values = None
        self.avg_doc_length = None
        self.doc_term_matrices = None
        self.vocabulary_ = None

    def fit(self, texts: List[str]):
        self.doc_term_matrices = self.vectorizer.fit_transform(texts)
        self.vocabulary_ = self.vectorizer.vocabulary_
        n_docs, n_terms = self.doc_term_matrices.shape

        df = np.array((self.doc_term_matrices > 0).sum(axis=0)).flatten()
        self.idf_values = np.log((n_docs - df + 0.5) / (df + 0.5))

        doc_lengths = np.array(self.doc_term_matrices.sum(axis=1)).flatten()
        self.avg_doc_length = np.mean(doc_lengths)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.idf_values is None:
            self.fit(texts)

        term_freq = self.vectorizer.transform(texts)
        doc_lengths = np.array(term_freq.sum(axis=1)).flatten()

        bm25_scores = []
        for i in range(len(texts)):
            tf_row = term_freq[i].toarray().flatten()
            doc_len = doc_lengths[i] if doc_lengths[i] > 0 else 1.0

            numerator = tf_row * (self.k1 + 1)
            denominator = tf_row + self.k1 * (1 - self.b + self.b * self.avg_doc_length / doc_len)

            denominator = np.where(denominator == 0, 1e-10, denominator)

            scores = self.idf_values * (numerator / denominator)
            bm25_scores.append(scores.tolist())

        return bm25_scores

    def embed_query(self, text: str) -> List[float]:
        if self.idf_values is None:
            raise ValueError("Model must be fitted before embedding queries")

        query_vector = self.vectorizer.transform([text])
        query_tf = query_vector.toarray().flatten()

        query_length = float(query_vector.sum())

        query_len = query_length if query_length > 0 else 1.0

        numerator = query_tf * (self.k1 + 1)
        denominator = query_tf + self.k1 * (1 - self.b + self.b * self.avg_doc_length / query_len)

        denominator = np.where(denominator == 0, 1e-10, denominator)

        scores = self.idf_values * (numerator / denominator)
        return scores.tolist()


class BM25EmbedderFactory(EmbedderFactory):
    def __init__(self, k1: float = 1.5, b: float = 0.75, **count_kwargs):
        self.k1 = k1
        self.b = b
        self.count_kwargs = count_kwargs

    def create_embedder(self, **additional_params) -> Embeddings:
        params = {**self.count_kwargs, **additional_params}
        vectorizer = CountVectorizer(**params)
        return BM25Embeddings(vectorizer, k1=self.k1, b=self.b)