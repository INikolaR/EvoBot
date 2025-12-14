from core.retriever_factory import RetrieverFactory
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Chroma

class ChromaRetrieverFactory(RetrieverFactory):
    def __init__(self):
        pass

    def create_retriever(self, texts: List[str], embedder: Embeddings, search_kwargs: dict, search_type: str) -> VectorStoreRetriever:
        db = Chroma.from_texts(texts, embedder, persist_directory="./chroma_db")
        retriever = db.as_retriever(
            search_kwargs=search_kwargs,
            search_type=search_type
        )
        return retriever
