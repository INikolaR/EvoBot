from core.embedder_factory import EmbedderFactory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

class HFModelEmbedderFactory(EmbedderFactory):
    def __init__(self):
        pass
    def create_embedder(self, hf_model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> Embeddings:
        return HuggingFaceEmbeddings(model_name=hf_model_name)