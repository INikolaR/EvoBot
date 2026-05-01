from assistant.core.embedder_factory import EmbedderFactory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
import torch

class HFModelEmbedderFactory(EmbedderFactory):
    def __init__(self):
        pass
    def create_embedder(self, hf_model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> Embeddings:
        return HuggingFaceEmbeddings(
            model_name=hf_model_name,
            model_kwargs={
                "device": "cuda:0",
                "torch_dtype": torch.float16,
                "trust_remote_code": True
            },
            encode_kwargs={"normalize_embeddings": True, "batch_size": 1},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
