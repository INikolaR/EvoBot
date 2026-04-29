from langchain_huggingface import HuggingFaceEmbeddings
from assistant.core.chunker import Chunker
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

class EmbeddingSemanticChunker(Chunker):
    def __init__(self, model_name: str, threshold_type: str = "percentile", threshold_amount: float = 95):
        self._model_name = model_name
        self._threshold_type = threshold_type
        self._threshold_amount = threshold_amount
        self._splitter = SemanticChunker(
            embeddings=HuggingFaceEmbeddings(model_name=model_name),
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_amount
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)
    
    def describe(self) -> str:
        return f"Semantic-model-{self._model_name}-ttype-{self._threshold_type}-tamount-{self._threshold_amount}"
