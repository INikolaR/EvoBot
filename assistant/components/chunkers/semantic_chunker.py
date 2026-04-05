from assistant.core.chunker import Chunker
from typing import List
from langchain_text_splitters import SemanticChunker
from langchain_core.embeddings import Embeddings

class EmbeddingSemanticChunker(Chunker):
    def __init__(self, embeddings: Embeddings, threshold_type: str = "percentile", threshold_amount: float = 95):
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_amount
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
