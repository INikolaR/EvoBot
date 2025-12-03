from core.chunker import Chunker
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveCharacterChunker(Chunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int  = 100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)