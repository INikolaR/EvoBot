from abc import ABC, abstractmethod
from typing import List

class Chunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass
