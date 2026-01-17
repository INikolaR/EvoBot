from abc import ABC, abstractmethod

class Generator(ABC):
    @abstractmethod
    def __call__(self, input_data: str) -> str:
        pass