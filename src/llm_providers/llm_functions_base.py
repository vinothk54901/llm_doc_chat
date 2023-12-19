from abc import ABC, abstractmethod


class LLMFunctions(ABC):
    @abstractmethod
    def llm(self):
        pass
    @abstractmethod
    def embeddings(self):
        pass