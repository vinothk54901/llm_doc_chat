from src.llm_providers.llm_functions_base import LLMFunctions
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


class LLMOpenAI(LLMFunctions):
    def __init__(self):
        self.temperature=0
        self.model_name = "gpt-3.5-turbo"
        self.embeddings_model_name = "text-embedding-ada-002"

    def llm(self):
        return  ChatOpenAI(temperature=self.temperature,model_name=self.model_name)

    def embeddings(self):
        return OpenAIEmbeddings(model=self.embeddings_model_name) 

