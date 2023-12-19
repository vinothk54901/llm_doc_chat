from src.llm_providers.open_ai import LLMOpenAI
from src.llm_providers.hugging_face import LLMHuggingFace

class LLMProviderFactory:
    @staticmethod
    def create_llm_instance(llm_option):
        if llm_option == "OpenAI":
            return LLMOpenAI()
        elif llm_option == "HuggingFace":
            return LLMHuggingFace()
        else:
            raise ValueError("Please choose a valid llm option")        

