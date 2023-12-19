from src.llm_providers.llm_functions_base import LLMFunctions
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings


class LLMHuggingFace(LLMFunctions):
    def __init__(self):
        self.llm_model_repo_id ="google/flan-t5-xxl"
        self.model_kwargs = {"temperature":0.5, "max_length":2048}
        self.embeddings_model_name = "hkunlp/instructor-base"

    def llm(self):
        return HuggingFaceHub(repo_id=self.llm_model_repo_id, model_kwargs=self.model_kwargs)


    def embeddings(self):
        return HuggingFaceInstructEmbeddings(model_name=self.embeddings_model_name)
    

    