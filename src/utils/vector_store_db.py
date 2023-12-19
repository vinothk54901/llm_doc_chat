import streamlit
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from src.llm_providers.llm_provider_factory import LLMProviderFactory   

#Todo:
#Right now we are using FAISS as VectorStore and also locally which will get destroyed once the app got killed.
# In Future change as external DB for Persistence. (Qadrant)

def get_vector_store(text_chunks,llm_instance,st_session:streamlit):
    st_session.write("type:",type(text_chunks))
    # if llm_option == "OpenAI":
    #     embeddings = OpenAIEmbeddings()
    # elif llm_option == "HuggingFace":    
    #     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    #create_llm_instance = LLMProviderFactory.create_llm_instance(llm_option=llm_option)
    embeddings = llm_instance.embeddings()
    if isinstance(text_chunks[0],str):
        vectorstore = FAISS.from_texts(texts=text_chunks,embedding = embeddings)
        
    else:    
        vectorstore = FAISS.from_documents(documents=text_chunks,embedding = embeddings)
    return vectorstore
