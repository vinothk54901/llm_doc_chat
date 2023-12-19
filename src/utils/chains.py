from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import streamlit

#from langchain.chat_models import ChatOpenAI #emove
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import HuggingFaceHub #remove
#from src.llm_providers.llm_provider_factory import LLMProviderFactory   


system_message_prompt = SystemMessagePromptTemplate.from_template(
    #"Your name is Roboto, you are a nice virtual assistant. The context is:\n{context}\n The document that contains the answer is: \n{source}\n"
"""End every answer with "Have a nice day!". Use the following pieces of context to answer the users question. 
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        ----------------
        {context}\n """
)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
)


def get_conversation_chain(vectorstore,llm_instance,st_session:streamlit):

    llm = llm_instance.llm()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True,input_key='question', output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4, "include_metadata": True}
        ),
        memory=memory,
        get_chat_history=lambda h :h,
        combine_docs_chain_kwargs={
         "prompt": ChatPromptTemplate.from_messages([
             system_message_prompt,
             human_message_prompt,
         ]),
     },
        return_source_documents=True,
        verbose=True
    )
    st_session.write(conversation_chain)

    return conversation_chain
