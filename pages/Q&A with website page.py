import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA


# def get_pdf_text(pdf_docs):
#     text =""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() 
#     return text        

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=800,
                                          chunk_overlap=200,
                                          length_function=len,

    )
    chunks = text_splitter.split_documents(text)
    return chunks
#Todo:
#Right now we are using FAISS as VectorStore and also locally which will get destroyed once the app got killed.
# In Future change as external DB for Persistence. (Qadrant)

def get_vector_store(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_documents(documents=text_chunks,embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title='Altimetrik Ask Me Anything From PDF passed',page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Altimetrik's Ask Me Anything From PDF passed :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        #pdf_docs = st.file_uploader("Upload your PDF documents and click on 'Process'",accept_multiple_files= True)
        # Load data from the specified URL
        url = st.text_input("Insert The website URL")

        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get text from PDF's
                # raw_text = get_pdf_text(pdf_docs)
                loader = WebBaseLoader(url)
                data = loader.load()
                #Get Text Chunks
                text_chunks = get_text_chunks(data)

                #Create Vector store with embeddings
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vector_store)

main()                