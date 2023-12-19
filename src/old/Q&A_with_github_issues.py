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
from langchain.document_loaders import GitHubIssuesLoader


from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.chains import RetrievalQA

import requests
from bs4 import BeautifulSoup
       
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_message_prompt = SystemMessagePromptTemplate.from_template(
    #"Your name is Roboto, you are a nice virtual assistant. The context is:\n{context}\n The document that contains the answer is: \n{source}\n"
"""End every answer with "Have a nice day!". Use the following pieces of context to answer the users question. 
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        You read the metadata and count all the issues and solution given.
        ----------------
        {context}\n"""
)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
)

def get_all_links(url):
    response = requests.get(url)
    bsoup = BeautifulSoup(response.text,'html.parser')
    alllinks = bsoup.findAll('a')
    Links=[]
    for link in alllinks:
        href_link = link.get('href')
        if href_link.startswith('https://') or href_link.startswith('https://'):
            if href_link not in Links:
                Links.append(link.get('href'))
    return Links            

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



def get_vector_store(text_chunks,llm_option):
    if llm_option == "OpenAI":
        embeddings = OpenAIEmbeddings()
    elif llm_option == "HuggingFace":    
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_documents(documents=text_chunks,embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore,llm_option):
    if llm_option == "OpenAI":
        llm = ChatOpenAI()
    elif llm_option == "HuggingFace":    
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1024})
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
    st.write(conversation_chain)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history'] 
    st.write(response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace( 
                "{{MSG}}", message.content), unsafe_allow_html=True)
    st.success("\n I hope this helps." + "" if not response['source_documents'] else "\n Refer: " +str(list(map(lambda x: x.metadata['url'], response['source_documents']))))

def main():
    load_dotenv()
    st.set_page_config(page_title='Ask Me Anything - WebPage',page_icon=":earth_asia:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.image('App_images/robot-icon.jpg', width=78)
    st.header("Ask Me Anything - WebPage :earth_asia: ")
    st.write("")
    st.write("3. Ask a question from a webpage:")
    user_question = st.text_input(" ")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.write("Ask Me Anything - GithubIssues :earth_asia: ")
        st.write("Enter the Githhub API URL & it details ,Choose LLM and click on 'Process'")
        # Load data from the specified URL
        repo = st.text_input(label="Repo",placeholder="langchain-ai/langchain",) 
        st.write(repo)
        ACCESS_TOKEN = st.text_input(label="Personal_access_token",type="password")
        st.write("2. Choose LLM")
        llm_option = st.selectbox(
        '(OpenAI or HuggingFace)',
        ('OpenAI', 'HuggingFace'))

        st.write('You selected:', llm_option)
        if st.button("Process"):
            with st.spinner("Processing..."):
                loader = GitHubIssuesLoader(
                repo=repo,
                access_token=ACCESS_TOKEN, 
                )

                data = loader.load()
                st.write(data)
                #data ="data"
                #Get Text Chunks
                text_chunks = get_text_chunks(data)
                st.write(text_chunks)

                #Create Vector store with embeddings
                vector_store = get_vector_store(text_chunks,llm_option)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vector_store,llm_option)

main()                