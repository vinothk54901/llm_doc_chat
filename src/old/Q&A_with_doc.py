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


from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

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


def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text        

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=800,
                                          chunk_overlap=200,
                                          length_function=len,

    )
    chunks = text_splitter.split_text(text)
    return chunks
#Todo:
#Right now we are using FAISS as VectorStore and also locally which will get destroyed once the app got killed.
# In Future change as external DB for Persistence. (Qadrant)

def get_vector_store(text_chunks,llm_option):
    if llm_option == "OpenAI":
        embeddings = OpenAIEmbeddings()
    elif llm_option == "HuggingFace":    
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore,llm_option):
    if llm_option == "OpenAI":
        llm = ChatOpenAI()
    elif llm_option == "HuggingFace":    
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
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
    st.success("\n I hope this helps." + "" if not response['source_documents'] else "\n Refer: " +str(list(map(lambda x: x.metadata['source'], response['source_documents']))))


def main():
    load_dotenv()
    st.set_page_config(page_title='Ask Me Anything - Document',page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
     
    st.image('App_images/robot1.png', width=78)
    st.header("Ask Me Anything - Document :books:")
    st.write('\n')
    st.write("3. Ask Questions form your documents")
    user_question = st.text_input("")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.write("Upload your PDF documents,Choose LLM and click on 'Process'")
        st.write("1. Browse Your documents")
        pdf_docs = st.file_uploader("",accept_multiple_files= True)
        st.write("2. Choose LLM")
        llm_option = st.selectbox(
        '(OpenAI or HuggingFace)',
        ('OpenAI', 'HuggingFace'))

        st.write('You selected:', llm_option)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get text from PDF's
                raw_text = get_pdf_text(pdf_docs)

                #Get     Text Chunks
                text_chunks = get_text_chunks(raw_text)

                #Create Vector store with embeddings
                vector_store = get_vector_store(text_chunks,llm_option)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vector_store,llm_option)

main()                