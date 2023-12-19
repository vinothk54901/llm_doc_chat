import streamlit as st
from dotenv import load_dotenv
from src.llm_providers.llm_provider_factory import LLMProviderFactory   
from src.utils import data_preparation,vector_store_db,chains

from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

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

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.write(response['source_documents'])
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
    user_question = st.text_input(label="qa_web_question",label_visibility="hidden")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.write("Ask Me Anything - WebPage :earth_asia: ")
        st.write("Enter the WebPage URL,Choose LLM and click on 'Process'")
        st.write("1. Enter Web Page URL")
        # Load data from the specified URL
        url = st.text_input(label="url",label_visibility='hidden')
        st.write("2. Choose LLM")
        llm_option = st.selectbox(
        '(OpenAI or HuggingFace)',
        ('OpenAI', 'HuggingFace'))

        st.write('You selected:', llm_option)


        if st.button("Process"):
            with st.spinner("Processing..."):

                #Get data from the webpage url
                raw_docs = data_preparation.get_data_from_url(url,st)

                #Get Text Chunks
                text_chunks = data_preparation.get_text_from_docs(raw_docs,st)

               
                #Create instance of OPenAI or HuggingFace based on the option chosen in LLM
                llm_instance = LLMProviderFactory.create_llm_instance(llm_option=llm_option)

                #Create Vector store with embeddings
                vector_store = vector_store_db.get_vector_store(text_chunks,llm_instance,st)

                # create conversation chain
                st.session_state.conversation = chains.get_conversation_chain(
                    vector_store,llm_instance,st)

main()                