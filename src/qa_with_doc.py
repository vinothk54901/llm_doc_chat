import streamlit as st
from dotenv import load_dotenv
from src.utils import data_preparation,vector_store_db,chains
from src.llm_providers.llm_provider_factory import LLMProviderFactory   
from htmlTemplates import css, bot_template, user_template


from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# for k in st.session_state.keys():
#     st.session_state['conversation'] = None
#     st.session_state['chat_history'] = None
  
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
    st.write(response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    st.success("\n I hope this helps.")


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
    user_question = st.text_input(label="doc_chat",label_visibility="hidden")
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
                raw_text = data_preparation.get_pdf_text(pdf_docs)

                #Get     Text Chunks
                raw_text = data_preparation.get_text_chunks(raw_text,st)

                            #Create instance of OPenAI or HuggingFace based on the option chosen in LLM
                llm_instance = LLMProviderFactory.create_llm_instance(llm_option=llm_option)            

                #Create Vector store with embeddings
                vector_store = vector_store_db.get_vector_store(raw_text,llm_instance,st)

                # create conversation chain
                st.session_state.conversation = chains.get_conversation_chain(
                    vector_store,llm_instance,st)

main()                