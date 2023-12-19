import streamlit as st
from dotenv import load_dotenv
from src.utils import data_preparation,vector_store_db,chains
from htmlTemplates import css, bot_template, user_template
from src.llm_providers.llm_provider_factory import LLMProviderFactory   


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    #st.success("\n I hope this helps." + "" if not response['source_documents'] else "\n  Source: " + str(list(map(lambda doc: f"{response['source_documents'].index(doc)+1}. '{doc.metadata['source']}' at page: {doc.metadata['page']}", response['source_documents']))))
    st.success( "" if not response['source_documents'] else "\n I hope this helps."+"\n  Source: " + str(list(map(lambda doc: f"{response['source_documents'].index(doc)+1}. '{doc.metadata['source']}'", response['source_documents']))))


def main():
    load_dotenv()
    st.set_page_config(page_title='Ask Me Anything - Document',page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
     
    st.image('App_images/robot1.png', width=78)
    st.header("Ask Me Anything - Directory 📂")
    st.write('\n')
    st.write("3. Ask Questions form your documents")
    user_question = st.text_input(label="qa_dir_question",label_visibility="hidden")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload your PDF documents,Choose LLM and click on 'Process'")
        st.write("1.Place Your Documents Directory")
        directory_path = st.text_input('Directory Path')        
        st.write("2.Choose LLM")
        llm_option = st.selectbox(
        '(OpenAI or HuggingFace)',
        ('OpenAI', 'HuggingFace'))

        st.write('You selected:', llm_option)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get text from PDF's
                docs = data_preparation.get_docs_from_directory(directory_path,st) 
                st.write("dataprepdocs: ",docs)
                #Get      Text Chunks
                text_chunks = data_preparation.get_text_chunks_recursive(docs,st)
                
                #Create instance of OPenAI or HuggingFace based on the option chosen in LLM
                llm_instance = LLMProviderFactory.create_llm_instance(llm_option=llm_option)

                #Create Vector store with embeddings
                vector_store = vector_store_db.get_vector_store(docs,llm_instance,st)

                # create conversation chain
                st.session_state.conversation = chains.get_conversation_chain(
                    vector_store,llm_instance,st)

main()                