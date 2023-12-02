import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.llms import HuggingFaceHub
from io import BytesIO
from gtts import gTTS


def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text    

def generate_audio(text,lang='en'):
    sound_file=BytesIO()
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(sound_file)
    st.audio(sound_file)



def generate_response(txt):
    # Instantiate the LLM model
    #llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512},huggingfacehub_api_token=openai_api_key)
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Text input
#txt_input = st.text_area('Enter your text', '', height=200)
raw_text=""
st.subheader("Your documents")
pdf_docs = st.file_uploader("Upload your PDF documents and click on 'Process'",accept_multiple_files= True)
       

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type = 'password')
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('hf'):
        with st.spinner('Calculating...'):
            raw_text = get_pdf_text(pdf_docs)
            response = generate_response(raw_text)
            result.append(response)
            del openai_api_key
    

if len(result):
    st.info(response)
    generate_audio(response) 
