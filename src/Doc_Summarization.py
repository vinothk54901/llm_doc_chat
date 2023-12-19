import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.llms import HuggingFaceHub
from gtts import gTTS
#from playsound import playsound
import io
#from pydub import AudioSegment
#from pydub.playback import play
from io import BytesIO
#import base64

def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text    

def generate_audio(text,lang='en'):
    # Initialize gTTS object with the text and language (en for English)
    # lang_code = {"English":"en", "Japanese":"ja"}
    # selected = st.radio("Language",["English", "Japanese"])
    #tts = gTTS(text, lang=lang_code[selected])
    sound_file=BytesIO()
    tts = gTTS(text, lang=lang)
    # tempname = "./output.mp3"
    # tts.save(tempname)
    # autoplay_audio(tempname)
    tts.write_to_fp(sound_file)
    st.audio(sound_file)
    # audio_stream = BytesIO()
    # tts.write_to_fp(audio_stream)
    # audio_stream.seek(0)
    # sound = AudioSegment.from_mp3(audio_stream)
    # play(sound)
    # tts.save('./output.mp3')
    # playsound('./output.mp3')
    # 




def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
 

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
st.set_page_config(page_title='Summarization - Documentation :books:')
st.title('Summarization - Documentation :books:')

# Text input
#txt_input = st.text_area('Enter your text', '', height=200)
raw_text=""
st.subheader("Your documents")
pdf_docs = st.file_uploader("Upload your PDF documents and click on 'Submit'",accept_multiple_files= True)
       

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
            #del openai_api_key

    

if len(result):
    st.info(response)
    # play_audio = st.button('Generate and Play Audio')
    # if play_audio:
    generate_audio(response) 
                