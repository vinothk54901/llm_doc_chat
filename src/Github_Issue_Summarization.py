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
from langchain.document_loaders import GitHubIssuesLoader
import datetime


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
 

def generate_response(texts):
    # Instantiate the LLM model
    #llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.2, "max_length":1024},huggingfacehub_api_token=openai_api_key)
    # Split text
    #text_splitter = CharacterTextSplitter()
    #texts = text_splitter.split_documents(txt)
    # Create multiple documents
    #docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    st.write(chain)
        
    return chain.run(texts)

# Page title
st.set_page_config(page_title='Summarization - Git Issue :books:')
st.title('Summarization - Git Issue :books:')

# Initialize date and time variables
date = None
time = None
result = []
with st.form("GithubIssueForm"):
        st.write("Ask Me Anything - GithubIssues :earth_asia: ")
        st.write("Enter the Githhub API URL & it details ,Choose LLM and click on 'Process'")
        # Load data from the specified URL
        repo = st.text_input(label="Organisation/repo:red[*]",placeholder="Ex: langchain-ai/langchain",) 
        st.write(repo)
        ACCESS_TOKEN = st.text_input(label="Personal_access_token:red[*]",type="password")
        # st.write("2. Choose LLM")
        llm_option = st.selectbox(
        '(OpenAI or HuggingFace):red[*]',
        ('OpenAI', 'HuggingFace'))

        # st.write('You selected:', llm_option)
        # st.text_input(label="github_api_url")
        openai_api_key = st.text_input('OpenAI/HuggingFace API Key:red[*]', type = 'password')
        col1, col2 = st.columns(2)
        with col1:
            include_pr = st.radio(label="Include PR:red[*]",options=(True,False),horizontal=True )
            assignee = st.text_input(label="Assigned To",placeholder="* or none or assinged to name",help="""Filter on assigned user. Pass 'none' for no user and '*' for any user.""")
            dateinput = st.date_input("Select Date")
            sort_on = st.radio(label="Sort On",options=(None,"created", "updated", "comments"))
        with col2:
            state = st.radio(label= "State:",options=("open", "closed", "all"),horizontal=True )
            creator=st.text_input(label="Creator",help="Filter on the user that created the issue.")
            timeinput=st.time_input("Select Time")
            sort_by = st.radio(label="Sort By:",options=(None,"asc", "desc"))
        if st.form_submit_button(label="Submit",use_container_width=True):
            with st.spinner("Processing..."):
                loader = GitHubIssuesLoader(
                repo=repo,
                access_token=ACCESS_TOKEN, 
                include_prs=include_pr,
                state = None if not state else state,
                assignee= None if not assignee else assignee,
                creator= None if not creator else creator,
                since = None if not date else datetime.datetime.combine(dateinput, timeinput).isoformat("T") + "Z",
                sort = None if not sort_on else sort_on,
                direction = None if not sort_by else sort_by,
                )

                docs = loader.load_and_split()
                print(st.write(docs))
                st.write(docs)
                response = generate_response(docs)
                result.append(response)
                # for doc in docs:
                #    response = generate_response(doc)
                #    result.append(response)
                st.write(result)

if len(result):
    st.info(response)
    # play_audio = st.button('Generate and Play Audio')
    # if play_audio:
    generate_audio(response) 
                