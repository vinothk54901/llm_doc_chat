import streamlit
from langchain.document_loaders import PyPDFDirectoryLoader,DirectoryLoader,TextLoader,Docx2txtLoader,UnstructuredWordDocumentLoader 
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import GitHubIssuesLoader
import uuid
from langchain_core.documents import Document
from typing import List



def get_docs_from_directory(directory_path:str,st_session:streamlit):
    extensions = ['pdf','txt','docx']
    documents= None
    for ext in extensions:
        try:
            st_session.write(ext)
            loader = None
            loaded_documents: List[Document] = []
            glob_pattern = f'**/*.{ext}'
            if ext =="pdf":
                st_session.write("entered pdf")
                loader = PyPDFDirectoryLoader(directory_path, glob="**/[!.]*.pdf",recursive =True)
            elif ext == "docx":
                loader = DirectoryLoader(glob=glob_pattern,loader_cls=Docx2txtLoader,path=directory_path,recursive=True)
                # loader = UnstructuredWordDocumentLoader(file_path=) 
            elif ext == "txt":
                loader = DirectoryLoader(glob=glob_pattern,loader_cls=TextLoader,path=directory_path,recursive=True)

            docs = loader.load() if callable(loader.load) else []
            loaded_documents.extend(docs)
            st_session.write("loadeddocs:",loaded_documents)

            # if loaded_documents:
            #     # file_type_counts[ext] = len(loaded_documents)
            #     for doc in loaded_documents:
            #         # file_path = doc.metadata['source']
            #         # relative_path = os.path.relpath(file_path, repo_path)
            #         file_id = str(uuid.uuid4())
            #         # doc.metadata['source'] = relative_path
            #         doc.metadata['file_id'] = file_id

            #         documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            st_session.write(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue            
    # st_session.write(docs)
    return loaded_documents  

def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text        


def get_text_chunks_recursive(docs,st_session:streamlit):
    text_splitter = RecursiveCharacterTextSplitter(
                                          chunk_size=200,
                                          chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(docs)
    st_session.write(chunks)
    return chunks

def get_text_chunks(raw_text,st_session:streamlit):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=800,
                                          chunk_overlap=200,
                                          length_function=len,

    )
    chunks = text_splitter.split_text(raw_text)
    st_session.write(chunks)
    return chunks

def get_text_from_docs(raw_docs,st_session:streamlit):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=800,
                                          chunk_overlap=200,
                                          length_function=len,

    )
    chunks = text_splitter.split_documents(raw_docs)
    st_session.write(chunks)
    return chunks

def get_all_links(url,st_session:streamlit):
    response = requests.get(url)
    bsoup = BeautifulSoup(response.text,'html.parser')
    alllinks = bsoup.findAll('a')
    st_session.write(alllinks)
    Links=[]
    for link in alllinks:
        href_link = link.get('href')
        if href_link.startswith('https://') or href_link.startswith('https://'):
            if href_link not in Links:
                Links.append(link.get('href'))
    return Links            

def get_data_from_url(url,st_session:streamlit):
    st_session.write(url)
    all_url_list = get_all_links(url,st_session)
    st_session.write(all_url_list)
    st_session.info("Out of all, only 5 taken for demo")
    st_session.write(all_url_list[:5])
    loader = WebBaseLoader(all_url_list[:5])
    data = loader.load()
    return data

def get_data_from_github_issues(repo:str,access_token:str,st_session:streamlit):
    loader = GitHubIssuesLoader(
                repo=repo,
                access_token=access_token, 
                )

    data = loader.load()
    st_session.write(data)
    return data