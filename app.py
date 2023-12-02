
import streamlit as st
from st_pages import show_pages_from_config, add_page_title

st.set_page_config(
    page_title="ASK ME ANYTHING",
    page_icon="👋",
)


# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

st.write("# Welcome to ASK ME ANYTHING! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ASK ME ANYTHING is an LLM powered app Which can do below features

    ### 1. Q&A chat from

    - Document 

        You can browse and Upload files from which you can ASK Questions.

        You can enter the directory where all your files are present and from which you can ASK Questions  
    
    - WebPage URL
    
        You can insert the webpage url, which extracts all the parent link and child link and from which you can Ask Questions.
    
    ### 2. Summarization
    
    - Document Summarization
    
        You can browse and Upload files to summarize and also Audio will be generated which you can listen the Summarized text.
    
    - Text Summarization
    
        You can paste your text and can get the summary.
    
        
    **👈 Select a demo from the sidebar** to see some examples
    of what ASK ME ANYTHING can do!
"""
)
