import streamlit as st

st.set_page_config(
    page_title="ASK ME ANYTHING",
    page_icon="👋",
)

st.write("# Welcome to ASK ME ANYTHING! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ASK ME ANYTHING is an LLM powered app for doing PDF doc chat and PDF summarization.
    
    **👈 Select a demo from the sidebar** to see some examples
    of what ASK ME ANYTHING can do!
"""
)
