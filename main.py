import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from model import csv_handler, pdf_handler
from langchain_helper import final_result

# Created a Main title and sidebar title
st.title("File Interact Bot")
st.sidebar.title("File path")

# Created an option for file type
option = st.sidebar.selectbox('Please Select your file type',('CSV', 'PDF'), index=None)

if option == 'CSV':
    csv_handler()
if option == 'PDF':
    pdf_handler()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = final_result(prompt)
    print(response)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response['result'])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['result']})
    