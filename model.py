import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS



paths = []
DB_FAISS_PATH = "vectorstores/db_faiss"
def create_vector_db(texts, embeddings):
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

def csv_handler():
    paths = []  # Initialize the list to store file paths
    for i in range(1):
        url = st.sidebar.text_input(f"File Path")
        paths.append(url)

    process_url_clicked = st.sidebar.button("Process File")

    class CustomCSVLoader(CSVLoader):
        def __init__(self, file_path, encoding='utf-8'):
            super().__init__(file_path)
            self.encoding = encoding

        def load(self):
            try:
                with open(self.file_path, 'r', encoding=self.encoding) as csvfile:
                    return self._CSVLoader__read_file(csvfile)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None  # Return None to indicate an error

    loaded_data = []  # Initialize the list to store loaded data

    if process_url_clicked:
        for csv_path in paths:
            loader = CustomCSVLoader(csv_path)
            data = loader.load()

            if data is not None:
                loaded_data.append(data)

        if loaded_data:
            # Using list comprehension and the + operator
            documents = [item for sublist in loaded_data for item in sublist]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                              model_kwargs={'device': 'cpu'})
            create_vector_db(texts, embeddings)

def pdf_handler():
    for i in range(1):
        url = st.sidebar.text_input(f"File Path {i+1}")
        paths.append(url)
    process_url_clicked = st.sidebar.button("Process File")

     # Create an empty list to store loaded data
    loaded_data = [] 
    # Loop through each CSV path and load the data
    if process_url_clicked:
        for pdf_path in paths:
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
            loaded_data.append(data)   
        docs = loaded_data
        # Using list comprehension and the + operator
        documents = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
        create_vector_db(texts, embeddings)