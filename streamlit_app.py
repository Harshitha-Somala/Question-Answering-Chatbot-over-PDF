import streamlit as st
import pandas as pd
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
from langchain.embeddings import HuggingFaceHubEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# global variable
files_loaded = None


def get_text_docs(file):
    # save the file temporarily
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(file.read())
    
    # Load the PDF into pages/documents
    loader = PyPDFLoader(temp_path)
    pages =  loader.load_and_split()
    return pages

def get_text_chunks(docs):
    # Make chunks
    text_splitter = CharacterTextSplitter(        
        separator = ' ',
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_vector_space(chunks):
    # Load API Key
    load_dotenv()

    # Load Embeddings
    repo_id = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceHubEmbeddings(
        repo_id= repo_id
    )

    # Load Vector space into Chroma db
    db = Chroma.from_documents(chunks, hf)
    return db

def generate_response(files):

    # PDF to documents
    docs = get_text_docs(files)

    # Chunk the docs
    chunks = get_text_chunks(docs)

    # Embed and store in vector space
    vectorspace = get_vector_space(chunks)

    return vectorspace

def reset_conversation():
    st.session_state.messages = []

def main():
    # Set page name
    st.set_page_config(page_title="Chit-Chat",
                       page_icon=":blue_book:")
    
    # Set title
    st.title("How about a chat with your PDF📚")

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []
    
    # Set sidebar to upload files
    with st.sidebar:
        st.header("Upload PDFs 👇")
        uploaded_files = st.file_uploader("Choose a PDF file and hit run",
                                        type = ['pdf'],
                                        accept_multiple_files=False)

        files_loaded = uploaded_files

        # Hit Reset chat to clear contents
        st.button('Reset Chat', on_click=reset_conversation)
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter your query here.."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Query on similarity search and get top k searches
        response = generate_response(files_loaded)
        k = 10
        results = response.similarity_search(prompt, k=k)

        # Choose the result with Minimum Distance
        similarity_scores = {i:d[1] for i,d in enumerate(response.similarity_search_with_score(prompt, k=k ))}
        ss = pd.Series(similarity_scores)
        best_idx = ss.idxmin()
        output_based_on_scores = results[best_idx].page_content
        
        # Return response
        with st.chat_message("assistant"):
            st.markdown(output_based_on_scores)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": output_based_on_scores})


if __name__ == '__main__':
    main()

    
