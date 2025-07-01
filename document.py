import streamlit as st

import pymupdf
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

#Session States

if 'model' not in st.session_state:
    st.session_state['model'] = ChatOllama(model='data', base_url='http://localhost:11434')

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = OllamaEmbeddings(model='data', base_url='http://localhost:11434')

if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = FAISS(embedding_function=st.session_state['embeddings'], index=faiss.IndexFlatL2(len(st.session_state['embeddings'].embed_query("Hello World"))), 
      docstore=InMemoryDocstore(), index_to_docstore_id={})

if 'retriver' not in st.session_state:
    st.session_state['retriver'] = st.session_state['vector_store'].as_retriever(search_type = 'similarity', search_kwargs = {'k': 3})

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False

# Functions

def create_chunks(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(raw_documents)

def add_chunks(document_chunks):
    st.session_state['vector_store'].add_texts(texts=document_chunks)

system_prompt = (
"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.
Keep the answer concise.\n\n"""
"{context}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),])

question_chain = create_stuff_documents_chain(st.session_state['model'], prompt)
retriver_chain = create_retrieval_chain(st.session_state['retriver'], question_chain)

documents = []
def response_invoke(query):      
    for chunk in retriver_chain.stream({"input": query}):
        if chunk.get('context'):
            documents.append(chunk)
        elif chunk.get('answer'):
            yield chunk['answer']

# Chat

with st.sidebar:

    with st.spinner():
        uploaded_doc = st.file_uploader(
        "Upload your document",
        type="txt", key='txt')
        if uploaded_doc and st.session_state['uploaded'] == False:
            context = str(uploaded_doc.read(), 'utf-8')
            processed_chunks = create_chunks(context)
            add_chunks(processed_chunks)
            st.session_state['uploaded'] = True

    with st.spinner():
        uploaded_pdf = st.file_uploader(
        "Upload your document",
        type="pdf", key='pdf')
        if uploaded_pdf and st.session_state['uploaded'] == False:
            bytearray = uploaded_pdf.read()
            pdf = pymupdf.open(stream=bytearray, filetype="pdf")
            context = ""
            for page in pdf:
                context = context + "\n\n" + page.get_text()
            pdf.close()
            processed_chunks = create_chunks(context)
            add_chunks(processed_chunks)
            st.session_state['uploaded'] = True

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('Type here...'):
    st.session_state['messages'].append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.write(prompt)
    
    with st.chat_message('assistant'):
        message = st.write_stream(response_invoke(prompt))
        st.session_state['messages'].append({'role': 'assistant', 'content': message})

        for i in range(3):
            st.write({f'Resource {i+1}': documents[0]['context'][i].page_content})
        documents = []