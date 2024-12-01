# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# os.environ["GROQ_API_KEY"] = "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"

# @st.cache_resource
# def load_model():
#     return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# @st.cache_data
# def load_hidden_pdfs(directory="hidden_docs"):
#     all_texts = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(directory, filename))
#             pages = loader.load_and_split()
#             all_texts.extend([page.page_content for page in pages])
#     return all_texts

# @st.cache_resource
# def create_vector_store(document_texts):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(document_texts, embedder)

# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# model = load_model()
# document_texts = load_hidden_pdfs()
# vector_store = create_vector_store(document_texts)
# retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())
    
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
        
# user_input = st.text_input("Pose your Questions:")

# if user_input:
#     if user_input.lower() == "stop":
#         st.write("Chatbot: Goodbye!")
#         st.stop()
#     else:
#         response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#         answer = response["answer"]
#         st.session_state["chat_history"].append((user_input, answer))
#         for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#             st.write(f"Q{i}: {question}")
#             st.write(f"Chatbot: {reply}")

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

os.environ["GROQ_API_KEY"] = "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"

@st.cache_resource
def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

@st.cache_data
def load_hidden_pdfs(directory="hidden_docs"):
    all_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load_and_split()
            all_texts.extend([page.page_content for page in pages])
    return all_texts

@st.cache_resource
def create_and_save_vector_store(document_texts, save_path="faiss_index"):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(document_texts, embedder)
    vector_store.save_local(save_path) 
    return vector_store

@st.cache_resource
def load_vector_store(save_path="faiss_index"):
    if os.path.exists(save_path):
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(save_path, embedder)
    return None

@st.cache_resource
def get_vector_store(document_texts, save_path="faiss_index"):
    if os.path.exists(save_path):
        st.write("Loading existing FAISS index...")
        return load_vector_store(save_path)
    else:
        st.write("Creating and saving a new FAISS index...")
        return create_and_save_vector_store(document_texts, save_path)

st.title("BDM Chatbot")
st.write("Ask questions directly based on the preloaded BDM documents.")

model = load_model()
document_texts = load_hidden_pdfs()

vector_store = get_vector_store(document_texts)

retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Pose your Questions:")

if user_input:
    if user_input.lower() == "stop":
        st.write("Chatbot: Goodbye!")
        st.stop()
    else:
        response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
        answer = response["answer"]
        st.session_state["chat_history"].append((user_input, answer))
        for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
            st.write(f"Q{i}: {question}")
            st.write(f"Chatbot: {reply}")

