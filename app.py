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

model = load_model()
@st.cache_data
def load_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return [page.page_content for page in pages]

@st.cache_resource
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

st.title("BDM Chatbot")
st.write("Upload your PDF documents to ask specific questions.")

pdf_file = st.file_uploader("Upload PDF", type="pdf")

if pdf_file is not None:
    with open("uploaded_document.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    document_texts = load_pdf_text("uploaded_document.pdf")
    vector_store = create_vector_store(document_texts)
    retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_input("Pose your Questions:")

    if user_input:
        response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
        answer = response["answer"]
        st.session_state["chat_history"].append((user_input, answer))
        st.write("Chatbot:", answer)

else:
    st.write("Please upload a PDF document to start.")
