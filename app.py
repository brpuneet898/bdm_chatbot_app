# version 1 

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

# version 2 - added custom embeddings

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
# def create_and_save_vector_store(document_texts, save_path="faiss_index"):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(document_texts, embedder)
#     vector_store.save_local(save_path) 
#     return vector_store

# @st.cache_resource
# def load_vector_store(save_path="faiss_index"):
#     if os.path.exists(save_path):
#         embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         return FAISS.load_local(save_path, embedder)
#     return None

# @st.cache_resource
# def get_vector_store(document_texts, save_path="faiss_index"):
#     if os.path.exists(save_path):
#         st.write("Loading existing FAISS index...")
#         return load_vector_store(save_path)
#     else:
#         st.write("Creating and saving a new FAISS index...")
#         return create_and_save_vector_store(document_texts, save_path)

# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# model = load_model()
# document_texts = load_hidden_pdfs()

# vector_store = get_vector_store(document_texts)

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

# version 3 - added email id authentication

# import os
# import re
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
# def create_and_save_vector_store(document_texts, save_path="faiss_index"):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(document_texts, embedder)
#     vector_store.save_local(save_path) 
#     return vector_store

# @st.cache_resource
# def load_vector_store(save_path="faiss_index"):
#     if os.path.exists(save_path):
#         embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         return FAISS.load_local(save_path, embedder)
#     return None

# @st.cache_resource
# def get_vector_store(document_texts, save_path="faiss_index"):
#     if os.path.exists(save_path):
#         st.write("Loading existing FAISS index...")
#         return load_vector_store(save_path)
#     else:
#         st.write("Creating and saving a new FAISS index...")
#         return create_and_save_vector_store(document_texts, save_path)

# def is_valid_email(email):
#     email_pattern = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
#     return bool(re.match(email_pattern, email))

# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# email = st.text_input("Enter your email ID:")

# if email:
#     if is_valid_email(email):
#         st.session_state['email_valid'] = True
#         st.write("Email is valid! Now you can ask your questions.")

#         model = load_model()
#         document_texts = load_hidden_pdfs()

#         vector_store = get_vector_store(document_texts)

#         retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

#         if "chat_history" not in st.session_state:
#             st.session_state["chat_history"] = []

#         user_input = st.text_input("Pose your Questions:")

#         if user_input:
#             if user_input.lower() == "stop":
#                 st.write("Chatbot: Goodbye!")
#                 st.stop()
#             else:
#                 response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#                 answer = response["answer"]
#                 st.session_state["chat_history"].append((user_input, answer))
#                 for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#                     st.write(f"Q{i}: {question}")
#                     st.write(f"Chatbot: {reply}")

#     else:
#         st.write("Invalid email. Please use the format: XXfXXXXXXX@ds.study.iitm.ac.in")

# else:
#     st.write("Please enter your email ID to proceed.")

# version 4 - added name field and added json saving format.

# import os
# import re
# import json
# import streamlit as st
# from datetime import datetime
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
# def create_and_save_vector_store(document_texts, save_path="faiss_index"):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(document_texts, embedder)
#     vector_store.save_local(save_path) 
#     return vector_store

# @st.cache_resource
# def load_vector_store(save_path="faiss_index"):
#     if os.path.exists(save_path):
#         embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         return FAISS.load_local(save_path, embedder)
#     return None

# @st.cache_resource
# def get_vector_store(document_texts, save_path="faiss_index"):
#     if os.path.exists(save_path):
#         st.write("Loading existing FAISS index...")
#         return load_vector_store(save_path)
#     else:
#         st.write("Creating and saving a new FAISS index...")
#         return create_and_save_vector_store(document_texts, save_path)

# def is_valid_email(email):
#     email_pattern = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
#     return bool(re.match(email_pattern, email))

# def save_session_data(email, name, questions_and_answers):
#     session_data = {
#         "email": email,
#         "name": name,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "questions_and_answers": questions_and_answers
#     }
    
#     save_path = "session_data.json"

#     if os.path.exists(save_path):
#         with open(save_path, "r") as file:
#             all_sessions = json.load(file)
#     else:
#         all_sessions = []

#     all_sessions.append(session_data)
#     with open(save_path, "w") as file:
#         json.dump(all_sessions, file, indent=4)

# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# email = st.text_input("Enter your email ID:")
# name = st.text_input("Enter your name (optional):")

# if email:
#     if is_valid_email(email):
#         st.session_state['email_valid'] = True
#         st.write("Email is valid! Now you can ask your questions.")

#         model = load_model()
#         document_texts = load_hidden_pdfs()

#         vector_store = get_vector_store(document_texts)

#         retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

#         if "chat_history" not in st.session_state:
#             st.session_state["chat_history"] = []

#         user_input = st.text_input("Pose your Questions:")

#         if user_input:
#             if user_input.lower() == "stop":
#                 st.write("Chatbot: Goodbye!")
#                 save_session_data(email, name, st.session_state["chat_history"])
#                 st.stop()
#             else:
#                 response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#                 answer = response["answer"]
#                 st.session_state["chat_history"].append((user_input, answer))
#                 for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#                     st.write(f"Q{i}: {question}")
#                     st.write(f"Chatbot: {reply}")

#     else:
#         st.write("Invalid email. Please use the format: XXfXXXXXXX@ds.study.iitm.ac.in")

# else:
#     st.write("Please enter your email ID to proceed.")

# version 5 - added github details 

import os
import re
import json
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests  # For GitHub API interaction

# GitHub Repository details
GITHUB_API_URL = "https://api.github.com/repos/brpuneet898/bdm_chatbot_app/contents/session_data/"
GITHUB_TOKEN = "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"  # Use a GitHub token for authentication

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

def is_valid_email(email):
    email_pattern = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
    return bool(re.match(email_pattern, email))

# Function to save session data to GitHub
def save_session_to_github(session_data):
    timestamp = session_data["timestamp"]
    filename = f"session_data/{timestamp}.json"
    
    # Headers for GitHub API authentication
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if the session_data directory exists on GitHub
    # If not, create the directory by uploading a new file
    session_json = json.dumps(session_data, indent=4)

    # Data to be sent to GitHub API in base64 format
    data = {
        "message": f"Add session data for {timestamp}",
        "content": session_json.encode("utf-8").decode("utf-8")  # GitHub expects the content to be base64-encoded string
    }

    # Create or update the file on GitHub
    response = requests.put(f"https://api.github.com/repos/brpuneet898/bdm_chatbot_app/contents/{filename}", headers=headers, json=data)
    
    if response.status_code == 201:
        st.write(f"Session data saved to GitHub as {filename}")
    else:
        st.write(f"Error saving session data to GitHub: {response.text}")
        
# Streamlit UI code
st.title("BDM Chatbot")
st.write("Ask questions directly based on the preloaded BDM documents.")

email = st.text_input("Enter your email ID:")
name = st.text_input("Enter your name (optional):")

if email:
    if is_valid_email(email):
        st.session_state['email_valid'] = True
        st.write("Email is valid! Now you can ask your questions.")

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
                # Save session data to GitHub
                session_data = {
                    "email": email,
                    "name": name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "questions_and_answers": st.session_state["chat_history"]
                }
                save_session_to_github(session_data)
                st.stop()
            else:
                response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
                answer = response["answer"]
                st.session_state["chat_history"].append((user_input, answer))
                for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
                    st.write(f"Q{i}: {question}")
                    st.write(f"Chatbot: {reply}")

    else:
        st.write("Invalid email. Please use the format: XXfXXXXXXX@ds.study.iitm.ac.in")

else:
    st.write("Please enter your email ID to proceed.")

