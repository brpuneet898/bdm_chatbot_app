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

# version 5 - added download feature

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

# Your API Key (ensure it's handled securely, not hardcoded in production)
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

def save_session_data(email, name, questions_and_answers):
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_data = {
        "email": email,
        "name": name,
        "timestamp": timestamp,
        "questions_and_answers": questions_and_answers
    }
    
    save_path = f"session_data_{timestamp}.json"
    with open(save_path, "w") as file:
        json.dump(session_data, file, indent=4)

    return save_path

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
                # Save session data with timestamped filename
                session_file = save_session_data(email, name, st.session_state["chat_history"])

                # Provide download link for session data
                with open(session_file, "rb") as file:
                    st.download_button("Download Session Data", file, file_name=session_file)

                st.session_state["chat_history"] = []  # Reset chat history after saving
                st.stop()  # End the app session
                
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

# version 6 - added supabase

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
# from supabase import create_client, Client  # Import Supabase client

# # Allow dangerous deserialization if you're sure the data source is trusted
# st.set_option('server.allow_dangerous_deserialization', True)


# # Supabase credentials (get these from the Supabase dashboard)
# SUPABASE_URL = "https://armzsxwnhybsgedffijs.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFybXpzeHduaHlic2dlZGZmaWpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzMwODcxMzEsImV4cCI6MjA0ODY2MzEzMX0.g7Ty0qNFCVJiEp38IQ_Uw9yEn4jzA67XPsLCmQ8f26o"
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Your API Key (ensure it's handled securely, not hardcoded in production)
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
#     # Create timestamped filename
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     session_data = {
#         "email": email,
#         "name": name,
#         "timestamp": timestamp,
#         "questions_and_answers": questions_and_answers
#     }
    
#     save_path = f"session_data_{timestamp}.json"
#     with open(save_path, "w") as file:
#         json.dump(session_data, file, indent=4)

#     return save_path

# def save_to_supabase(email, name, question, answer):
#     """ Save the chat entry to Supabase database """
#     response = supabase.table("chat_sessions").insert({
#         "email": email,
#         "name": name,
#         "question": question,
#         "answer": answer
#     }).execute()

#     if response.status_code == 201:
#         st.write("Successfully saved to the database!")
#     else:
#         st.write("Failed to save to the database:", response)

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
#                 # Save session data with timestamped filename
#                 session_file = save_session_data(email, name, st.session_state["chat_history"])

#                 # Provide download link for session data
#                 with open(session_file, "rb") as file:
#                     st.download_button("Download Session Data", file, file_name=session_file)

#                 st.session_state["chat_history"] = []  # Reset chat history after saving
#                 st.stop()  # End the app session
                 
#             else:
#                 response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#                 answer = response["answer"]
#                 st.session_state["chat_history"].append((user_input, answer))
#                 save_to_supabase(email, name, user_input, answer)  # Save each Q&A to Supabase

#                 for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#                     st.write(f"Q{i}: {question}")
#                     st.write(f"Chatbot: {reply}")

#     else:
#         st.write("Invalid email. Please use the format: XXfXXXXXXX@ds.study.iitm.ac.in")

# else:
#     st.write("Please enter your email ID to proceed.")

