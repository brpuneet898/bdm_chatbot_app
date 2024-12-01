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

# version 2 - without custom embeddings 

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
from supabase import create_client, Client

# Supabase setup
url = "https://armzsxwnhybsgedffijs.supabase.co"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFybXpzeHduaHlic2dlZGZmaWpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzMwODcxMzEsImV4cCI6MjA0ODY2MzEzMX0.g7Ty0qNFCVJiEp38IQ_Uw9yEn4jzA67XPsLCmQ8f26o"
supabase: Client = create_client(url, api_key)

# Groq API Key
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
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

# Email validation regex for the specific pattern
def is_valid_email(email):
    return bool(re.match(r'^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$', email))

# Function to save chat history to Supabase
def save_to_supabase(session_data):
    table_name = "chat_sessions"
    response = supabase.table(table_name).insert(session_data).execute()
    if response.status_code == 201:
        st.success("Session saved to Supabase.")
    else:
        st.error("Error saving session to Supabase.")

# Streamlit UI
st.title("BDM Chatbot")
st.write("Ask questions directly based on the preloaded BDM documents.")

# Input fields for email and name
email = st.text_input("Enter your Email (XXfXXXXXXX@ds.study.iitm.ac.in):")
name = st.text_input("Enter your Name (Optional):")

# Button to proceed after email validation
if email and not is_valid_email(email):
    st.error("Invalid email format! Please enter a valid email.")
else:
    # User is allowed to interact with the chatbot after email validation
    model = load_model()
    document_texts = load_hidden_pdfs()
    vector_store = create_vector_store(document_texts)
    retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Pose your Questions:")

    if user_input:
        if user_input.lower() == "stop":
            # Enable download after stop
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_data_{timestamp}.json"
            file_path = f"/mnt/data/{filename}"
            
            # Save chat history to file
            with open(file_path, "w") as f:
                json.dump(st.session_state["chat_history"], f)
                
            st.download_button(label="Download Session Data", data=open(file_path, "rb"), file_name=filename)
            
            # Save session data to Supabase
            session_data = {
                "email": email,
                "name": name,
                "chat_history": st.session_state["chat_history"],
                "timestamp": timestamp
            }
            save_to_supabase(session_data)
            
            st.write("Chatbot: Goodbye!")
            st.stop()

        else:
            response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
            answer = response["answer"]
            st.session_state["chat_history"].append((user_input, answer))
            for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
                st.write(f"Q{i}: {question}")
                st.write(f"Chatbot: {reply}")


# version 5 - added download feature

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
#                 for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#                     st.write(f"Q{i}: {question}")
#                     st.write(f"Chatbot: {reply}")

#     else:
#         st.write("Invalid email. Please use the format: XXfXXXXXXX@ds.study.iitm.ac.in")

# else:
#     st.write("Please enter your email ID to proceed.")

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
# from supabase import create_client, Client

# # Supabase configuration
# SUPABASE_URL = "https://armzsxwnhybsgedffijs.supabase.co"
# SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFybXpzeHduaHlic2dlZGZmaWpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzMwODcxMzEsImV4cCI6MjA0ODY2MzEzMX0.g7Ty0qNFCVJiEp38IQ_Uw9yEn4jzA67XPsLCmQ8f26o"
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# # Groq API Key
# os.environ["GROQ_API_KEY"] = "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"

# # Email validation function
# def is_valid_email(email):
#     email_pattern = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
#     return bool(re.match(email_pattern, email))

# # Save session data to a JSON file
# def save_session_data(email, name, questions_and_answers):
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

#     return save_path, session_data

# # Store session data in Supabase
# def store_session_in_supabase(email, name, questions_and_answers):
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     session_data = {
#         "email": email,
#         "name": name,
#         "timestamp": timestamp,
#         "questions_and_answers": questions_and_answers
#     }
#     # Inserting the session data into Supabase
#     response = supabase.table("sessions").insert(session_data).execute()
#     return response

# # Load the model and documents
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
#     return FAISS.from_texts(document_texts)

# # Streamlit app layout
# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# email = st.text_input("Enter your email ID:")
# name = st.text_input("Enter your name (optional):")

# if email:
#     if is_valid_email(email):
#         st.session_state['email_valid'] = True
#         st.write("Email is valid! Now you can ask your questions.")

#         # Load model and documents
#         model = load_model()
#         document_texts = load_hidden_pdfs()
#         vector_store = create_vector_store(document_texts)
#         retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

#         # Initialize chat history if not present
#         if "chat_history" not in st.session_state:
#             st.session_state["chat_history"] = []

#         # User input
#         user_input = st.text_input("Pose your Questions:")

#         if user_input:
#             if user_input.lower() == "stop":
#                 st.write("Chatbot: Goodbye!")
                
#                 # Save session data and provide a download link
#                 session_file, session_data = save_session_data(email, name, st.session_state["chat_history"])

#                 with open(session_file, "rb") as file:
#                     st.download_button("Download Session Data", file, file_name=session_file)

#                 # Store session data in Supabase
#                 store_session_in_supabase(email, name, session_data["questions_and_answers"])

#                 # Reset chat history
#                 st.session_state["chat_history"] = []  # Reset chat history after saving
#                 st.stop()

#             else:
#                 # Get the response from the retriever
#                 response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#                 answer = response["answer"]
                
#                 # Update chat history
#                 st.session_state["chat_history"].append((user_input, answer))
                
#                 # Display the conversation
#                 for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#                     st.write(f"Q{i}: {question}")
#                     st.write(f"Chatbot: {reply}")
#     else:
#         st.write("Invalid email. Please use the format: XXfXXXXXXX@ds.study.iitm.ac.in")

# else:
#     st.write("Please enter your email ID to proceed.")


