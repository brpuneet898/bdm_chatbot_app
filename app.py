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

# version 5 - sending email to my emial id 

import os
import re
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
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

def is_valid_email(email):
    email_pattern = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
    return bool(re.match(email_pattern, email))

def save_session_data(email, name, questions_and_answers):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_data = {
        "email": email,
        "name": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "questions_and_answers": questions_and_answers
    }
    
    save_path = f"session_data_{timestamp}.json"

    with open(save_path, "w") as file:
        json.dump(session_data, file, indent=4)
    
    return save_path

def send_email_with_attachment(recipient_email, attachment_path):
    sender_email = "21f3002005@ds.study.iitm.ac.in"  
    sender_password = "etfn7vp9" 

    subject = "Session Data"
    body = "Please find attached the session data."
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))
    with open(attachment_path, "rb") as attachment_file:
        attach_part = MIMEApplication(attachment_file.read(), _subtype="json")
        attach_part.add_header("Content-Disposition", "attachment", filename=os.path.basename(attachment_path))
        msg.attach(attach_part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

st.title("BDM Chatbot")
st.write("Ask questions directly based on the preloaded BDM documents.")

email = st.text_input("Enter your email ID:")
name = st.text_input("Enter your name:")

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
                session_file_path = save_session_data(email, name, st.session_state["chat_history"])
                send_email_with_attachment("21f3002005@ds.study.iitm.ac.in", session_file_path)
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

