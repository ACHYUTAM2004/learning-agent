import streamlit as st
import os
import textwrap
import tempfile
import google.generativeai as genai

# --- Function Imports ---
from utils.pdf_parser import extract_text
from utils.embeddings import generate_embeddings
from utils.supabase_handler import (
    semantic_search, upload_pdf, store_embeddings, 
    get_or_create_user, save_message, get_chat_history
)

# --- API & Model Configuration ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-pro-latest')


# --- Core Logic Functions ---
def generate_answer(query):
    """RAG pipeline for answering questions based on a PDF."""
    query_embedding = generate_embeddings([query])[0]
    relevant_chunks = semantic_search(query_embedding, top_k=10)
    
    if not relevant_chunks:
        return "Sorry, I couldn't find relevant information in the document."
        
    context = "\n".join([chunk['chunk'] for chunk in relevant_chunks])
    
    prompt = f"""
    You are an expert AI Learning Partner. Use the following context from a document to answer the user's question.

    Context:
    {context}
    ---
    Question: {query}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def generate_topic_answer(query, chat_history):
    """Generates an answer for a general topic using the AI's knowledge."""
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    prompt = f"""
    You are an expert AI Learning Partner helping a user understand a topic.
    
    Conversation History:
    {history_context}
    ---
    User's New Question: "{query}"

    Provide a clear, helpful, and structured response.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def process_file(file_path, original_filename):
    """Processes an uploaded PDF file."""
    try:
        upload_pdf(file_path, original_filename)
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        text = extract_text(pdf_bytes)
        chunks = textwrap.wrap(text, 1000)
        embeddings = generate_embeddings(chunks)
        store_embeddings(original_filename, chunks, embeddings)
        return True
    except Exception as e:
        if "Duplicate" in str(e):
            st.warning(f"File '{original_filename}' has already been processed.")
            return True
        st.error(f"Error processing file: {e}")
        return False


# --- Streamlit UI ---
st.set_page_config(page_title="AI Learning Partner", page_icon="🧠")
st.title("🧠 AI Learning Partner")

if "user_info" not in st.session_state:
    st.session_state.user_info = None

if st.session_state.user_info is None:
    st.markdown("Welcome! Please enter a username to start your session.")
    username = st.text_input("Username")
    if st.button("Start Session"):
        if username:
            with st.spinner("Setting up..."):
                st.session_state.user_info = get_or_create_user(username)
            st.rerun()
        else:
            st.warning("Please enter a username.")
else:
    # --- MAIN APP AFTER LOGIN ---
    username = st.session_state.user_info['username']
    user_id = st.session_state.user_info['id']
    
    st.sidebar.success(f"Logged in as **{username}**")

    # Initialize session state variables
    if "mode" not in st.session_state:
        st.session_state.mode = "Discuss a General Topic"
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None
    if "messages" not in st.session_state:
        chat_history = get_chat_history(user_id)
        st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

    # --- MODE SELECTOR ---
    mode = st.sidebar.radio(
        "Choose your learning mode:",
        ("Discuss a General Topic", "Study a Document"),
        key="mode_selector"
    )
    st.session_state.mode = mode

    # --- UI FOR DOCUMENT MODE ---
    if st.session_state.mode == "Study a Document":
        st.sidebar.header("Upload Your Document")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file:
            if uploaded_file.name != st.session_state.processed_file:
                with st.spinner("Processing file..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    success = process_file(tmp_file_path, uploaded_file.name)
                    os.remove(tmp_file_path)

                    if success:
                        st.session_state.processed_file = uploaded_file.name
                        st.success(f"Processed '{uploaded_file.name}'!")
                        st.session_state.messages = [] # Clear chat for new document
                    else:
                        st.session_state.processed_file = None

    # --- CHAT INTERFACE (COMMON FOR BOTH MODES) ---
    st.subheader(f"Mode: {st.session_state.mode}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to learn about?"):
        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.mode == "Study a Document":
                    if st.session_state.processed_file:
                        response = generate_answer(prompt)
                        save_message(user_id, "user", prompt, st.session_state.processed_file)
                        save_message(user_id, "assistant", response, st.session_state.processed_file)
                    else:
                        response = "Please upload a document to begin."
                else: # Topic Mode
                    response = generate_topic_answer(prompt, st.session_state.messages)
                    save_message(user_id, "user", prompt) # No document name
                    save_message(user_id, "assistant", response)
                
                st.markdown(response)
        
        # Add assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": response})