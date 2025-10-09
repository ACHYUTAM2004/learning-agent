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
pro_model = genai.GenerativeModel('gemini-pro-latest')
flash_model = genai.GenerativeModel('gemini-flash-latest')


# --- Core Logic Functions ---
def generate_answer(query,model,knowledge_level):
    """RAG pipeline for answering questions based on a PDF."""
    query_embedding = generate_embeddings([query])[0]
    relevant_chunks = semantic_search(query_embedding, top_k=10)
    
    if not relevant_chunks:
        return "Sorry, I couldn't find relevant information in the document."
        
    context = "\n".join([chunk['chunk'] for chunk in relevant_chunks])
    
    prompt = f"""
    You are an expert AI Learning Partner. The user you are helping has a knowledge level of **'{knowledge_level}'**.
    You must tailor your explanation's depth, language, and complexity to match this level. For 'Beginners', use simple terms and analogies. For 'Experts', provide technical, nuanced details.

    A user has asked the following question: "{query}"

    Some context has been retrieved from a document they provided:
    ---
    Context:
    {context}
    ---

    Please follow these steps to answer the question, always keeping the user's knowledge level in mind:
    1.  First, carefully analyze the provided context to see if it directly answers the user's question.
    2.  If the context fully answers the question, provide the answer based **only** on that context, adapting the explanation for the user's knowledge level.
    3.  If the context is insufficient, use your own general knowledge to provide a complete and accurate response, still tailored to the user's knowledge level.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def generate_topic_answer(query, chat_history,model,knowledge_level):
    """Generates an answer for a general topic using the AI's knowledge."""
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    prompt = f"""
    You are an AI Learning Partner. Your user's knowledge level is '{knowledge_level}'.
    Tailor your explanation's depth and language accordingly. For Beginners, use simple terms and analogies. For Experts, provide technical and nuanced details.

    Review the conversation history and answer the user's latest question.

    Conversation History:
    {history_context}
    ---
    User's New Question: "{query}"
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

# --- USER ONBOARDING & SESSION MANAGEMENT ---
if "user_info" not in st.session_state:
    st.session_state.user_info = None

if st.session_state.user_info is None:
    st.markdown("Welcome! Please enter a username to start your session.")
    username = st.text_input("Username")
    
    # ADDED: Knowledge level selection
    knowledge_level = st.selectbox(
        "What is your knowledge level on most topics?",
        ("Beginner", "Intermediate", "Expert")
    )

    if st.button("Start Session"):
        if username:
            with st.spinner("Setting up your session..."):
                # Pass the knowledge level when creating the user
                st.session_state.user_info = get_or_create_user(username, knowledge_level)
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
        knowledge_level = st.session_state.user_info['knowledge_level']
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.mode == "Study a Document":
                    if st.session_state.processed_file:
                        response = generate_answer(prompt,model=pro_model,knowledge_level=knowledge_level)
                        save_message(user_id, "user", prompt, st.session_state.processed_file)
                        save_message(user_id, "assistant", response, st.session_state.processed_file)
                    else:
                        response = "Please upload a document to begin."
                else: # Topic Mode
                    response = generate_topic_answer(prompt, st.session_state.messages,model=flash_model,knowledge_level=knowledge_level)
                    save_message(user_id, "user", prompt) # No document name
                    save_message(user_id, "assistant", response)
                
                st.markdown(response)
        
        # Add assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": response})