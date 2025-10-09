import streamlit as st
import tempfile
import os
import textwrap
import google.generativeai as genai

# --- Function Imports ---
from utils.pdf_parser import extract_text
from utils.embeddings import generate_embeddings
from utils.supabase_handler import semantic_search, upload_pdf, store_embeddings, get_or_create_user, get_chat_history, save_message

# --- API Configuration ---
# Make sure you have your GOOGLE_API_KEY in a .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-latest')

# --- Core Logic Functions ---
def generate_answer(query):
    """Orchestrates the RAG pipeline to generate an answer for a given query."""
    query_embedding = generate_embeddings([query])[0]
    relevant_chunks = semantic_search(query_embedding,top_k=15)
    
    if not relevant_chunks:
        return "Sorry, I couldn't find any relevant information in the uploaded documents."
        
    context = " ".join([chunk['chunk'] for chunk in relevant_chunks])
    
    prompt = f"""
    You are an expert AI Learning Partner. Your goal is to provide a comprehensive and helpful answer to the user's question.

    A user has asked the following question: "{query}"

    Some context has been retrieved from a document they provided:
    ---
    Context:
    {context}
    ---

    Please follow these steps to answer the question:
    1. First, carefully analyze the provided context to see if it directly answers the user's question.
    2. If the context fully answers the question, provide the answer based **only** on that context.
    3. If the context is insufficient or does not contain the answer, use your own general knowledge to provide a complete and accurate response. When doing so, you can optionally mention that the information extends beyond the provided document.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

def process_file(file_path, original_filename):
    """Processes an uploaded PDF file."""
    try:
        # 1. Upload to Supabase Storage
        upload_pdf(file_path, original_filename)
        
        # 2. Extract text and create embeddings
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        text = extract_text(pdf_bytes)
        chunks = textwrap.wrap(text, 1000)
        embeddings = generate_embeddings(chunks)
        
        # 3. Store in Supabase vector table
        store_embeddings(original_filename, chunks, embeddings)
        
        return True
    except Exception as e:
        # Check for the specific duplicate file error
        if "Duplicate" in str(e):
            st.warning(f"⚠️ File '{original_filename}' has already been processed. You can start asking questions.")
            return True # It's not a failure, the file is ready to be queried
        st.error(f"❌ Error processing file: {e}")
        return False


# --- Streamlit UI ---

# Set the page title and icon
st.set_page_config(page_title="AI Learning Partner", page_icon="🧠")

st.title("🧠 AI Learning Partner")
# --- USER ONBOARDING & SESSION MANAGEMENT ---
if "user_info" not in st.session_state:
    st.session_state.user_info = None

if st.session_state.user_info is None:
    st.markdown("Welcome! Please enter a username to start your personalized learning session.")
    username = st.text_input("Username")
    if st.button("Start Session"):
        if username:
            with st.spinner("Setting up your session..."):
                st.session_state.user_info = get_or_create_user(username)
            st.rerun() # Rerun the script to move to the main app view
        else:
            st.warning("Please enter a username.")

else:
    # --- MAIN APP LOGIC (runs after user is identified) ---
    username = st.session_state.user_info['username']
    user_id = st.session_state.user_info['id']
    st.sidebar.success(f"Logged in as **{username}**")

    # Initialize session state for the app
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None
    if "messages" not in st.session_state:
        # Load chat history from the database
        chat_history = get_chat_history(user_id)
        st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]


    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Your Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Process the file only if it's a new file
            if uploaded_file.name != st.session_state.processed_file:
                with st.spinner("Processing file... This may take a moment."):
                    # Save the uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Call the processing function
                    success = process_file(tmp_file_path, uploaded_file.name)
                    
                    # Clean up the temporary file
                    os.remove(tmp_file_path)

                    if success:
                        st.session_state.processed_file = uploaded_file.name
                        st.success(f"✅ Successfully processed '{uploaded_file.name}'!")
                        # Clear previous chat history when a new file is uploaded
                        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question..."):
        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(user_id, "user", prompt, st.session_state.processed_file)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_answer(prompt)
                st.markdown(response)
        
        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message(user_id, "assistant", response, st.session_state.processed_file)

