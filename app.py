import streamlit as st
import tempfile
import os
import textwrap
import google.generativeai as genai

# --- Function Imports ---
from utils.pdf_parser import extract_text
from utils.embeddings import generate_embeddings
from utils.supabase_handler import semantic_search, upload_pdf, store_embeddings

# --- API Configuration ---
# Make sure you have your GOOGLE_API_KEY in a .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-latest')

# --- Core Logic Functions ---
def generate_answer(query):
    """Orchestrates the RAG pipeline to generate an answer for a given query."""
    query_embedding = generate_embeddings([query])[0]
    relevant_chunks = semantic_search(query_embedding)
    
    if not relevant_chunks:
        return "Sorry, I couldn't find any relevant information in the uploaded documents."
        
    context = " ".join([chunk['chunk'] for chunk in relevant_chunks])
    
    prompt = f"""
    Based on the following context from a document, please answer the user's question.
    Provide a clear and concise answer. If the context is insufficient, say so.

    Context:
    {context}

    Question: {query}

    Answer:
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
st.markdown("Upload a PDF and ask questions to get summaries, definitions, and more.")

# Initialize session state for chat history and processed file
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

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
if prompt := st.chat_input("Ask a question about your document..."):
    # First, check if a file has been processed
    if st.session_state.processed_file is None:
        st.warning("Please upload and process a PDF file first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_answer(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
