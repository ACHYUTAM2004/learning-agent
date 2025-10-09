import os
import streamlit as st  # Make sure this import is present
from supabase import create_client, Client
from typing import List

# Use Streamlit's secrets management for deployment
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
BUCKET_NAME = st.secrets["SUPABASE_BUCKET"]

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_pdf(file_path, destination_path):
    """
    Uploads a file to the Supabase storage bucket.
    `file_path` is the local path to the file.
    `destination_path` is the desired name/path in the bucket.
    """
    with open(file_path, 'rb') as f:
        # The upsert option will prevent errors if the file already exists
        supabase.storage.from_(BUCKET_NAME).upload(file=f, path=destination_path, file_options={"upsert": "true"})
    
    return supabase.storage.from_(BUCKET_NAME).get_public_url(destination_path)


def store_embeddings(file_name: str, text_chunks: List[str], embeddings: List[List[float]]):
    """
    Stores text chunks and their embeddings in a single batch insert.
    """
    data_to_insert = [
        {
            "file_name": file_name,
            "chunk": chunk,
            "embedding": embedding
        }
        for chunk, embedding in zip(text_chunks, embeddings)
    ]
    
    supabase.table("documents").insert(data_to_insert).execute()


def semantic_search(query_embedding: List[float], top_k: int = 5):
    """
    ✅ CORRECT VERSION: Performs similarity search by calling the 'match_documents' RPC function.
    """
    results = supabase.rpc('match_documents', {
        'query_embedding': query_embedding,
        'match_count': top_k
    }).execute()
    
    return results.data

# --- NEW FUNCTIONS FOR PERSONALIZATION ---

def get_or_create_user(username):
    """Fetches a user by username or creates a new one."""
    # Check if user exists
    user = supabase.table("users").select("*").eq("username", username).execute()
    
    if user.data:
        return user.data[0]
    else:
        # Create a new user if not found
        new_user = supabase.table("users").insert({"username": username}).execute()
        return new_user.data[0]

def save_message(user_id, role, content, document_name=None):
    """Saves a chat message to the conversations table."""
    supabase.table("conversations").insert({
        "user_id": user_id,
        "role": role,
        "content": content,
        "document_name": document_name
    }).execute()

def get_chat_history(user_id):
    """Fetches the chat history for a given user."""
    history = supabase.table("conversations").select("role, content").eq("user_id", user_id).order("created_at").execute()
    return history.data