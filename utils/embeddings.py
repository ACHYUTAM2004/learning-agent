import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini API client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_embeddings(text_chunks):
    """
    Given a list of text chunks, returns a list of embeddings using Google's model.
    """
    try:
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=text_chunks,
            task_type="RETRIEVAL_DOCUMENT" # Important for retrieval tasks
        )
        return result['embedding']
    except Exception as e:
        print(f"An error occurred with Gemini embedding: {e}")
        return []