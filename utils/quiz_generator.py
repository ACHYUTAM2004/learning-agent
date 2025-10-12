import json
import streamlit as st

def generate_quiz(context, model, num_questions=2):
    """
    Generates a multiple-choice quiz from a given context using the provided model.
    """
    
    # Use the num_questions argument in the prompt
    prompt = f"""
    You are an expert quiz designer. Based on the following text context, create a {num_questions}-question multiple-choice quiz.
    The questions should test the user's understanding of the key concepts in the text.

    **Instructions for output:**
    - Provide the output in a strict JSON format.
    - The JSON should be a list of objects, where each object represents a question.
    - Each question object must have three keys: "question" (string), "options" (a list of 4 strings), and "correct_answer" (a string that exactly matches one of the options).

    Context:
    ---
    {context}
    ---
    """

    try:
        response = model.generate_content(prompt)
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        quiz_questions = json.loads(json_response)
        return quiz_questions
    except Exception as e:
        st.error(f"Failed to generate or parse quiz: {e}")
        return None