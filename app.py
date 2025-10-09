import streamlit as st
import os
import textwrap
import tempfile
import google.generativeai as genai
import re

# --- Function Imports ---
from utils.pdf_parser import extract_text
from utils.embeddings import generate_embeddings
from utils.supabase_handler import (
    semantic_search, upload_pdf, store_embeddings, 
    get_or_create_user, save_message, get_chat_history
)
from utils.quiz_generator import generate_quiz

# --- API & Model Configuration ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
pro_model = genai.GenerativeModel('gemini-pro-latest')
flash_model = genai.GenerativeModel('gemini-flash-latest')


# --- Core Logic Functions ---
def generate_lesson_plan(topic, model):
    """Generates a structured lesson plan for a given topic."""
    prompt = f"""
    You are an expert curriculum designer. A user wants a guided lesson on '{topic}'. 
    Create a 4-step lesson plan as a simple numbered list. Each step should be a distinct sub-topic.
    Do not add any introductory or concluding text, only the numbered list.
    Example:
    1. First sub-topic
    2. Second sub-topic
    3. Third sub-topic
    4. Fourth sub-topic
    """
    try:
        response = model.generate_content(prompt)
        # Use regex to parse the numbered list
        plan = re.findall(r'\d+\.\s*(.*)', response.text)
        return plan if plan else None
    except Exception as e:
        st.error(f"Failed to generate lesson plan: {e}")
        return None

def explain_sub_topic(sub_topic, knowledge_level, model):
    """Explains a single sub-topic from the lesson plan."""
    prompt = f"""
    You are a teacher explaining the sub-topic: '{sub_topic}'. 
    Explain this concept clearly and concisely to a user with a '{knowledge_level}' level of understanding. 
    Focus only on this sub-topic. End your explanation naturally without asking a question.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

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

# --- Initialize all session state variables ---
if "user_info" not in st.session_state:
    st.session_state.user_info = None
    st.session_state.messages = []
    st.session_state.mode = "Discuss a General Topic"
    st.session_state.processed_file = None
    st.session_state.in_guided_session = False
    st.session_state.lesson_plan = None
    st.session_state.lesson_step = 0
    st.session_state.quiz_mode = False
    st.session_state.quiz_questions = None
    st.session_state.current_question_index = 0
    st.session_state.score = 0

# --- User Onboarding ---
if st.session_state.user_info is None:
    st.markdown("Welcome! Please enter a username to start your session.")
    username = st.text_input("Username")
    knowledge_level = st.selectbox("What is your knowledge level on most topics?", ("Beginner", "Intermediate", "Expert"))
    if st.button("Start Session"):
        if username:
            with st.spinner("Setting up your session..."):
                st.session_state.user_info = get_or_create_user(username, knowledge_level)
            st.rerun()
        else:
            st.warning("Please enter a username.")
else:
    # --- MAIN APP AFTER LOGIN ---
    username = st.session_state.user_info['username']
    user_id = st.session_state.user_info['id']
    st.sidebar.success(f"Logged in as **{username}**")

    # Load chat history only once when the session starts for a logged-in user
    if not st.session_state.messages:
        chat_history = get_chat_history(user_id)
        if chat_history:
            st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

    # Mode and Knowledge Level Selectors in sidebar
    st.session_state.mode = st.sidebar.radio("Choose your learning mode:", ("Discuss a General Topic", "Study a Document"))
    st.sidebar.markdown("---")
    levels = ("Beginner", "Intermediate", "Expert")
    current_level = st.session_state.user_info.get('knowledge_level', 'Intermediate')
    current_index = levels.index(current_level)
    new_level = st.sidebar.selectbox("Adjust your knowledge level:", levels, index=current_index)
    if new_level != current_level:
        st.session_state.user_info['knowledge_level'] = new_level
        st.sidebar.info(f"Knowledge level set to **{new_level}**.")

    st.subheader(f"Mode: {st.session_state.mode}")

    # --- THIS IS THE MAIN UI CONTROLLER ---
    if st.session_state.quiz_mode:
        # --- QUIZ UI ---
        # (This section handles displaying and managing the quiz)
        index = st.session_state.current_question_index
        questions = st.session_state.quiz_questions
        if index < len(questions):
            q = questions[index]
            st.info(f"Question {index + 1}/{len(questions)}: {q['question']}")
            with st.form(key=f"quiz_form_{index}"):
                user_answer = st.radio("Choose your answer:", options=q['options'], index=None)
                if st.form_submit_button("Submit Answer"):
                    if user_answer == q['correct_answer']:
                        st.success("Correct! 🎉")
                        st.session_state.score += 1
                    else:
                        st.error(f"Not quite. The correct answer was: {q['correct_answer']}")
                    st.session_state.current_question_index += 1
                    st.rerun()
        else:
            st.success(f"Quiz complete! Your final score is: {st.session_state.score}/{len(questions)}")
            if st.button("End Session"):
                st.session_state.in_guided_session = False
                st.session_state.quiz_mode = False
                st.session_state.messages = []
                st.rerun()

    elif st.session_state.in_guided_session:
        # --- GUIDED SESSION UI ---
        # (This section handles the step-by-step lesson)
        plan = st.session_state.lesson_plan
        step = st.session_state.lesson_step
        st.sidebar.markdown("### Lesson Plan")
        for i, sub_topic in enumerate(plan):
            st.sidebar.markdown(f"**➡️ {i+1}. {sub_topic}**" if i == step else f"{i+1}. {sub_topic}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if step < len(plan) - 1:
            if st.button("Continue to Next Step"):
                st.session_state.lesson_step += 1
                with st.spinner("Preparing the next topic..."):
                    next_sub_topic = plan[st.session_state.lesson_step]
                    explanation = explain_sub_topic(next_sub_topic, st.session_state.user_info['knowledge_level'], flash_model)
                    st.session_state.messages.append({"role": "assistant", "content": explanation})
                st.rerun()
        else:
            st.info("You've completed the guided lesson!")
            if st.button("Ready for a quiz?"):
                with st.spinner("Generating your quiz..."):
                    conversation_context = " ".join([msg['content'] for msg in st.session_state.messages if msg['role'] == 'assistant'])
                    quiz = generate_quiz(conversation_context, pro_model)
                    if quiz:
                        st.session_state.quiz_questions = quiz
                        st.session_state.current_question_index = 0
                        st.session_state.score = 0
                        st.session_state.quiz_mode = True
                        st.rerun()

    else:
        # --- DEFAULT LOBBY / FREE-FORM CHAT UI ---
        # (This section is the starting point for both modes)
        if st.session_state.mode == "Discuss a General Topic":
            topic = st.text_input("What topic would you like a guided lesson on?")
            if st.button("Start Guided Session"):
                if topic:
                    with st.spinner("Creating a lesson plan..."):
                        plan = generate_lesson_plan(topic, pro_model)
                        if plan:
                            st.session_state.lesson_plan = plan
                            st.session_state.in_guided_session = True
                            st.session_state.lesson_step = 0
                            st.session_state.messages = []
                            first_sub_topic = plan[0]
                            explanation = explain_sub_topic(first_sub_topic, st.session_state.user_info['knowledge_level'], flash_model)
                            st.session_state.messages.append({"role": "assistant", "content": f"Great! I've prepared a lesson on '{topic}'. Here is the first part:"})
                            st.session_state.messages.append({"role": "assistant", "content": explanation})
                            st.rerun()

        if st.session_state.mode == "Study a Document":
            st.sidebar.header("Upload Your Document")
            uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file and uploaded_file.name != st.session_state.processed_file:
                with st.spinner("Processing file..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    success = process_file(tmp_file_path, uploaded_file.name)
                    os.remove(tmp_file_path)
                    if success:
                        st.session_state.processed_file = uploaded_file.name
                        st.success(f"Processed '{uploaded_file.name}'!")
                        st.session_state.messages = []
                        st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a free-form question..."):
            if st.session_state.mode == "Study a Document" and not st.session_state.processed_file:
                st.warning("Please upload a document first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        knowledge_level = st.session_state.user_info['knowledge_level']
                        if st.session_state.mode == "Study a Document":
                            response = generate_answer(prompt, pro_model, knowledge_level)
                            save_message(user_id, "user", prompt, st.session_state.processed_file)
                            save_message(user_id, "assistant", response, st.session_state.processed_file)
                        else:
                            # Free-form chat in Topic Mode
                            response = generate_topic_answer(prompt, st.session_state.messages, flash_model, knowledge_level)
                            save_message(user_id, "user", prompt)
                            save_message(user_id, "assistant", response)
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
