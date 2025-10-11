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
    get_or_create_user, save_message, get_chat_history, update_goal_progress, create_learning_goal
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

def generate_answer(query, model, knowledge_level, file_name):
    """
    RAG pipeline that always falls back to general knowledge if the
    document/video context is insufficient.
    """
    query_embedding = generate_embeddings([query])[0]
    
    # Perform the filtered search
    relevant_chunks = semantic_search(query_embedding, file_name, top_k=10)
    
    # --- THIS IS THE KEY CHANGE ---
    # We no longer stop if chunks are not found.
    # Instead, we create the context (which will be an empty string if nothing is found).
    context = "\n".join([chunk['chunk'] for chunk in relevant_chunks])
    
    # The existing hybrid prompt is smart enough to handle an empty context.
    prompt = f"""
    You are an AI Learning Partner. The user you are helping has a knowledge level of '{knowledge_level}'.
    You must tailor your explanation's depth and language to match this level.

    A user has asked the following question: "{query}"

    Some context has been retrieved from a document they provided:
    ---
    Context:
    {context}
    ---

    Please follow these steps to answer the question:
    1.  First, carefully analyze the provided context to see if it directly answers the question.
    2.  If the context provides a good answer, use **only** that context, adapting it for the user's knowledge level.
    3.  If the context is empty or insufficient, use your own general knowledge to provide a complete and accurate response, still tailored to the user's knowledge level.
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

def generate_explanation(question, user_answer, correct_answer, knowledge_level, model):
    """Generates an explanation for an incorrect quiz answer."""
    prompt = f"""
    You are a helpful tutor. A student with a '{knowledge_level}' knowledge level is taking a quiz.
    
    They were asked the following question:
    "{question}"

    The correct answer is: "{correct_answer}"
    They incorrectly chose: "{user_answer}"

    Please provide a brief, encouraging explanation (2-3 sentences) that clarifies the concept. Explain why the correct answer is right and, if relevant, why their choice might be a common misconception.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't generate an explanation at this time. Error: {e}"


# --- Streamlit UI ---
st.set_page_config(page_title="Synapse AI", page_icon="🧠")
# st.title("🧠 AI Learning Partner")

# --- Initialize Session State ---
if "user_info" not in st.session_state:
    st.session_state.user_info = None 
    st.session_state.messages = []
    st.session_state.mode = "Guided Learning Session" 
    st.session_state.processed_file = None 
    st.session_state.in_guided_session = False 
    st.session_state.lesson_plan = None 
    st.session_state.lesson_step = 0 
    st.session_state.quiz_mode = False 
    st.session_state.quiz_questions = None 
    st.session_state.current_question_index = 0 
    st.session_state.score = 0

if "current_goal" not in st.session_state:
    st.session_state.current_goal = None

if "show_login" not in st.session_state:
    st.session_state.show_login = False

# --- User Onboarding ---
if st.session_state.user_info is None:
    # --- NEW: Centered Home Page Layout ---
    st.write("") # Pushes content down for better vertical alignment
    st.write("")

    # Create three columns with the middle one being wider
    title_col1, title_col2 = st.columns([1, 4])

    with title_col1:
        # Place all content within the central column
        st.image(
            "logo.png", 
            width=100, # Set a specific width for the image
        )
    
    with title_col2:
    # Use markdown with CSS for vertical alignment
        st.markdown("""
            <style>
            .title-container {
                display: flex;
                align-items: center;
                height: 80px; /* Match this to your image width for good alignment */
            }
            </style>
            <div class="title-container">
                <h1 style='margin: 0;'>Welcome to Synapse AI</h1>
            </div>
        """, unsafe_allow_html=True)
            
    st.markdown("<h3 style='text-align: center;'>Your Personal AI Learning Partner</h3>", unsafe_allow_html=True)
    
    st.write("") # Add some space

    st.markdown("""
    Unlock a smarter way to learn. Whether you're studying a dense document, exploring a new topic, or preparing for an exam, Synapse AI is here to guide you.
    """)

    st.write("")

    # The button to trigger the login form
    if st.button("Login / Get Started", type="primary", use_container_width=True):
        st.session_state.show_login = True
        st.rerun()

    # --- LOGIN FORM (appears below after button click) ---
    if st.session_state.show_login:
        st.markdown("---")
        
        # Center the login form as well
        _, login_col, _ = st.columns([1, 2, 1])
        with login_col:
            st.subheader("Create or Load Your Profile")
            username = st.text_input("Enter your username:")
            
            if st.button("Start Session", use_container_width=True):
                if username:
                    with st.spinner("Setting up your session..."):
                        st.session_state.user_info = get_or_create_user(username)
                    st.rerun()
                else:
                    st.warning("Please enter a username.")
else:
    # --- MAIN APP AFTER LOGIN ---
    username = st.session_state.user_info['username']
    user_id = st.session_state.user_info['id']
    st.sidebar.success(f"Logged in as **{username}**")

    if "current_session_level" not in st.session_state:
        st.session_state.current_session_level = "Intermediate" # Default value

    # Load chat history once
    if not st.session_state.messages:
        chat_history = get_chat_history(user_id)
        if chat_history: st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

    # Sidebar selectors
    st.session_state.mode = st.sidebar.radio("Choose your learning mode:", ("Guided Learning Session", "General Q&A", "Study a Document"))
    st.sidebar.markdown("---")
    st.session_state.current_session_level = st.sidebar.selectbox(
        "Set knowledge level for this topic:",
        ("Beginner", "Intermediate", "Expert"),
        index=("Beginner", "Intermediate", "Expert").index(st.session_state.current_session_level)
    )

    st.subheader(f"Mode: {st.session_state.mode}")

    # --- MAIN UI CONTROLLER ---
    if st.session_state.quiz_mode:
        # --- QUIZ UI ---
        index = st.session_state.current_question_index; questions = st.session_state.quiz_questions; total_questions = len(questions)
        if index < total_questions:
            st.progress(index / total_questions, text=f"Question {index + 1} of {total_questions}"); st.metric(label="Your Score", value=f"{st.session_state.score} / {total_questions}"); st.markdown("---")
            q = questions[index]
            with st.container(border=True):
                st.subheader(f"Question {index + 1}:"); st.markdown(q['question'])
                if f"answer_submitted_{index}" not in st.session_state:
                    with st.form(key=f"quiz_form_{index}"):
                        user_answer = st.radio("Choose your answer:", options=q['options'], index=None, label_visibility="collapsed")
                        if st.form_submit_button("Submit Answer"):
                            st.session_state[f"user_answer_{index}"] = user_answer; st.session_state[f"answer_submitted_{index}"] = True; st.rerun()
                else:
                    user_answer = st.session_state[f"user_answer_{index}"]
                    if user_answer == q['correct_answer']:
                        st.success("Correct! 🎉")
                        if f"feedback_given_{index}" not in st.session_state: st.balloons(); st.session_state.score += 1
                    else:
                        st.error(f"Not quite. The correct answer was: **{q['correct_answer']}**")
                        if f"feedback_given_{index}" not in st.session_state:
                            with st.spinner("Generating an explanation..."):
                                explanation = generate_explanation(q['question'], user_answer, q['correct_answer'], st.session_state.user_info['knowledge_level'], pro_model)
                                st.info(explanation)
                    st.session_state[f"feedback_given_{index}"] = True
                    if st.button("Next Question"): st.session_state.current_question_index += 1; st.rerun()
        else:
            st.success(f"Quiz complete! Your final score is: {st.session_state.score}/{total_questions}")
            if st.button("End Session"): st.session_state.in_guided_session = False; st.session_state.quiz_mode = False; st.session_state.messages = []; st.rerun()

    elif st.session_state.in_guided_session:
        # --- "TEACH & QUIZ" GUIDED SESSION UI with GOAL TRACKING ---
        plan = st.session_state.lesson_plan
        step = st.session_state.lesson_step
        goal_info = st.session_state.current_goal # Get current goal info
        
        # --- NEW: Display Goal and Progress Bar in Sidebar ---
        if goal_info:
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**Your Goal:** {goal_info['goal']}")
            # Calculate progress based on the step before the quiz
            progress_percent = step / goal_info['total_steps']
            st.sidebar.progress(progress_percent, text=f"Step {step} of {goal_info['total_steps']}")
            st.sidebar.markdown("---")
        
        st.sidebar.markdown("### Lesson Plan")
        for i, sub_topic in enumerate(plan):
            st.sidebar.markdown(f"**➡️ {i+1}. {sub_topic}**" if i == step else f"{i+1}. {sub_topic}")

        # Display the latest explanation
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # This new state tracks if we are teaching or quizzing for the current step
        if 'step_phase' not in st.session_state:
            st.session_state.step_phase = 'teaching'

        # --- TEACHING PHASE ---
        if st.session_state.step_phase == 'teaching':
            if st.button("I'm ready for a quick quiz on this!"):
                with st.spinner("Generating mini-quiz..."):
                    # Use only the last explanation as context for the quiz
                    context = st.session_state.messages[-1]['content']
                    quiz = generate_quiz(context, pro_model, num_questions=2) # Generate a short, 1-question quiz
                    if quiz:
                        st.session_state.quiz_questions = quiz
                        st.session_state.current_question_index = 0
                        st.session_state.step_phase = 'quizzing' # Switch to the quizzing phase
                        st.rerun()
        
        # --- QUIZZING PHASE ---
        elif st.session_state.step_phase == 'quizzing':
            index = st.session_state.current_question_index
            questions = st.session_state.quiz_questions
            total_questions = len(questions)

            if index < total_questions:
                q = questions[index]
                st.info(f"Quick Question: {q['question']}")

                # Use a session state key to track if an answer has been submitted
                if f"mini_answer_submitted_{index}" not in st.session_state:
                    with st.form(key=f"mini_quiz_form_{index}"):
                        user_answer = st.radio("Choose your answer:", options=q['options'], index=None, label_visibility="collapsed")
                        if st.form_submit_button("Submit Answer"):
                            st.session_state[f"mini_user_answer_{index}"] = user_answer
                            st.session_state[f"mini_answer_submitted_{index}"] = True
                            st.rerun()
                else:
                    # This block runs AFTER submission to show feedback
                    user_answer = st.session_state[f"mini_user_answer_{index}"]
                    
                    if user_answer == q['correct_answer']:
                        st.success("Correct! 🎉")
                        if f"mini_feedback_given_{index}" not in st.session_state:
                             st.session_state[f"mini_feedback_given_{index}"] = True
                    else:
                        st.error(f"Not quite. The correct answer was: **{q['correct_answer']}**")
                        # Generate an AI explanation for the wrong answer
                        if f"mini_feedback_given_{index}" not in st.session_state:
                            with st.spinner("Generating an explanation..."):
                                knowledge_level = st.session_state.user_info['knowledge_level']
                                explanation = generate_explanation(q['question'], user_answer, q['correct_answer'], knowledge_level, pro_model)
                                st.info(explanation)
                            st.session_state[f"mini_feedback_given_{index}"] = True
                    
                    # Show a "Continue" button to proceed
                    if st.button("Continue"):
                        st.session_state.current_question_index += 1
                        st.rerun()
            else:
                # This block runs after the mini-quiz for a step is finished
                st.info("Great work on that section!")
                st.session_state.lesson_step += 1

                goal_info = st.session_state.current_goal
                if goal_info:
                    update_goal_progress(goal_info['id'], st.session_state.lesson_step)
                
                if st.session_state.lesson_step < len(plan):
                    # If there are more steps, prepare the next topic
                    with st.spinner("Preparing the next topic..."):
                        next_sub_topic = plan[st.session_state.lesson_step]
                        explanation = explain_sub_topic(next_sub_topic, st.session_state.user_info['knowledge_level'], flash_model)
                        st.session_state.messages.append({"role": "assistant", "content": explanation})
                        st.session_state.step_phase = 'teaching'
                        st.session_state.current_question_index = 0
                    st.rerun()
                else:
                    # If all steps are done, end the guided session and offer the final quiz
                    st.success("You've completed the entire guided lesson! Well done! 🎉")
                    st.session_state.in_guided_session = False
                    
                    if st.button("Take the Final Review Quiz"):
                        with st.spinner("Generating your final quiz..."):
                            conversation_context = " ".join([msg['content'] for msg in st.session_state.messages if msg['role'] == 'assistant'])
                            final_quiz = generate_quiz(conversation_context, pro_model, num_questions=5)
                            if final_quiz:
                                st.session_state.quiz_questions = final_quiz
                                st.session_state.current_question_index = 0
                                st.session_state.score = 0
                                st.session_state.quiz_mode = True # Activate the main, comprehensive quiz UI
                                st.rerun()
    else:
        # --- DEFAULT LOBBY / FREE-FORM CHAT UI ---
        if st.session_state.mode == "Guided Learning Session":
            st.info("Enter a topic, and the AI will create a structured lesson for you.")
            topic = st.text_input("What topic would you like a guided lesson on?")
             # --- NEW: Add input for the learning goal ---
            goal = st.text_input("What is your goal for this session?", placeholder="e.g., Understand the basics for a class")
            if st.button("Start Guided Session"):
                if topic and goal:
                    with st.spinner("Creating a lesson plan..."):
                        plan = generate_lesson_plan(topic, pro_model)
                        if plan:
                            st.session_state.current_goal = create_learning_goal(user_id, topic, goal, len(plan))
                            st.session_state.lesson_plan = plan
                            st.session_state.in_guided_session = True
                            st.session_state.lesson_step = 0
                            st.session_state.messages = []
                            first_sub_topic = plan[0]
                            explanation = explain_sub_topic(first_sub_topic, st.session_state.user_info['knowledge_level'], flash_model)
                            st.session_state.messages.append({"role": "assistant", "content": f"Great! I've prepared a lesson on '{topic}'. Here is the first part:"})
                            st.session_state.messages.append({"role": "assistant", "content": explanation})
                            st.rerun()
                else:
                    st.warning("Please enter a topic and a goal.")

        elif st.session_state.mode == "General Q&A":
            for message in st.session_state.messages:
                with st.chat_message(message["role"]): st.markdown(message["content"])
            if prompt := st.chat_input("Ask a free-form question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_topic_answer(prompt, st.session_state.messages, flash_model, knowledge_level=st.session_state.current_session_level)
                        save_message(user_id, "user", prompt); save_message(user_id, "assistant", response); st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        elif st.session_state.mode == "Study a Document":
            st.sidebar.header("Upload Your Document")
            uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file and uploaded_file.name != st.session_state.processed_file:
                with st.spinner("Processing file..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: tmp_file.write(uploaded_file.getvalue()); tmp_file_path = tmp_file.name
                    success = process_file(tmp_file_path, uploaded_file.name)
                    os.remove(tmp_file_path)
                    if success: st.session_state.processed_file = uploaded_file.name; st.success(f"Processed '{uploaded_file.name}'!"); st.session_state.messages = []; st.rerun()
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]): st.markdown(message["content"])
            if prompt := st.chat_input("Ask a question about the document..."):
                if not st.session_state.processed_file: st.warning("Please upload a document first.")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = generate_answer(prompt, pro_model, knowledge_level=st.session_state.current_session_level,file_name=st.session_state.processed_file)
                            save_message(user_id, "user", prompt, st.session_state.processed_file); save_message(user_id, "assistant", response, st.session_state.processed_file); st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

        