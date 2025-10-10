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
from utils.web_scraper import search_and_scrape

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

def generate_enhanced_topic_answer(query, chat_history, model, knowledge_level):
    """Generates an enhanced answer using a web search and cites the source."""
    with st.spinner(f"Searching the web for '{query}'..."):
        # The function now returns two values
        scraped_context, source_url = search_and_scrape(query)

    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # NEW: The prompt is updated to include the source URL instruction
    prompt = f"""
    You are an expert AI Learning Partner. Your user's knowledge level is '{knowledge_level}'.
    
    Review the fresh context scraped from a web article:
    ---
    Web Context:
    {scraped_context}
    ---

    Based on the web context and your own knowledge, provide a comprehensive answer to the user's question: "{query}"

    **IMPORTANT:** At the very end of your answer, you MUST cite the source of the web context.
    Format it on a new line exactly like this, using Markdown for the link: 
    *Source: [{source_url}]({source_url})*
    """
    
    try:
        response = model.generate_content(prompt)
        # If no web context was found, provide a fallback message
        if not scraped_context:
            return "I couldn't find a relevant article on the web, but here is an answer from my general knowledge:\n\n" + response.text
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI Learning Partner", page_icon="🧠")
st.title("🧠 AI Learning Partner")

# --- Initialize Session State ---
if "user_info" not in st.session_state:
    st.session_state.user_info = None; st.session_state.messages = []; st.session_state.mode = "Guided Learning Session"; st.session_state.processed_file = None; st.session_state.in_guided_session = False; st.session_state.lesson_plan = None; st.session_state.lesson_step = 0; st.session_state.quiz_mode = False; st.session_state.quiz_questions = None; st.session_state.current_question_index = 0; st.session_state.score = 0

# --- User Onboarding ---
if st.session_state.user_info is None:
    st.markdown("Welcome! Please enter a username to start your session.")
    username = st.text_input("Username")
    knowledge_level = st.selectbox("What is your knowledge level on most topics?", ("Beginner", "Intermediate", "Expert"))
    if st.button("Start Session"):
        if username:
            with st.spinner("Setting up..."):
                st.session_state.user_info = get_or_create_user(username, knowledge_level)
            st.rerun()
        else:
            st.warning("Please enter a username.")
else:
    # --- MAIN APP AFTER LOGIN ---
    username = st.session_state.user_info['username']; user_id = st.session_state.user_info['id']
    st.sidebar.success(f"Logged in as **{username}**")

    # Load chat history once
    if not st.session_state.messages:
        chat_history = get_chat_history(user_id)
        if chat_history: st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

    # Sidebar selectors
    st.session_state.mode = st.sidebar.radio("Choose your learning mode:", ("Guided Learning Session", "General Q&A", "Study a Document"))
    st.sidebar.markdown("---")
    levels = ("Beginner", "Intermediate", "Expert"); current_level = st.session_state.user_info.get('knowledge_level', 'Intermediate'); current_index = levels.index(current_level)
    new_level = st.sidebar.selectbox("Adjust your knowledge level:", levels, index=current_index)
    if new_level != current_level:
        st.session_state.user_info['knowledge_level'] = new_level
        st.sidebar.info(f"Knowledge level set to **{new_level}**.")

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
        # --- "TEACH & QUIZ" GUIDED SESSION UI ---
        plan = st.session_state.lesson_plan
        step = st.session_state.lesson_step
        
        # Display the lesson plan in the sidebar for context
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

            if index < len(questions):
                # Display the mini-quiz question using a form
                q = questions[index]
                with st.form(key=f"mini_quiz_form_{index}"):
                    st.info(f"Quick Question: {q['question']}")
                    user_answer = st.radio("Choose:", q['options'], index=None, label_visibility="collapsed")
                    if st.form_submit_button("Submit"):
                        if user_answer == q['correct_answer']:
                            st.success("Correct!")
                        else:
                            st.error(f"The correct answer was: {q['correct_answer']}")
                        st.session_state.current_question_index += 1
                        st.rerun()
            else:
                # This block runs after the mini-quiz for a step is finished
                st.info("Great work on that section!")
                st.session_state.lesson_step += 1 # Advance to the next lesson step
                
                if st.session_state.lesson_step < len(plan):
                    # If there are more steps, prepare the next topic
                    with st.spinner("Preparing the next topic..."):
                        next_sub_topic = plan[st.session_state.lesson_step]
                        explanation = explain_sub_topic(next_sub_topic, st.session_state.user_info['knowledge_level'], flash_model)
                        st.session_state.messages.append({"role": "assistant", "content": explanation})
                        st.session_state.step_phase = 'teaching' # Go back to the teaching phase for the new step
                        st.session_state.current_question_index = 0 # Reset quiz index
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
            if st.button("Start Guided Session"):
                if topic:
                    with st.spinner("Creating a lesson plan..."):
                        plan = generate_lesson_plan(topic, pro_model)
                        if plan:
                            st.session_state.lesson_plan = plan; st.session_state.in_guided_session = True; st.session_state.lesson_step = 0; st.session_state.messages = []
                            first_sub_topic = plan[0]
                            explanation = explain_sub_topic(first_sub_topic, st.session_state.user_info['knowledge_level'], flash_model)
                            st.session_state.messages.append({"role": "assistant", "content": f"Great! I've prepared a lesson on '{topic}'. Here is the first part:"})
                            st.session_state.messages.append({"role": "assistant", "content": explanation}); st.rerun()

        # In app.py, replace the General Q&A block

        elif st.session_state.mode == "General Q&A":
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Show the "Enhance with Web Search" button only after an assistant has responded
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                if st.button("Enhance with Web Search 🔍"):
                    with st.chat_message("assistant"):
                        with st.spinner("Searching the web and refining the answer..."):
                            # The last user question is two messages back in the history
                            last_user_query = st.session_state.messages[-2]['content']
                            knowledge_level = st.session_state.user_info['knowledge_level']
                            
                            enhanced_response = generate_enhanced_topic_answer(last_user_query, st.session_state.messages, flash_model, knowledge_level)
                            
                            # Replace the last, basic answer with the new, enhanced one
                            st.session_state.messages[-1] = {"role": "assistant", "content": enhanced_response}
                            save_message(user_id, "assistant", enhanced_response) # Optionally update the DB
                            st.rerun()

            # The main chat input
            if prompt := st.chat_input("Ask a free-form question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        knowledge_level = st.session_state.user_info['knowledge_level']
                        # Call the BASIC function for the initial, fast response
                        response = generate_topic_answer(prompt, st.session_state.messages, flash_model, knowledge_level)
                        save_message(user_id, "user", prompt)
                        save_message(user_id, "assistant", response)
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun() # Rerun to show the new "Enhance" button

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
                            response = generate_answer(prompt, pro_model, st.session_state.user_info['knowledge_level'])
                            save_message(user_id, "user", prompt, st.session_state.processed_file); save_message(user_id, "assistant", response, st.session_state.processed_file); st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})