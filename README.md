# 🧠 Synapse AI - Your Personal AI Learning Partner

Synapse AI is an intelligent, multi-modal learning assistant built with Streamlit and powered by Google's Gemini models. It's designed to help students, researchers, and lifelong learners deeply understand complex topics through personalized, interactive sessions.

Instead of just retrieving information, Synapse AI acts as a proactive tutor by creating guided lessons, quizzing your understanding, and adapting its explanations to your personal knowledge level.

## ✨ Key Features

  * **Multi-Modal Learning:** Study content from various sources:
      * **PDFs:** Upload academic papers, textbooks, or any document.
      * **YouTube Videos:** Paste a URL to learn from a video's transcript.
      * **General Topics:** Discuss any subject using the AI's general knowledge, enhanced with live web searches.
  * **Personalized & Adaptive:**
      * **Secure User Authentication:** Remembers you and your past conversations securely with Supabase Auth.
      * **Adaptive Explanations:** Tailors the complexity of its answers (Beginner, Intermediate, Expert) to your specific knowledge level for each topic.
  * **Guided Learning Sessions:**
      * The AI generates a structured, step-by-step lesson plan for any topic.
      * Includes **Mastery-Based Learning** with mini-quizzes after each step to reinforce concepts.
  * **Interactive Quizzing:**
      * Comprehensive final quizzes to test overall understanding.
      * **Instant AI-Powered Feedback** on incorrect answers to help you learn from mistakes.
  * **Goal Tracking:** Set a goal for your learning session and track your progress with a visual progress bar.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

  * Python 3.9+
  * A [Supabase](https://supabase.com/) account (for database and authentication)
  * A [Google AI API Key](https://aistudio.google.com/) (for Gemini models)
  * A [Serper API Key](https://serper.dev/) (for web search)

### Installation & Setup

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up the Python Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Supabase Setup:**

      * Create a new project on Supabase.
      * Go to the **SQL Editor** and run the SQL queries to create the `users`, `conversations`, `documents`, and `learning_goals` tables.
      * Go to **Authentication \> Providers** and enable the **Email** provider.
      * Go to **Authentication \> Policies** and enable **Row Level Security (RLS)** on all tables, creating the necessary policies for user access.

4.  **Environment Variables:**

      * The live version uses Streamlit Secrets. For local development, create a `.env` file in the root directory and add your credentials:

    <!-- end list -->

    ```
    SUPABASE_URL="YOUR_SUPABASE_URL"
    SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    SERPER_API_KEY="YOUR_SERPER_API_KEY"
    ```

5.  **Run the App:**

    ```bash
    streamlit run app.py
    ```

## Usage Guide

1.  **Sign Up / Login:** Create an account or log in using your email and password.
2.  **Choose a Mode** from the sidebar:
      * **Guided Learning Session:** Enter a topic (e.g., "Quantum Computing") and a goal. The AI will generate a lesson plan and walk you through it step-by-step with mini-quizzes.
      * **General Q\&A:** Have a free-form chat on any topic. Use the "Enhance with Web Search" button to get up-to-date information with sources.
      * **Study a Document / YouTube Video:** Upload a PDF or paste a YouTube URL. The AI will process the content, and you can then ask specific questions about it.
3.  **Set Your Knowledge Level:** Use the dropdown in the sidebar to change your knowledge level for the current topic at any time, and the AI will immediately adapt its explanations.
