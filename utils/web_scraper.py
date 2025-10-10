# In utils/web_scraper.py

import streamlit as st
from duckduckgo_search import DDGS
from newspaper import Article, Config  # 👈 Import Config
import nltk

# --- NLTK Data Check ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    st.info("Download complete.")


def search_and_scrape(query: str):
    """
    Performs a web search with a browser User-Agent to avoid being blocked.
    """
    try:
        # --- NEW: Set up a browser-like configuration ---
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 10
        # --- END NEW ---

        st.info("1/4 - Starting web search...")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
        
        if not results:
            st.warning("2/4 - Search did not return any results.")
            return "Could not find any web results.", None
        
        url = results[0]['href']
        st.info(f"2/4 - Search successful. Found URL: {url}")

        # --- MODIFIED: Pass the config to the Article ---
        article = Article(url, config=config)
        
        st.info("3/4 - Downloading article content...")
        article.download()
        
        st.info("4/4 - Parsing article text...")
        article.parse()
        
        if not article.text:
            st.warning("4/4 - Parsing failed. No text could be extracted.")
            return "", None

        st.success("Scraping complete!")
        return article.text[:2000], url
        
    except Exception as e:
        st.error(f"Web scraping failed with an exception: {e}")
        return "", None