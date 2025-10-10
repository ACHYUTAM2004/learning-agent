# In utils/web_scraper.py

import streamlit as st  # Import Streamlit for logging
from duckduckgo_search import DDGS
from newspaper import Article
import nltk

# --- NLTK Data Check ---
# Newspaper3k depends on the 'punkt' tokenizer. 
# This checks if it's available and downloads it if not.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    st.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    st.info("Download complete.")


def search_and_scrape(query: str):
    """
    Performs a web search with live debugging output.
    """
    try:
        st.info("1/4 - Starting web search...")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
        
        if not results:
            st.warning("2/4 - Search did not return any results.")
            return "Could not find any web results.", None
        
        url = results[0]['href']
        st.info(f"2/4 - Search successful. Found URL: {url}")

        article = Article(url)
        
        st.info("3/4 - Downloading article content...")
        article.download()
        
        st.info("4/4 - Parsing article text...")
        article.parse()
        
        if not article.text:
            st.warning("4/4 - Parsing failed. No text could be extracted from the page.")
            return "", None

        st.success("Scraping complete!")
        return article.text[:2000], url
        
    except Exception as e:
        st.error(f"Web scraping failed with an exception: {e}")
        return "", None