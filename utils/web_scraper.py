import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

def search_and_scrape(query: str):
    """
    Performs a web search, scrapes the first result, and returns
    the cleaned text content AND the source URL.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
            if not results:
                return "Could not find any web results.", None
            
            first_result_url = results[0]['href']

        response = requests.get(first_result_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        scraped_text = ' '.join([p.get_text() for p in paragraphs])
        
        # Return both the text and the URL
        return scraped_text[:2000], first_result_url
        
    except Exception as e:
        print(f"Web scraping failed: {e}")
        return "", None # Return empty string and None on failure