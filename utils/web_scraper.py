from duckduckgo_search import DDGS
from newspaper import Article

def search_and_scrape(query: str):
    """
    Performs a web search, scrapes the first result using newspaper3k,
    and returns the cleaned text content and the source URL.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
            if not results:
                return "Could not find any web results.", None
            
            url = results[0]['href']

        # Use newspaper3k to download and parse the article
        article = Article(url)
        article.download()
        article.parse()
        
        # Return the article's text and the source URL
        return article.text[:2000], url
        
    except Exception as e:
        print(f"Web scraping with newspaper3k failed: {e}")
        return "", None