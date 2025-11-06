# tools/tavily.py
from strands import tool
from tavily import TavilyClient
import os

# Initialize client lazily to avoid errors during import
client = None

def get_client():
    global client
    if client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable is not set. "
                "Please set it with: export TAVILY_API_KEY='your-key-here'"
            )
        client = TavilyClient(api_key=api_key)
    return client

@tool
def tavily_search(query: str, max_results: int = 5, topic: str = "general") -> list[dict]:
    """
    Search the web using Tavily AI search API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
        topic: Search topic - "general" for broad searches (stocks, weather, data), 
               "news" for news articles, "research" for academic content (default: "general")
    
    Returns:
        List of search results with title, url, and snippet
    """
    client = get_client()
    resp = client.search(query, max_results=max_results, topic=topic)
    results = []
    for r in resp.get("results", []):
        results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content", "")[:200] if r.get("content") else "",
            "score": r.get("score", 0)  # Relevance score
        })
    return results
