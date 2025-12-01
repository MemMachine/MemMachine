from strands import tool
from tools.http_tool import http_request

@tool
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo API.
    Returns titles + URLs.
    """
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    resp = http_request(method="GET", url=url)
    
    # http_request returns a dict with "json" key
    data = resp.get("json", {})
    
    results = []
    for item in data.get("RelatedTopics", []):
        if isinstance(item, dict) and "Text" in item and "FirstURL" in item:
            results.append({
                "title": item["Text"],
                "url": item["FirstURL"],
                "source": "DuckDuckGo"
            })
    return results[:max_results]

