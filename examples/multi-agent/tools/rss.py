from strands import tool
import feedparser

@tool
def rss_feed(url: str, limit: int = 5) -> list[dict]:
    """
    Parse an RSS feed and return articles.
    """
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries[:limit]:
        results.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "summary": entry.get("summary", "")
        })
    return results

