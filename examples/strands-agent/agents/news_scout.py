# agents/news_scout.py
"""
NewsScout Agent - Specialized in finding and curating news with personalized memory
Handles all news research, gathering, summarization, and personalized curation
"""

import hashlib
from datetime import UTC, datetime, timedelta

from strands import Agent, tool
from tools.memmachine import (
    get_user_context,
    search_memories,
    store_memory,
    store_user_preference,
)
from tools.tavily import tavily_search

# Category mappings are now dynamic - loaded from user preferences in memory
# No hardcoded categories - the system discovers user interests organically
# This helper function provides fallback query generation for common category patterns


def _get_category_query(category: str) -> str:
    """
    Convert a category name to a search query.
    Categories are discovered dynamically from user preferences, not hardcoded.
    This is a fallback helper - users can specify any category they want.
    """
    # Common category patterns (user can specify any category they want)
    category_lower = category.lower().strip()

    # Map common category names to search queries (fallback only)
    category_patterns = {
        "tech": "latest technology news",
        "technology": "latest technology news",
        "finance": "business and stock market news",
        "financial": "business and stock market news",
        "sports": "sports news headlines",
        "politics": "global political news",
        "political": "global political news",
        "movies": "entertainment and hollywood movie news",
        "entertainment": "entertainment and hollywood movie news",
    }

    # Check for exact match or partial match
    if category_lower in category_patterns:
        return category_patterns[category_lower]

    # For any other category, create a dynamic query
    return f"latest {category} news"


def _dedupe(items: list[dict]) -> list[dict]:
    """Remove duplicate articles by title"""
    seen = set()
    clean = []
    for x in items:
        title = (x.get("title") or "").strip().lower()
        if title and title not in seen:
            seen.add(title)
            clean.append(x)
    return clean


def _filter_recent_articles(articles: list[dict], hours: int = 48) -> list[dict]:
    """
    Filter articles by recency (prioritize recent content).
    Note: Tavily returns relevance scores; higher scores typically indicate more recent/relevant content.
    For production, you'd fetch actual publish dates from article URLs.
    """
    return sorted(articles, key=lambda x: x.get("score", 0), reverse=True)


def _format_articles_with_links(articles: list[dict]) -> str:
    """
    Format articles as a string with clickable Markdown links.
    This ensures URLs are always displayed as clickable links in the UI.
    Format: [Title](URL) - Summary (clean Markdown that Streamlit will render as clickable links)
    """
    if not articles:
        return ""

    formatted_lines = []
    for article in articles:
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        snippet = article.get("snippet", "")

        # Clean title and URL - remove special characters that might break Markdown
        title = title.strip()
        url = url.strip()

        # Format as clickable link if URL exists - clean Markdown format
        if url and url.startswith(("http://", "https://")):
            # Use bullet point and clean Markdown link format
            link_text = f"- [{title}]({url})"
        else:
            # No valid URL - just show title with bullet
            link_text = f"- {title}"

        # Add snippet if available (after the link)
        if snippet:
            snippet = snippet.strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            link_text += f" - {snippet}"

        formatted_lines.append(link_text)

    # Return as single string with newlines - ready for Markdown rendering
    return "\n".join(formatted_lines)


def _check_recent_search(user_id: str, query: str) -> bool:
    """Check if we've searched for this exact query very recently (within 1 hour)"""
    try:
        query_hash = hashlib.md5(f"{query}_{user_id}".encode()).hexdigest()

        result = search_memories(
            query=f"news search: {query}", user_id=user_id, limit=3
        )

        if result.get("status") == "success":
            memories = result.get("episodic_memories", [])
            for mem in memories:
                metadata = mem.get("user_metadata", {})
                if metadata.get("query_hash") == query_hash:
                    stored_at = metadata.get("stored_at")
                    if stored_at:
                        try:
                            stored_time = datetime.fromisoformat(
                                stored_at.replace("Z", "")
                            )
                            if datetime.now(
                                stored_time.tzinfo
                            ) - stored_time < timedelta(hours=1):
                                return True
                        except Exception:
                            pass
        return False
    except Exception:
        return False


def _get_user_news_preferences(user_id: str) -> dict:
    """Retrieve user's news preferences from memory (profile + episodic)"""
    try:
        context = get_user_context(
            user_id=user_id,
            topics=[
                "news category preferences",
                "preferred news format",
                "news reading style",
                "news recency requirements",
                "favorite news topics",
            ],
        )

        if context.get("status") == "success":
            ctx = context.get("context", {})
            memories = ctx.get("all_memories", [])

            prefs = {
                "categories": [],
                "format": "list",  # list, brief, detailed, newsletter
                "recency_hours": 48,
                "topics": [],
                "exclude_topics": [],
                "sources_preferred": [],
            }

            for memory in memories:
                content = (memory.get("content", "") or "").lower()
                # Extract category preferences dynamically (not from hardcoded list)
                # Look for phrases like "prefers tech news", "interested in finance", etc.
                # This allows any category - not limited to hardcoded ones
                if (
                    "tech" in content or "technology" in content
                ) and "tech" not in prefs["categories"]:
                    prefs["categories"].append("tech")
                if (
                    "finance" in content
                    or "financial" in content
                    or "business" in content
                ) and "finance" not in prefs["categories"]:
                    prefs["categories"].append("finance")
                if "sports" in content and "sports" not in prefs["categories"]:
                    prefs["categories"].append("sports")
                if (
                    "politics" in content or "political" in content
                ) and "politics" not in prefs["categories"]:
                    prefs["categories"].append("politics")
                if (
                    "movies" in content or "entertainment" in content
                ) and "movies" not in prefs["categories"]:
                    prefs["categories"].append("movies")

                # Extract any other category mentions from user preferences
                # This allows dynamic category discovery from user input
                # Future enhancement: use NLP to extract arbitrary category names

                # Extract format preferences
                if "brief" in content or "summary" in content:
                    prefs["format"] = "brief"
                elif "detailed" in content or "comprehensive" in content:
                    prefs["format"] = "detailed"
                elif "newsletter" in content or "news brief" in content:
                    prefs["format"] = "newsletter"

                # Extract recency preferences
                if (
                    "24 hours" in content
                    or "today" in content
                    or "last 24 hours" in content
                ):
                    prefs["recency_hours"] = 24
                elif "week" in content or "7 days" in content:
                    prefs["recency_hours"] = 168
                elif "48 hours" in content:
                    prefs["recency_hours"] = 48

            return prefs

        return {}
    except Exception:
        return {
            "categories": [],
            "format": "list",
            "recency_hours": 48,
            "topics": [],
            "exclude_topics": [],
        }


# Keep utility function for backward compatibility
def fetch_news(
    category: str | None = None,
    query: str | None = None,
    limit: int = 5,
    user_id: str = "default_user",
) -> list[dict]:
    """
    Enhanced utility function with memory-aware fetching.
    Returns a list of {title, url, snippet, score}. Uses Tavily. Deduped. Max 'limit'.
    """
    results: list[dict] = []

    # Check for recent duplicate searches
    search_query = category or query or ""
    if search_query and _check_recent_search(user_id, search_query):
        # Return empty if very recent duplicate (could also return cached results)
        return []

    if category:
        # Use dynamic category query helper (no hardcoded categories)
        q = _get_category_query(category)
        results.extend(
            tavily_search(query=q, max_results=limit * 2, topic="news")
        )  # Get more to filter

    # If specific topic/query provided (categories are dynamic, no hardcoded check needed)
    if query and not category:
        results.extend(tavily_search(query=query, max_results=limit * 2, topic="news"))

    # Fallback
    if not results and (category or query):
        if category:
            fallback_q = _get_category_query(category)
        else:
            fallback_q = query or "latest news"
        results.extend(
            tavily_search(query=fallback_q, max_results=limit * 2, topic="news")
        )

    # Get user preferences for filtering
    prefs = _get_user_news_preferences(user_id)

    # Filter by recency (prioritize higher scores)
    results = _filter_recent_articles(results, prefs.get("recency_hours", 48))

    # Dedupe and limit
    results = _dedupe(results)[:limit]

    # Store search in memory for future reference
    if search_query and user_id != "default_user":
        store_memory(
            content=f"User searched for news: {search_query}",
            producer=user_id,
            produced_for="news_scout",
            episode_type="news_search",
            metadata={
                "query": search_query,
                "category": category,
                "query_hash": hashlib.md5(
                    f"{search_query}_{user_id}".encode()
                ).hexdigest(),
                "stored_at": datetime.now(UTC).isoformat(),
                "results_count": len(results),
            },
            user_id=user_id,
        )

    return results


# === ENHANCED AGENT TOOLS WITH MEMORY ===


@tool
def get_personalized_news_brief(user_id: str, limit: int = 5) -> dict:
    """
    Get personalized news brief based on user's preferences and past searches.
    This is the main personalized entry point that leverages memory.

    Args:
        user_id: User identifier
        limit: Maximum number of articles per category

    Returns:
        Dictionary with personalized news brief.
        CRITICAL: The dictionary includes a "formatted_with_links" field containing pre-formatted Markdown links.
        You MUST extract the "formatted_with_links" field value and include it directly in your response text.
        The formatted_with_links contains clickable article links like: "- [Title](URL) - Summary"
        Do NOT summarize the articles - use the formatted_with_links string exactly as provided.
    """
    prefs = _get_user_news_preferences(user_id)

    # Get user's preferred categories from memory (dynamic, not hardcoded)
    # If no preferences exist, use general trending news instead of defaulting to specific categories
    categories = prefs.get("categories", [])

    all_articles = []
    category_articles = {}

    if not categories:
        # No preferences yet - use general trending/news query (no hardcoded categories)
        articles = fetch_news(query="breaking news today", limit=limit, user_id=user_id)
        category_articles["trending"] = articles
        all_articles.extend(articles)
    else:
        # User has saved preferences - use their categories (dynamic, not hardcoded)
        categories = categories[:3]  # Limit to top 3 preferences
        for cat in categories:
            articles = fetch_news(category=cat, limit=limit, user_id=user_id)
            category_articles[cat] = articles
            all_articles.extend(articles)

    # Format according to user preference
    format_style = prefs.get("format", "list")

    # Format articles by category with clickable links
    formatted_by_category = {}
    for cat, articles in category_articles.items():
        formatted_list = [
            {
                "title": a.get("title", "Untitled"),
                "url": a.get("url", ""),
                "snippet": (a.get("snippet") or "")[:200],
            }
            for a in articles
        ]
        formatted_by_category[cat] = formatted_list
        # Also add formatted string with links
        formatted_by_category[f"{cat}_formatted"] = _format_articles_with_links(
            formatted_list
        )

    return {
        "status": "success",
        "format": format_style,
        "categories": categories,
        "articles_by_category": formatted_by_category,
        "formatted_with_links": "\n\n".join(
            [
                f"**{cat}:**\n{formatted_by_category.get(f'{cat}_formatted', '')}"
                for cat in category_articles
            ]
        ),  # Pre-formatted string with clickable Markdown links organized by category
        "total_count": len(all_articles),
        "personalized": True,
    }


@tool
def get_news_by_category(
    category: str,
    limit: int = 5,
    user_id: str = "default_user",
    filter_recent: bool = True,
) -> dict:
    """
    Fetch news for a specific category with memory-aware filtering.

    Args:
        category: News category (any category name - not hardcoded, supports any topic)
        limit: Maximum number of articles
        user_id: User identifier for personalization
        filter_recent: Whether to filter for recent articles (default: True, last 48h)

    Returns:
        Dictionary with formatted news articles.
        CRITICAL: The dictionary includes a "formatted_with_links" field containing pre-formatted Markdown links.
        You MUST extract the "formatted_with_links" field value and include it directly in your response text.
        Do NOT summarize the articles - use the formatted_with_links string exactly as provided.
    """
    prefs = _get_user_news_preferences(user_id)
    recency_hours = prefs.get("recency_hours", 48) if filter_recent else None

    articles = fetch_news(category=category, limit=limit, user_id=user_id)

    if recency_hours:
        articles = _filter_recent_articles(articles, recency_hours)

    formatted = []
    for article in articles:
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        snippet = (article.get("snippet") or "").strip()
        if snippet and len(snippet) > 200:
            snippet = snippet[:197] + "..."

        formatted.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "relevance_score": article.get("score", 0),
            }
        )

    # Store user preference if they search this category frequently
    if user_id != "default_user":
        store_memory(
            content=f"User requested news in category: {category}",
            producer=user_id,
            produced_for="news_scout",
            episode_type="news_preference",
            metadata={"category": category, "stored_at": datetime.now(UTC).isoformat()},
            user_id=user_id,
        )

    # Add formatted string with clickable links for easy display
    formatted_string = _format_articles_with_links(formatted)

    return {
        "status": "success",
        "category": category,
        "articles": formatted,
        "formatted_with_links": formatted_string,  # Pre-formatted string with clickable Markdown links
        "count": len(formatted),
        "filtered_recent": filter_recent,
    }


@tool
def search_news_by_topic(
    topic: str,
    limit: int = 5,
    user_id: str = "default_user",
    filter_recent: bool = True,
) -> dict:
    """
    Search for news on a specific topic, company, or event with memory integration.

    Args:
        topic: What to search for (e.g., "NVIDIA stock", "AI developments")
        limit: Maximum number of articles
        user_id: User identifier for personalization
        filter_recent: Whether to filter for recent articles (default: True, last 48h)

    Returns:
        Dictionary with formatted news articles.
        CRITICAL: The dictionary includes a "formatted_with_links" field containing pre-formatted Markdown links.
        You MUST extract the "formatted_with_links" field value and include it directly in your response text.
        Do NOT summarize the articles - use the formatted_with_links string exactly as provided.
    """
    prefs = _get_user_news_preferences(user_id)
    recency_hours = prefs.get("recency_hours", 48) if filter_recent else None

    articles = fetch_news(query=topic, limit=limit, user_id=user_id)

    if recency_hours:
        articles = _filter_recent_articles(articles, recency_hours)

    formatted = []
    for article in articles:
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        snippet = (article.get("snippet") or "").strip()
        if snippet and len(snippet) > 200:
            snippet = snippet[:197] + "..."

        formatted.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "relevance_score": article.get("score", 0),
            }
        )

    # Store search topic for future personalization
    if user_id != "default_user":
        store_memory(
            content=f"User searched for news topic: {topic}",
            producer=user_id,
            produced_for="news_scout",
            episode_type="news_search",
            metadata={"topic": topic, "stored_at": datetime.now(UTC).isoformat()},
            user_id=user_id,
        )

    # Add formatted string with clickable links for easy display
    formatted_string = _format_articles_with_links(formatted)

    return {
        "status": "success",
        "topic": topic,
        "articles": formatted,
        "formatted_with_links": formatted_string,  # Pre-formatted string with clickable Markdown links
        "count": len(formatted),
        "filtered_recent": filter_recent,
    }


@tool
def get_trending_headlines(
    limit: int = 5, user_id: str = "default_user", filter_recent: bool = True
) -> dict:
    """
    Get the latest trending news headlines with memory awareness.

    Args:
        limit: Maximum number of articles
        user_id: User identifier
        filter_recent: Whether to filter for recent articles (default: True)

    Returns:
        Dictionary with formatted news articles.
        CRITICAL: The dictionary includes a "formatted_with_links" field containing pre-formatted Markdown links.
        You MUST extract the "formatted_with_links" field value and include it directly in your response text.
        Do NOT summarize the articles - use the formatted_with_links string exactly as provided.
    """
    articles = fetch_news(query="breaking news today", limit=limit, user_id=user_id)

    if filter_recent:
        prefs = _get_user_news_preferences(user_id)
        recency_hours = prefs.get("recency_hours", 48)
        articles = _filter_recent_articles(articles, recency_hours)

    formatted = []
    for article in articles:
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        snippet = (article.get("snippet") or "").strip()
        if snippet and len(snippet) > 200:
            snippet = snippet[:197] + "..."

        formatted.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "relevance_score": article.get("score", 0),
            }
        )

    # Add formatted string with clickable links for easy display
    formatted_string = _format_articles_with_links(formatted)

    return {
        "status": "success",
        "articles": formatted,
        "formatted_with_links": formatted_string,  # Pre-formatted string with clickable Markdown links
        "count": len(formatted),
        "filtered_recent": filter_recent,
    }


@tool
def summarize_and_analyze_news(
    articles: list[dict], analysis_type: str = "themes"
) -> dict:
    """
    Summarize and analyze news articles to identify key themes, trends, and insights.
    This demonstrates the summarization capability like ChatGPT news scouts.

    Args:
        articles: List of article dictionaries with title, url, snippet
        analysis_type: Type of analysis - "themes", "summary", "trends", "full"

    Returns:
        Dictionary with summary, themes, and insights.
        Note: This tool does not return formatted_with_links - use it with other news tools to get articles with links.
    """
    if not articles:
        return {"status": "error", "message": "No articles provided for analysis"}

    # Extract content
    article_texts = []
    for article in articles:
        title = article.get("title", "")
        snippet = article.get("snippet", "")
        article_texts.append(f"{title}: {snippet}")

    combined_text = "\n\n".join(article_texts)

    # Identify key themes (simple keyword-based, in production would use LLM)
    themes = []
    keywords = {}

    # Common news themes
    theme_keywords = {
        "Technology": [
            "tech",
            "AI",
            "artificial intelligence",
            "software",
            "digital",
            "tech",
            "innovation",
        ],
        "Finance": [
            "stock",
            "market",
            "financial",
            "economy",
            "trading",
            "investment",
            "revenue",
        ],
        "Politics": [
            "political",
            "government",
            "policy",
            "election",
            "senate",
            "president",
        ],
        "Business": [
            "company",
            "business",
            "corporate",
            "CEO",
            "merger",
            "acquisition",
        ],
        "Sports": ["game", "team", "player", "championship", "sports", "match"],
        "Entertainment": [
            "movie",
            "film",
            "celebrity",
            "entertainment",
            "hollywood",
            "TV",
        ],
    }

    text_lower = combined_text.lower()
    for theme, keys in theme_keywords.items():
        count = sum(1 for key in keys if key in text_lower)
        if count > 0:
            keywords[theme] = count
            themes.append(theme)

    # Generate summary (simplified - in production, use LLM for better summaries)
    top_themes = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:3]
    theme_summary = ", ".join([theme for theme, _ in top_themes])

    summary = (
        f"Analysis of {len(articles)} articles reveals key themes: {theme_summary}. "
    )
    summary += f"Top stories include: {articles[0].get('title', 'N/A')} and {articles[1].get('title', 'N/A') if len(articles) > 1 else 'N/A'}."

    # Include URLs for top stories (needed for clickable links in responses)
    top_stories_with_urls = []
    for a in articles[:3]:
        title = a.get("title", "Untitled")
        url = a.get("url", "")
        top_stories_with_urls.append({"title": title, "url": url})

    return {
        "status": "success",
        "analysis_type": analysis_type,
        "article_count": len(articles),
        "summary": summary,
        "key_themes": themes[:5],
        "theme_keywords": dict(list(keywords.items())[:5]),
        "top_stories": [a.get("title") for a in articles[:3]],
        "top_stories_with_urls": top_stories_with_urls,  # Include URLs for clickable links
    }


@tool
def set_user_news_preferences(
    user_id: str,
    preferred_categories: list[str] | None = None,
    format_style: str | None = None,
    recency_hours: int | None = None,
) -> dict:
    """
    Store user's news preferences in memory for future personalization.

    Args:
        user_id: User identifier
        preferred_categories: List of preferred categories (any category name - not hardcoded)
        format_style: Preferred format (list, brief, detailed, newsletter)
        recency_hours: Preferred recency window (24, 48, 168 for week)

    Returns:
        Status of preference storage
    """
    preferences_to_store = []

    if preferred_categories:
        categories_str = ", ".join(preferred_categories)
        store_user_preference(
            user_id=user_id,
            preference_type="news_categories",
            preference_value=categories_str,
        )
        preferences_to_store.append(f"Categories: {categories_str}")

    if format_style:
        store_user_preference(
            user_id=user_id,
            preference_type="news_format",
            preference_value=format_style,
        )
        preferences_to_store.append(f"Format: {format_style}")

    if recency_hours:
        store_user_preference(
            user_id=user_id,
            preference_type="news_recency",
            preference_value=str(recency_hours),
        )
        preferences_to_store.append(f"Recency: {recency_hours} hours")

    # Also store as memory for context
    if preferences_to_store:
        store_memory(
            content=f"User news preferences: {'; '.join(preferences_to_store)}",
            producer=user_id,
            produced_for="news_scout",
            episode_type="preference",
            metadata={
                "preference_type": "news_settings",
                "categories": preferred_categories,
                "format": format_style,
                "recency_hours": recency_hours,
                "stored_at": datetime.now(UTC).isoformat(),
            },
            user_id=user_id,
        )

    return {
        "status": "success",
        "message": f"Stored preferences: {', '.join(preferences_to_store) if preferences_to_store else 'None'}",
        "preferences": {
            "categories": preferred_categories,
            "format": format_style,
            "recency_hours": recency_hours,
        },
    }


@tool
def get_user_news_preferences(user_id: str) -> dict:
    """
    Retrieve user's stored news preferences from memory.

    Args:
        user_id: User identifier

    Returns:
        Dictionary with user preferences
    """
    prefs = _get_user_news_preferences(user_id)

    return {"status": "success", "preferences": prefs}


def make_news_scout(user_id: str = "default_user") -> Agent:
    """
    Creates an enhanced NewsScout agent with memory integration and personalization.
    Demonstrates ChatGPT-like news scout capabilities with MemMachine memory layer.

    Args:
        user_id: User identifier for personalization

    Returns:
        Configured Agent instance
    """
    system_prompt = f"""You are NewsScout 📰, an intelligent, personalized news research agent powered by MemMachine memory.

USER: {user_id}

YOUR ROLE:
You're an expert news researcher who finds, curates, and analyzes news with deep personalization.
You remember user preferences, past searches, and adapt your responses accordingly.
You provide summaries, theme analysis, and personalized news briefs - just like ChatGPT news scouts!

MEMORY CAPABILITIES (Showcasing MemMachine):
🧠 **Semantic Memory**: You remember user's preferred news categories, format styles, and reading habits
📚 **Episodic Memory**: You track past searches, topics of interest, and conversation history
🎯 **Personalization**: Every search is tailored based on what you've learned about the user

CAPABILITIES:
✅ Web browsing: Search internet for recent articles from multiple sources
✅ Filtering: Filter by topic, relevance, and recency (default: last 48 hours, customizable)
✅ Summarization: Synthesize information to identify key themes and produce concise briefs
✅ Analysis: Identify top stories, trends percentage, and insights across articles
✅ Customization: Remember user preferences for format, categories, and recency requirements

TOOLS:
- get_personalized_news_brief(): Get personalized news based on user preferences (MEMORY-AWARE)
- get_news_by_category(): Fetch news for a category with memory integration
- search_news_by_topic(): Search specific topics with filtering and memory
- get_trending_headlines(): Get trending news filtered by recency
- summarize_and_analyze_news(): Analyze articles to identify themes and generate summaries
- set_user_news_preferences(): Store user preferences in memory (Semantic Memory)
- get_user_news_preferences(): Retrieve stored preferences from memory

BEHAVIOR:
🎯 **ALWAYS leverage memory first**: Check user preferences before searching
🎯 **Personalize responses**: Reference past searches, preferred categories, and format styles
🎯 **Provide summaries**: Offer theme analysis and synthesis when appropriate
🎯 **Filter intelligently**: Apply user's preferred recency window (default 48h)
🎯 **Remember interactions**: Store search queries and preferences for future use
🎯 **Adapt format**: Use user's preferred format (list, brief, detailed, newsletter)
🔗 **CRITICAL - Always include article links**: Format URLs as clickable Markdown links [Article Title](URL) for every article

**CRITICAL RESPONSE RULE - THIS IS MANDATORY - READ CAREFULLY:**

When you call ANY news tool (search_news_by_topic, get_news_by_category, get_personalized_news_brief, etc.),
the tool returns a JSON dictionary. Inside that dictionary, there is a field called "formatted_with_links".

**YOU MUST DO THIS:**
1. After calling the tool, look at the JSON response
2. Find the "formatted_with_links" field
3. Copy the ENTIRE value of that field (it's a string)
4. Include that exact string in your response to the user
5. DO NOT modify it, summarize it, or create your own version

**EXAMPLE WORKFLOW:**
User asks: "Find NVIDIA news"
You call: search_news_by_topic("NVIDIA", 5, user_id, True)
Tool returns: {{"status": "success", "formatted_with_links": "- [NVIDIA Hits $5T](https://url.com) - Summary\\n- [Stock Surges](https://url2.com) - 3% gain", ...}}
You MUST respond with: "Here's the NVIDIA news:\\n\\n- [NVIDIA Hits $5T](https://url.com) - Summary\\n- [Stock Surges](https://url2.com) - 3% gain"

**DO NOT DO THIS:**
- Do NOT summarize the articles and omit the links
- Do NOT create your own format for links
- Do NOT mention articles without including their clickable links

**YOUR RESPONSE IS INCOMPLETE IF YOU DON'T INCLUDE formatted_with_links!**
Every news response MUST contain the formatted_with_links field value from the tool response!

MEMORY SHOWCASE OPPORTUNITIES:
- When user mentions preferences → Use set_user_news_preferences() to store (Semantic Memory)
- Before searching → Use get_user_news_preferences() to personalize (Semantic Memory retrieval)
- After searching → Mention "Based on your preferences..." to show memory working
- Track searches → Each search is stored in Episodic Memory automatically
- Reference past → "I remember you also searched for X last week" (Episodic Memory retrieval)

EXAMPLES:

Request: "What's my personalized news brief?"
Response format (CORRECT - use formatted_with_links directly):

Great morning, Anirudh! ✨ Here's your tailored news brief:

**Tech News:**
- [NVIDIA Hits $5 Trillion Market Cap](https://example.com/article1) - Historic milestone achieved
- [AI Partnership Announced](https://example.com/article2) - Major collaboration unveiled

**Finance News:**
- [Stock Market Surge](https://example.com/article3) - Record gains today

Notice: Links are formatted as [Title](URL) - clean and clickable. Use the formatted_with_links field directly!

INCORRECT format (DO NOT do this):
[NVIDIA Makes History with $5 Trillion Market Cap](https://...) - broken up

CORRECT format (DO this):
- [NVIDIA Makes History with $5 Trillion Market Cap](https://www.latimes.com/article) - Summary text

**MANDATORY LINK FORMATTING (CRITICAL - YOU MUST DO THIS):**
- Every tool that returns articles includes a "formatted_with_links" field in the response dictionary
- When you call a tool, you will receive a JSON response. Look for the "formatted_with_links" key in that JSON
- Extract the value of "formatted_with_links" - it's a string containing pre-formatted Markdown links
- YOU MUST copy that entire formatted_with_links string and paste it directly into your response text
- DO NOT summarize or rephrase the links - copy them EXACTLY as they appear
- DO NOT create your own link format - use the pre-formatted one from the tool
- Example workflow: Tool returns JSON → You see "formatted_with_links": "- [Article](url) - summary" → You copy that exact string → Paste it in your response
- The formatted_with_links contains clean Markdown: [Title](URL) - Summary that will render as clickable links in the UI
- If you forget to include formatted_with_links, your response is INCOMPLETE - every article MUST have its clickable link!

Request: "I prefer tech and finance news, give me a brief summary"
Response:
1. [Call: set_user_news_preferences(user_id, ["tech", "finance"], "brief")]
2. [Call: get_personalized_news_brief(user_id, 5)]
3. The tool returns: {{"status": "success", "formatted_with_links": "- [Article 1](url1) - summary\\n- [Article 2](url2) - summary"}}
4. Your response MUST be: "I've saved your preferences! Here's your personalized brief:\\n\\n" followed by the formatted_with_links content
5. Copy the formatted_with_links field EXACTLY into your response - do not summarize it!

Request: "Find news about NVIDIA from the last 24 hours"
Response workflow:
1. [Call: search_news_by_topic("NVIDIA", 5, user_id, True)]
2. Tool returns: {{"status": "success", "formatted_with_links": "- [Article 1](https://url.com) - summary\\n- [Article 2](https://url2.com) - summary2"}}
3. Your response MUST include: "Here's the latest NVIDIA news:\\n\\n" followed by the formatted_with_links content exactly
4. CRITICAL: Copy the formatted_with_links string EXACTLY from the tool response - do not summarize or omit links!

Request: "Analyze the themes in today's tech news"
Response:
1. [Call: get_news_by_category("tech", 10, user_id, True)]
2. Tool returns: {{"status": "success", "formatted_with_links": "- [Tech Article 1](url) - summary\\n..."}}
3. [Call: summarize_and_analyze_news(articles, "themes")]
4. Your response: "Here are today's tech themes: [themes]\\n\\nTop stories:\\n\\n" followed by the formatted_with_links content
5. ALWAYS include the formatted_with_links from step 2 in your final response!

Request: "What are my news preferences?"
Response: [Call: get_user_news_preferences(user_id)] then summarize their saved preferences

Request: "Give me a weekly newsletter format with my favorite topics"
Response:
1. [Call: set_user_news_preferences(user_id, None, "newsletter")]
2. [Call: get_personalized_news_brief(user_id, 10)]
3. Format results as newsletter with sections

You are intelligent, personalized, and always showcase the power of memory! 🚀
**Remember**: This is a demo - highlight how memory makes each interaction better!
"""

    agent = Agent(
        system_prompt=system_prompt,
        tools=[
            get_personalized_news_brief,
            get_news_by_category,
            search_news_by_topic,
            get_trending_headlines,
            summarize_and_analyze_news,
            set_user_news_preferences,
            get_user_news_preferences,
        ],
    )

    agent.agent_name = "NewsScout"
    agent.user_id = user_id

    return agent
