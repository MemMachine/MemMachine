# Multi-Agent Architecture Overview 🤖

## System Design

This is a **true multi-agent system** where specialized AI agents coordinate to deliver personalized news briefings.

```
┌─────────────────────────────────────────────────────────┐
│                  USER INTERACTION                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   AdvisorBuddy 🎙️    │
         │   (Orchestrator)      │
         │                       │
         │  Delegation Tools:    │
         │  • ask_memory_keeper  │
         │  • ask_news_scout     │
         └───────┬───────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ MemoryKeeper │   │  NewsScout   │
│     🧠       │   │     📰       │
├──────────────┤   ├──────────────┤
│ Tools:       │   │ Tools:       │
│ • store_info │   │ • by_category│
│ • recall     │   │ • by_topic   │
│ • search     │   │ • trending   │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
  MemMachine    Tavily API
  (localhost:8080)   (Web Search)
```

## The Three Agents

### 1. 🎙️ AdvisorBuddy (Main Orchestrator)

**File**: `agents/advisor_buddy.py`

**Role**: Main conversation partner and coordinator
- Maintains warm, engaging personality
- Understands user intent from natural language
- Delegates to specialist agents
- Synthesizes multi-agent responses
- Handles complex multi-step workflows

**Delegation Tools**:
```python
@tool
def ask_memory_keeper(request: str) -> str:
    """Delegate memory operations"""
    
@tool
def ask_news_scout(request: str) -> str:
    """Delegate news research"""
```

**Example Workflow**:
```
User: "I invested in NVIDIA yesterday for $1450"

AdvisorBuddy:
  1. Recognizes: User shared investment info + likely wants news
  2. Calls: ask_memory_keeper("Store user invested in NVIDIA for $1450")
  3. Calls: ask_news_scout("Search for NVIDIA stock news")
  4. Synthesizes: "Nice! I've saved that. Here's the latest NVIDIA news..."
```

---

### 2. 🧠 MemoryKeeper (Memory Specialist)

**File**: `agents/memory_keeper.py`

**Role**: Manages all user context and conversation history
- Stores user information (name, preferences, investments, interests)
- Recalls past conversations
- Semantic search across user history
- Ensures proper user isolation

**Tools**:
```python
@tool
def store_user_info(user_id, info_type, info_value)
    
@tool
def recall_user_info(user_id, query=None)
    
@tool
def search_user_memories(user_id, query, limit=5)
```

**Integration**: MemMachine API (HTTP)
- Endpoint: `http://localhost:8080`
- User isolation via unique `group_id` per user
- Persistent storage across sessions

**Example**:
```
AdvisorBuddy → ask_memory_keeper("Store user's name is Alex")
  ↓
MemoryKeeper → store_user_info(user_id="test_user", 
                                info_type="name", 
                                info_value="Alex")
  ↓
MemMachine API → POST /v1/memories
  ↓
Storage: group_id = "morning-brief-test_user"
```

---

### 3. 📰 NewsScout (News Research Specialist)

**File**: `agents/news_scout.py`

**Role**: Finds and curates news articles with memory-aware personalization
- Category-based news (any category - dynamically discovered from user preferences)
- Topic-specific searches (companies, events, keywords)
- Trending headlines
- Personalized news briefs based on user preferences
- News analysis and summarization
- Article formatting with clickable Markdown links
- Memory integration for tracking user preferences and past searches

**Tools**:
```python
@tool
def get_personalized_news_brief(user_id: str, limit: int = 5) -> dict
    """Get personalized news based on user's stored preferences"""

@tool
def get_news_by_category(category: str, limit: int = 5, user_id: str = "default_user", filter_recent: bool = True) -> dict
    """Fetch news for a category with memory-aware filtering"""
    
@tool
def search_news_by_topic(topic: str, limit: int = 5, user_id: str = "default_user", filter_recent: bool = True) -> dict
    """Search specific topics with filtering and memory integration"""
    
@tool
def get_trending_headlines(limit: int = 5, user_id: str = "default_user", filter_recent: bool = True) -> dict
    """Get trending news filtered by recency"""

@tool
def summarize_and_analyze_news(articles: List[Dict], analysis_type: str = "themes") -> dict
    """Analyze articles to identify themes and generate summaries"""

@tool
def set_user_news_preferences(user_id: str, preferred_categories: Optional[List[str]] = None, format_style: Optional[str] = None, recency_hours: Optional[int] = None) -> dict
    """Store user's news preferences in memory"""

@tool
def get_user_news_preferences(user_id: str) -> dict
    """Retrieve user's stored news preferences from memory"""
```

**Integration**: Tavily Search API
- AI-powered web search
- News-specific queries
- Deduplication and formatting

**Example**:
```
AdvisorBuddy → ask_news_scout("Find NVIDIA stock news")
  ↓
NewsScout → search_news_by_topic("NVIDIA stock", limit=5, user_id="test_user", filter_recent=True)
  ↓
  ├─ Checks user preferences from memory (recency window, format)
  ├─ Searches Tavily API for NVIDIA stock articles
  ├─ Filters by recency (default: last 48 hours)
  ├─ Formats articles with clickable Markdown links
  └─ Stores search in memory for future personalization
  ↓
Return: {
  "status": "success",
  "topic": "NVIDIA stock",
  "articles": [...],
  "formatted_with_links": "- [NVIDIA Surges 15%](https://...) - Summary\n- [Article 2](https://...) - Summary",
  "count": 5
}
```

---

## Agent Coordination Examples

### Example 1: Simple Memory Storage
```
User: "My name is Anirudh"

AdvisorBuddy:
  ├─ Recognizes: User sharing personal info
  ├─ Calls: ask_memory_keeper("Store that user's name is Anirudh")
  │    └─→ MemoryKeeper → store_user_info("anirudh", "name", "Anirudh")
  │          └─→ MemMachine API stores with group_id="morning-brief-anirudh"
  └─ Responds: "Great to meet you, Anirudh! I'll remember that. 🎙️"
```

### Example 2: Simple News Request
```
User: "What's happening in tech?"

AdvisorBuddy:
  ├─ Recognizes: News request for tech category
  ├─ Calls: ask_news_scout("Find the latest tech news")
  │    └─→ NewsScout → get_news_by_category("tech", limit=5, user_id="anirudh", filter_recent=True)
  │          ├─ Checks user preferences from memory
  │          ├─ Searches Tavily API for tech news
  │          ├─ Filters by recency
  │          └─ Returns formatted articles with clickable links
  └─ Responds: "Here are the hottest tech headlines! 💻
               [formatted articles with clickable Markdown links]"
```

### Example 3: Complex Multi-Agent Workflow
```
User: "Tell me about my NVIDIA investment and get me news on it"

AdvisorBuddy:
  ├─ Recognizes: Two-part request (recall + fetch)
  │
  ├─ Step 1: Recall investment
  │    └─→ ask_memory_keeper("What did user say about NVIDIA investment?")
  │          └─→ MemoryKeeper → search_user_memories(query="NVIDIA investment")
  │                └─→ MemMachine returns: "User invested $1450 in NVIDIA"
  │
  ├─ Step 2: Fetch news
  │    └─→ ask_news_scout("Search for NVIDIA stock news")
  │          └─→ NewsScout → search_news_by_topic("NVIDIA stock", limit=5, user_id="test_user", filter_recent=True)
  │                ├─ Checks user preferences from memory
  │                ├─ Searches Tavily API for NVIDIA stock articles
  │                ├─ Formats articles with clickable Markdown links
  │                └─ Stores search in memory for future personalization
  │
  └─ Synthesizes response:
      "You invested $1450 in NVIDIA! Here's what's happening with them:
       [article 1]
       [article 2]
       ..."
```

---

## Key Design Principles

### 1. **Separation of Concerns**
Each agent has a specific domain of expertise:
- AdvisorBuddy: Conversation & coordination
- MemoryKeeper: Context & persistence
- NewsScout: Information retrieval

### 2. **LLM-Powered Delegation**
No hardcoded pattern matching! The orchestrator uses LLM reasoning to decide:
- Which agent to delegate to
- What to ask them
- How to combine their responses

### 3. **User Isolation**
Each user gets a unique namespace:
```python
group_id = f"morning-brief-{user_id}"
```
Prevents memory leakage between users.

### 4. **Extensibility**
Adding a new specialist agent requires only:
1. Create the agent file with tools
2. Add a delegation tool to AdvisorBuddy
3. Update the tools list

No complex routing logic needed!

### 5. **Graceful Degradation**
- MemMachine offline? → Session-only memory
- Tavily API unavailable? → Suggest direct sources
- Agent error? → Friendly fallback message

---

## Adding New Agents

### Pattern for New Specialist

**Step 1**: Create the specialist agent
```python
# agents/portfolio_analyst.py
from strands import Agent, tool

@tool
def analyze_holdings(investments: str) -> dict:
    # Analysis logic
    pass

def make_portfolio_analyst() -> Agent:
    return Agent(
        system_prompt="You are PortfolioAnalyst 📊...",
        tools=[analyze_holdings]
    )
```

**Step 2**: Add delegation tool to orchestrator
```python
# In advisor_buddy.py
from agents.portfolio_analyst import make_portfolio_analyst

portfolio_analyst = make_portfolio_analyst()

@tool
def ask_portfolio_analyst(request: str) -> str:
    """Delegate investment analysis"""
    result = portfolio_analyst(request)
    return extract_response(result)
```

**Step 3**: Add to tools list
```python
advisor = Agent(
    system_prompt=system_prompt,
    tools=[
        ask_memory_keeper,
        ask_news_scout,
        ask_portfolio_analyst  # ✅ New!
    ]
)
```

**Done!** The LLM automatically learns when to use the new agent.

---

## Technical Stack

- **Framework**: Strands SDK (LLM-powered agents)
- **LLM**: Claude Sonnet (via Strands)
- **Memory**: MemMachine (self-hosted, Docker)
- **Search**: Tavily API
- **Language**: Python 3.10+
- **UI**: Streamlit (web) + CLI

---

## File Structure

```
multi-agent/
├── agents/
│   ├── advisor_buddy.py        # 🎙️ Main orchestrator
│   ├── memory_keeper.py        # 🧠 Memory specialist
│   ├── news_scout.py           # 📰 News specialist
│   └── __init__.py
├── tools/
│   ├── memmachine.py           # MemMachine API client
│   ├── tavily.py               # Tavily search wrapper
│   ├── http_tool.py            # HTTP utilities
│   ├── websearch.py            # Web search utilities
│   ├── rss.py                  # RSS feed utilities
│   └── __init__.py
├── main.py                     # CLI entry point
├── app.py                      # Streamlit UI
├── README.md                   # User documentation
└── MULTI_AGENT_ARCHITECTURE.md # This file!
```

---

## Benefits of This Architecture

### ✅ **Maintainability**
- Each agent is self-contained
- Clear separation of concerns
- Easy to test individual agents

### ✅ **Scalability**
- Add new specialists without modifying existing code
- Parallel agent execution possible
- Independent scaling of components

### ✅ **Intelligence**
- LLM decides routing (no brittle pattern matching)
- Adaptive to user phrasing
- Handles complex multi-step workflows

### ✅ **User Experience**
- Natural conversation flow
- Personalized responses
- Persistent context across sessions

---

## Future Enhancements

### Potential New Agents
- **PortfolioAnalyst** 📊: Investment analysis and recommendations
- **WeatherBuddy** 🌤️: Local weather and forecasts
- **CalendarAssistant** 📅: Schedule management
- **ResearchAgent** 🔬: Deep dives on complex topics
- **TrendSpotter** 📈: Identify emerging trends

### Advanced Features
- **Multi-turn delegation**: Agents delegating to each other
- **Parallel execution**: Multiple agents working simultaneously
- **Confidence scoring**: Agents indicate certainty
- **Feedback loops**: Agents learn from user reactions

---

## Conclusion

This multi-agent architecture provides a **scalable, maintainable, and intelligent** foundation for building sophisticated AI assistants. The delegation pattern makes it easy to extend with new capabilities while maintaining clean code and excellent user experience.

🎉 **Now you have a TRUE multi-agent system!**

