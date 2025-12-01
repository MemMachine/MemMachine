# Multi-Morning: True Multi-Agent News System 🎙️☕

A sophisticated multi-agent AI system for personalized news briefings with persistent memory!

## Architecture 🏗️

This is a **true multi-agent system** where specialized agents coordinate to deliver personalized news:

### The Three Agents

**1.  AdvisorBuddy** (Main Orchestrator)
- Your friendly news host and conversation partner
- Coordinates and delegates to specialist agents
- Maintains warm, engaging conversation flow
- Handles multi-step workflows and personalization

**2.  MemoryKeeper** (Memory Specialist)
- Stores and recalls all user information
- Manages preferences, investments, interests, and conversation history
- Provides semantic search across past conversations
- Ensures proper user isolation and data privacy

**3.  NewsScout** (News Research Specialist)
- Finds and curates news articles from web sources
- Supports category-based and topic-specific searches
- Delivers trending headlines
- Formats articles with titles, URLs, and snippets

### Agent Coordination

```
User Input → AdvisorBuddy (Orchestrator)
              ├─→ ask_memory_keeper("Store user's input")
              │     └─→ MemoryKeeper → store_user_info() → MemMachine API
              │
              └─→ ask_news_scout("Find news")
                    └─→ NewsScout → search_news_by_topic() → Tavily API
```

## Features ✨

- 🤖 **Multi-Agent Architecture**: Three specialized agents working in coordination
- 🧠 **MemMachine Integration**: Persistent memory across sessions with proper user isolation
- 💬 **Conversational Intelligence**: Natural language understanding and context awareness
- 🎯 **Smart Delegation**: Orchestrator routes requests to appropriate specialist agents
- 😊 **Personality**: Warm, engaging, human-like interactions
- 💾 **Learning System**: Builds comprehensive user profiles over time
- 🔒 **User Isolation**: Each user gets their own memory namespace

### What Makes It Fun
1. **True Persistence**: Remembers you even after app restarts (via MemMachine)
2. **Personalization**: Remembers your name and uses it
3. **Context Awareness**: Tracks what you're reading and offers more
4. **Smart Responses**: Doesn't search when you're just chatting
5. **Engaging Tone**: Talks like a friendly morning show host
6. **Progress Tracking**: Shows how many articles remaining
7. **Learning System**: Builds your profile over time

## Quick Start

### 1. Start MemMachine Server (Required for Persistent Memory)
```bash
# From MemMachine root directory
cd MemMachine
./memmachine-compose.sh

# Verify it's running
curl http://localhost:8080/api/v2/health
```

### 2. Set up environment
```bash
cd examples/strands-agent
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt

# Install MemMachine client (from local source)
pip install -e ../../
```

### 3. Set API keys and configuration
```bash
export TAVILY_API_KEY='your-tavily-api-key'
export MEMMACHINE_URL='http://localhost:8080'  # Optional, this is default
export MEMMACHINE_ORG_ID='strands-agent'       # Optional, this is default
export MEMMACHINE_PROJECT_ID='morning-brief'   # Optional, this is default
```

### 4. Run!

#### Streamlit UI (Recommended)
```bash
streamlit run app.py
```

**🔑 For Persistent Memory Across Sessions:**

**Option 1: URL Parameter** (Easiest!)
```bash
# Open in browser with your name as user ID:
http://localhost:8501?user=user_name
```
This way, your memories persist even after restarting Streamlit!

**Option 2: Set in UI**
1. Run `streamlit run app.py`
2. In the sidebar, enter your name
3. Click "Set Persistent ID"
4. Your memories will now persist!

#### CLI Version
```bash
# With specific user ID (remembers you!)
python3 main.py user_name

# With default user
python3 main.py
```

> **⚠️ Important**: Without a persistent user ID, your memories are lost when the app restarts!

## Example Conversation

### Session 1 (First Time)
```
You: hi
Agent: Hey there! ☕ Good to see you! 🎙️
      I'm your Morning Brief host...
      
You: my name is Anirudh
Agent: Nice to meet you, Anirudh! 🎙️☕
      [Stores in MemMachine]

You: tech news
Agent: **Microsoft, OpenAI reach deal...**
      https://...
      
      [Article snippet]
      
      💬 Want more? I've got more! 

You: yes
Agent: **Nvidia's $1 billion stake...**
      [Next article automatically]
      
      💬 Want more? I've got more! 

You: exit
```

### Session 2 (Coming Back Later)
```
✅ MemMachine connected - persistent memory enabled!
📚 Loaded memories for Anirudh

You: hi
Agent: Hey there, Anirudh! ☕ Good to see you! 🎙️
      [Remembered your name from last session!]
      
You: what did I ask about before?
Agent: Based on your history, you were interested in tech news!
      Want me to catch you up on the latest? 🚀
```

## Technical Details

### Agent Files

```
agents/
├── advisor_buddy.py       # Main orchestrator (delegation tools)
├── memory_keeper.py        # Memory management specialist
├── news_scout.py           # News research specialist
└── __init__.py
```

### How Delegation Works

**AdvisorBuddy** uses delegation tools to coordinate with specialists:

```python
@tool
def ask_memory_keeper(request: str) -> str:
    """Delegate memory operations to MemoryKeeper agent"""
    result = memory_keeper(request)
    return extract_response(result)

@tool
def ask_news_scout(request: str) -> str:
    """Delegate news research to NewsScout agent"""
    result = news_scout(request)
    return extract_response(result)
```

### User Isolation

Each user gets a unique namespace:
```python
group_id = f"morning-brief-{user_id}"  # Unique per user!
```

This ensures memories never leak between users.

### Extending with More Agents

Want to add a new specialist? Here's the pattern:

#### 1. Create the Specialist Agent
```python
# agents/portfolio_analyst.py
from strands import Agent, tool

@tool
def analyze_portfolio(holdings: str) -> dict:
    """Analyze investment portfolio and provide insights"""
    # Your analysis logic here
    return {"status": "success", "analysis": "..."}

@tool
def get_stock_recommendations(risk_profile: str) -> dict:
    """Get personalized stock recommendations"""
    # Your recommendation logic here
    return {"status": "success", "recommendations": [...]}

def make_portfolio_analyst() -> Agent:
    system_prompt = """You are PortfolioAnalyst 📊, a financial specialist.
    
    YOUR ROLE:
    - Analyze investment portfolios
    - Provide stock recommendations
    - Track market trends
    
    TOOLS:
    - analyze_portfolio(): Evaluate holdings
    - get_stock_recommendations(): Suggest investments
    """
    
    return Agent(
        system_prompt=system_prompt,
        tools=[analyze_portfolio, get_stock_recommendations]
    )
```

#### 2. Add Delegation Tool to AdvisorBuddy
```python
# In advisor_buddy_v3.py, inside make_advisor_buddy():

from agents.portfolio_analyst import make_portfolio_analyst

# Initialize the specialist
portfolio_analyst = make_portfolio_analyst()

@tool
def ask_portfolio_analyst(request: str) -> str:
    """Delegate investment analysis to PortfolioAnalyst agent"""
    result = portfolio_analyst(request)
    if hasattr(result, 'text'):
        return result.text
    return str(result)

# Add to the tools list:
advisor = Agent(
    system_prompt=system_prompt,
    tools=[
        ask_memory_keeper,
        ask_news_scout,
        ask_portfolio_analyst  # ✅ New delegation tool!
    ]
)
```

**That's it!** AdvisorBuddy now intelligently delegates investment questions to your new PortfolioAnalyst specialist. No pattern matching needed - the LLM decides when to use each agent!

## Tools Available

### News & Search Tools
- **tavily_search**: AI-powered web search
- **http_request**: Direct HTTP calls for APIs
- **web_search**: DuckDuckGo fallback
- **fetch_news**: Curated news fetching

### MemMachine Memory Tools (v2 API 🧠)
- **store_memory**: Store any conversation or fact using MemMachine v2 API
- **search_memories**: Search past conversations and preferences (episodic & semantic memory)
- **get_user_context**: Load complete user profile
- **store_user_preference**: Save specific preferences
- **check_memmachine_health**: Verify MemMachine connection

**Note**: This system uses MemMachine v2 API with `MemMachineClient`, `Project`, and `Memory` classes. The API requires `org_id` and `project_id` for proper project isolation.

## Configuration

### Categories
Edit `agents/news_scout.py`:
```python
CATEGORY_QUERIES = {
    "tech": "latest technology news",
    "finance": "business and stock market news",
    "sports": "sports news headlines",
    "politics": "global political news",
    "movies": "entertainment and hollywood movie news",
    # Add your own!
}
```

### Personality
Edit `agents/advisor_buddy.py`:
```python
# Change greetings, responses, tone, etc.
```

## Advanced Features

### Session Persistence
Want to save conversation history?
```python
# Add to AdvisorBuddy.__init__
self.conversation_history = []

# Add to __call__
self.conversation_history.append({
    "user": user_input,
    "assistant": response,
    "timestamp": time.time()
})
```

### Multi-Turn Context
Already implemented! The agent:
- Remembers your name
- Tracks current topic
- Caches fetched articles
- Knows which article index you're on

### Custom Tools
Add your own tools:
```python
from strands import tool

@tool
def my_custom_tool(param: str) -> dict:
    # Your logic here
    return {"result": "..."}

# Add to agent
agent = Agent(tools=[my_custom_tool, ...])
```

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add secrets: `TAVILY_API_KEY`
4. Deploy!

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Troubleshooting

**Issue**: Agent searches for casual chat
- **Fix**: Update `_looks_like_news_request()` patterns

**Issue**: "Yes" doesn't show more articles
- **Fix**: Make sure conversation context is preserved (single instance)

**Issue**: Agent repeats same article
- **Fix**: Check that `article_index` is incrementing





