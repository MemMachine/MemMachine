### Purpose of the change

Adding the strands-agent example application to demonstrate MemMachine integration in a multi-agent system.

### Description

This PR adds the strands-agent example application, which showcases a multi-agent system (AdvisorBuddy, MemoryKeeper, NewsScout) integrated with MemMachine for persistent memory storage and retrieval.

The application includes:
- Streamlit UI for user interaction
- Multi-agent architecture with specialized agents
- MemMachine integration for persistent user context and preferences
- News fetching capabilities via Tavily API

### Type of change

- [x] New feature (non-breaking change which adds functionality)

### How Has This Been Tested?

- [x] Manual verification (list step-by-step instructions)

**Test Steps:**
1. Start MemMachine server: `docker-compose up -d`
2. Navigate to strands-agent: `cd examples/strands-agent`
3. Create virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`
6. Test storing and recalling user information through the UI

### Checklist

- [x] I have signed the commit(s) within this pull request
- [x] My code follows the style guidelines of this project
- [x] I have performed a self-review of my own code
- [x] I have commented my code
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings

### Screenshots/Gifs

N/A

### Further comments

None
