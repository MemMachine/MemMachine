# HippoSync - AI Chat with Persistent Memory

A production-ready AI chat application demonstrating MemMachine V2's persistent memory capabilities.

## Overview

HippoSync showcases how to build an AI assistant that remembers user context across multiple conversations using MemMachine's advanced memory architecture.

## Features

- 🧠 **Cross-Chat Memory**: Conversations persist across sessions
- 🤖 **Multi-Provider AI**: OpenAI GPT-4, Anthropic Claude, Google Gemini
- 👥 **Team Collaboration**: Shared project workspaces
- 📄 **Document Processing**: Upload and discuss files
- 🔐 **Secure Authentication**: JWT-based user management

## Architecture
```
React Frontend ←→ FastAPI Backend ←→ MemMachine V2
                                    ├── PostgreSQL (vectors)
                                    └── Neo4j (graph)
```

## MemMachine Integration

### Storing Episodic Memory
```python
import requests

def add_memory(user_email: str, content: str):
    response = requests.post(
        "http://localhost:8080/api/v2/memories",
        json={
            "org_id": f"user-{user_email}",
            "project_id": "personal",
            "agent_id": "web-assistant",
            "content": content
        }
    )
    return response.json()
```

### Storing Semantic Facts
```python
def add_semantic_memory(user_email: str, fact: str):
    response = requests.post(
        "http://localhost:8080/api/v2/memories/semantic/add",
        json={
            "org_id": f"user-{user_email}",
            "project_id": "personal",
            "content": fact,
            "memory_type": "semantic"
        }
    )
    return response.json()
```

### Searching Across Conversations
```python
def search_memories(user_email: str, query: str):
    response = requests.post(
        "http://localhost:8080/api/v2/memories/search",
        json={
            "org_id": f"user-{user_email}",
            "project_id": "personal",
            "query": query,
            "top_k": 20,
            "search_episodic": True,
            "search_semantic": True
        }
    )
    return response.json()
```


## Key Implementation Files

In the full repository:
- `backend/app/memmachine_client.py` - MemMachine V2 client
- `backend/app/routes/chat.py` - Chat with memory integration
- `backend/app/utils/memory.py` - Fact extraction utilities

## Memory Organization
```
user-{email}/
├── personal/              # Personal conversations
│   ├── thread-1/         # Individual chat threads
│   ├── thread-2/
│   └── ...
└── project-{id}/         # Team projects
    └── shared memory
```

## Example Usage

**First conversation:**
```
User: "My name is Sarah and I'm a software engineer"
Assistant: "Nice to meet you, Sarah! How can I help you today?"
[Memory stored: User's name is Sarah, occupation is software engineer]
```

**New conversation (different day):**
```
User: "What projects would be good for someone in my field?"
Assistant: "As a software engineer, I'd recommend..."
[Retrieved memory: User is a software engineer]
```

## Tech Stack

**Backend:**
- FastAPI
- SQLAlchemy
- MemMachine V2 API

**Frontend:**
- React 18
- Vite
- Tailwind CSS

**Infrastructure:**
- Docker
- PostgreSQL (via MemMachine)
- Neo4j (via MemMachine)

## Deployment Environments

- ✅ Local Development (Docker Compose)
- ✅ AWS EC2 (Production)
- ✅ Docker Swarm (Scalable)


## Author

**Viranshu Paruparla**
- GitHub: [@Viranshu-30](https://github.com/Viranshu-30)


