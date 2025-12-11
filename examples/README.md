# MemMachine Agents

This directory contains specialized AI agents that integrate with the MemMachine system. Each agent is designed to handle specific domains and use cases, providing tailored query construction and memory management capabilities. These agents leverage MemMachine's memory system to provide context-aware, personalized responses across various domains.

## Overview

The agents system is built on a modular architecture with:
- **Base Query Constructor**: Abstract base class for all query constructors
- **Specialized Agents**: Domain-specific implementations (CRM, Financial Analyst, Health Assistant, etc.)
- **FastAPI Servers**: RESTful APIs for each agent with comprehensive endpoints
- **Slack Integration**: Real-time communication capabilities for CRM workflows
- **Streamlit Frontend**: Interactive web interface for testing and demonstration

## Architecture

```
examples/
├── base_query_constructor.py         # Base class for query constructors
├── default_query_constructor.py      # Default/general-purpose query constructor
├── example_server.py                 # Example FastAPI server implementation
├── crm/                              # CRM-specific agent
│   ├── crm_server.py                 # CRM FastAPI server
│   ├── query_constructor.py          # CRM query constructor
│   ├── slack_server.py               # Slack integration for CRM
│   ├── slack_service.py              # Slack service utilities
│   └── README.md                     # CRM-specific documentation
├── financial_analyst/                # Financial analysis agent
│   ├── financial_server.py           # Financial analyst FastAPI server
│   └── query_constructor.py          # Financial query constructor
├── health_assistant/                 # Health and wellness assistant agent
│   ├── health_server.py              # Health assistant FastAPI server
│   └── query_constructor.py          # Health query constructor
├── writing_assistant/                # Writing assistant agent
│   ├── writing_assistant_server.py   # Writing assistant FastAPI server
│   ├── README.md                     # Writing assistant-specific documentation
│   └── query_constructor.py          # Writing query constructor
└── frontend/                         # Streamlit web interface
    ├── app.py                        # Main Streamlit application
    ├── llm.py                        # LLM integration
    ├── gateway_client.py             # API client
    ├── model_config.py               # Model configuration
    └── styles.css                    # Custom styling

```

## Connecting to MemMachine

Start MemMachine by either running the Python file or the Docker container. These example agents all use the REST API from memmachine's app.py, but you can also integrate using the MCP server.
These example agents all use the REST API from MemMachine's `app.py`, but you can also integrate using the MCP server for more advanced use cases.

## Available Agents

### When running it via Docker or python directly, it will default to using the profile_prompt.py. To use agents other than the default agent, make sure to change the prompt in the configuration file under the prompt/profile section.
If using Docker, make sure to use a local build image rather than the MemMachine Dockerhub image since that one uses the default profile_prompt.py. 

### 1. Default Agent (`example_server.py`)
- **Purpose**: General-purpose AI assistant for any chatbot or conversational interface
- **Port**: 8000 (configurable via `EXAMPLE_SERVER_PORT`)
- **Features**: 
  - Basic memory storage and retrieval
  - Conversation context management
  - User profile integration
- **Use Case**: General conversations, information management, and as a template for custom agents

### 2. CRM Agent (`crm/`)
- **Purpose**: Customer Relationship Management
- **Port**: 8000 (configurable via `CRM_PORT`)
- **Features**: 
  - Customer data management
  - Sales pipeline tracking
  - Slack integration for real-time communication
  - CRM-specific query construction
- **Use Case**: Sales teams, customer support, relationship management

### 3. Financial Analyst Agent (`financial_analyst/`)
- **Purpose**: Financial analysis and reporting
- **Port**: 8000 (configurable via `FINANCIAL_PORT`)
- **Features**:
  - Financial data analysis
  - Investment insights
  - Market trend analysis
  - Financial reporting
- **Use Case**: Financial advisors, investment teams, accounting departments

### 4. Health Assistant Agent (`health_assistant/`)
- **Purpose**: Health tracking, wellness guidance, and medical information assistance
- **Port**: 8000 (configurable via `HEALTH_PORT`)
- **Features**:
  - Health and wellness advice and recommendations
  - Tracking and analyzing health trends over time
  - Recording and managing medical history
  - Symptom tracking and health monitoring
  - Medication and appointment reminders
- **Use Case**: Health chatbots, patient care systems, wellness applications, and healthcare assistants

### 5. Health Assistant Agent (`writing_assistant/`)
- **Purpose**: AI-powered Writing assistant
- **Port**: 8000 (configurable via `WRITING_ASSISTANT_PORT`)
- **Features**:
  - Analyzes your writing samples to extract detailed style characteristics
  - Separate style profiles for different content types (email, blog, LinkedIn, etc.)
  - Generates new content that matches your established writing patterns
  - Use /submit command to easily submit writing samples
- **Use Case**: Technical writers, content creators, professionals looking to maintain a consistent writing style.

### 6. Streamlit Frontend (`frontend/`)
- **Purpose**: Web-based testing interface and demonstration platform for all agents
- **Port**: 8502 (configurable via Streamlit default)
- **Features**:
  - Interactive web UI for testing and demonstrating agents
  - Memory management interface with search and filtering
  - Real-time conversation testing with multiple models
  - Model selection and configuration across providers
  - Persona-based testing and user simulation
  - Response analysis and comparison tools
- **Use Case**: Development, testing, demonstration, and evaluation of agent capabilities

## Quick Start

### Prerequisites
- Python 3.12+
- FastAPI and Uvicorn
- Requests library
- Streamlit (for frontend)
- MemMachine backend running
- Environment variables configured
- OpenAI API key (or other LLM provider API key)

### Running an Agent

1. **Set up environment variables**:
   ```bash
   export MEMORY_BACKEND_URL="http://localhost:8080"
   export OPENAI_API_KEY="your-openai-api-key"
   export LOG_LEVEL="INFO"
   ```

2. **Run a specific agent**:
   ```bash
   # Default agent
   python example_server.py
   
   # CRM agent
   cd crm
   python crm_server.py
   
   # Financial analyst agent
   cd financial_analyst
   python financial_server.py

   # Health assistant agent
   cd health_assistant
   python health_server.py

   # Writing assistant agent
   cd writing_assistant
   python writing_assistant_server.py
   
   # Streamlit frontend (in separate terminal)
   cd frontend
   streamlit run app.py
   ```

3. **Access the services**:
   - Default Agent API: `http://localhost:8000`
   - CRM Agent API: `http://localhost:8000` (when running CRM server)
   - Financial Agent API: `http://localhost:8000` (when running Financial server)
   - Health Agent API: `http://localhost:8000` (when running Health server)
   - Writing Agent API: `http://localhost:8000` (when running Writing Assistant server)
   - Streamlit Frontend: `http://localhost:8502` (when running Streamlit app)
   - API Documentation: `http://localhost:8000/docs` (FastAPI auto-generated docs)

## Using the Streamlit Frontend for Testing

The Streamlit frontend provides an interactive web interface for testing all agents and their memory capabilities.

### Starting the Frontend

1. **Prerequisites**:
   - MemMachine backend running (see main README)
   - At least one agent server running (CRM, Financial, or Default)
   - Required environment variables set

2. **Run the frontend**:
   ```bash
   cd agents/frontend
   streamlit run app.py
   ```

3. **Access the interface**:
   - Open your browser to `http://localhost:8502`

### Frontend Features

#### Model Configuration
- **Model Selection**: Choose from various LLM providers (OpenAI, Anthropic, DeepSeek, Meta, Mistral)
- **API Key Management**: Configure API keys for different providers
- **Model Parameters**: Adjust temperature, max tokens, and other settings

#### Memory Testing
- **Persona Management**: Create and manage different user personas
- **Memory Storage**: Test memory storage and retrieval
- **Context Search**: Search through stored memories
- **Profile Management**: View and manage user profiles

#### Agent Testing
- **Real-time Chat**: Test conversations with different agents
- **Memory Integration**: See how agents use stored memories
- **Response Analysis**: Compare responses with and without memory context
- **Rationale Display**: View how personas influence responses

### Testing Workflow

1. **Start Services**:
   ```bash
   # Terminal 1: Start MemMachine backend
   cd memmachine/src
   python -m server.app
   
   # Terminal 2: Start an agent (e.g., CRM)
   cd agents/crm
   python crm_server.py
   
   # Terminal 3: Start the frontend
   cd examples/frontend
   streamlit run app.py
   ```

2. **Configure the Frontend**:
   - Set the CRM Server URL (default: `http://localhost:8000`)
   - Select your preferred model and provider
   - Enter your API key

3. **Test Memory Operations**:
   - Create a new persona or use existing ones
   - Send messages to test memory storage
   - Use search functionality to retrieve memories
   - Test different conversation patterns

4. **Analyze Results**:
   - View memory storage logs
   - Compare responses with/without memory context
   - Check persona influence on responses

### Environment Variables for Frontend

```bash
# Required for frontend functionality
CRM_SERVER_URL=http://localhost:8000
MODEL_API_KEY=your-openai-api-key
OPENAI_API_KEY=your-openai-api-key

# Optional: For other LLM providers on AWS Bedrock
ANTHROPIC_API_KEY=your-anthropic-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### Troubleshooting Frontend Issues

#### Common Issues:
1. **Connection Refused**: Ensure the agent server is running
2. **API Key Errors**: Verify your API keys are correct
3. **Memory Not Storing**: Check MemMachine backend is running
4. **Model Not Responding**: Verify model selection and API key

#### Debug Mode:
```bash
# Run with debug logging
LOG_LEVEL=DEBUG streamlit run app.py
```

### Frontend Architecture

The frontend consists of:
- **app.py**: Main Streamlit application
- **llm.py**: LLM integration and chat functionality
- **gateway_client.py**: API client for agent communication
- **model_config.py**: Model configuration and provider mapping
- **styles.css**: Custom styling for the interface

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_BACKEND_URL` | URL of the MemMachine backend service | `http://localhost:8080` |
| `OPENAI_API_KEY` | OpenAI API key for LLM access | Required |
| `EXAMPLE_SERVER_PORT` | Port for example server | `8000` |
| `CRM_PORT` | Port for CRM server | `8000` |
| `FINANCIAL_PORT` | Port for financial analyst server | `8000` |
| `HEALTH_PORT` | Port for health assistant server | `8000` |
| `WRITING_ASSISTANT_PORT` | Port for writing assistant server | `8000` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

### MemMachine Integration

All agents integrate with the MemMachine backend by:
1. Storing conversation episodes as memories
2. Retrieving relevant context for queries
3. Using profile information for personalized responses
4. Maintaining conversation history and context

## Query Constructor System

### Base Query Constructor
The `BaseQueryConstructor` class provides the foundation for all query constructors:

```python
class BaseQueryConstructor:
    def create_query(self, **kwargs) -> str:
        # Must be implemented by subclasses
        raise NotImplementedError
```

### Specialized Constructors

Each agent implements its own query constructor with domain-specific logic:

- **CRMQueryConstructor**: Optimized for customer relationship management
- **FinancialAnalystQueryConstructor**: Specialized for financial analysis
- **HealthAssistantQueryConstructor**: Specialized for health tracking and wellness guidance
- **DefaultQueryConstructor**: General-purpose query handling for any domain

## Slack Integration

The CRM agent includes Slack integration for real-time communication:

### Features
- Real-time message processing
- Webhook handling
- Interactive responses
- Thread management

### Setup
1. Configure Slack app with webhook URL
2. Set up environment variables for Slack
3. Deploy the slack_server.py endpoint

## Development

### Adding a New Agent

1. **Create agent directory**:
   ```bash
   mkdir examples/new_agent
   cd examples/new_agent
   ```

2. **Implement query constructor**:
   ```python
   from base_query_constructor import BaseQueryConstructor
   
   class NewAgentQueryConstructor(BaseQueryConstructor):
       def create_query(self, **kwargs) -> str:
           # Implement domain-specific logic
           pass
   ```

3. **Create FastAPI server**:
   ```python
   from fastapi import FastAPI
   from query_constructor import NewAgentQueryConstructor
   
   app = FastAPI(title="New Agent Server")
   constructor = NewAgentQueryConstructor()
   
   # Implement endpoints
   ```

4. **Add configuration**:
   - Environment variables
   - Port configuration
   - MemMachine backend integration

## Troubleshooting

### Common Issues

1. **MemMachine Backend Connection Error**:
   - Ensure the MemMachine backend is running on the correct port (default: 8080)
   - Check `MEMORY_BACKEND_URL` environment variable is set correctly
   - Verify network connectivity between agent and backend

2. **OpenAI API Errors**:
   - Verify `OPENAI_API_KEY` is set correctly and has sufficient credits
   - Check API key permissions and quotas

3. **Port Conflicts**:
   - Ensure only one agent runs on each port
   - Use different ports for multiple agents

4. **Import Errors**:
   - Check Python path configuration
   - Ensure all dependencies are installed

### Logging

All agents support configurable logging:

```bash
LOG_LEVEL=DEBUG  # For detailed debugging
LOG_LEVEL=INFO   # For normal operation
LOG_LEVEL=ERROR  # For error-only logging
```

## Contributing

When adding new agents or features:

1. **Follow existing architecture patterns**: Use the base query constructor and FastAPI server structure
2. **Implement proper error handling**: Include try-catch blocks and meaningful error messages
3. **Add comprehensive logging**: Use structured logging with appropriate log levels
4. **Include API documentation**: Document endpoints, request/response schemas, and examples
5. **Test with MemMachine backend integration**: Ensure memory storage and retrieval work correctly
6. **Add unit tests**: Create tests for your query constructor and API endpoints
7. **Update documentation**: Keep this README and any agent-specific documentation current

