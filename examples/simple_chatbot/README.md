# Simple Chatbot - MemMachine Integration Example

A complete Streamlit-based chatbot that demonstrates MemMachine's persistent memory capabilities with support for multiple LLM providers.

## Overview

This example provides a simple chatbot interface that showcases:
- **Persistent Memory**: Conversations are stored and retrieved using MemMachine's memory system
- **Multi-Provider LLM Support**: Works with OpenAI, Anthropic, Google Gemini etc.
- **Persona Management**: Support for multiple user personas with isolated memory profiles
- **Side-by-Side Comparison**: Compare MemMachine-enhanced responses vs control persona (no memory)
- **Memory Import**: Import conversation history from external sources (ChatGPT, etc.)
- **Session Management**: Create, rename, and manage multiple conversation sessions

## Features

### Memory Integration
- Automatic memory storage for all conversations
- Context-aware responses using retrieved memories
- Profile-based personalization

### Multi-Session Support
- Create multiple independent conversation sessions
- Rename and delete sessions
- Switch between sessions seamlessly
- The sessions are not persistent for now, meaning they will disappear when you refresh/restart the app. But the memories in the sessions is persistent in MemMachine.

### Persona Management
- Support for multiple user personas
- Custom persona names
- Isolated memory profiles per persona

### Comparison Mode
- Side-by-side comparison of MemMachine vs Control persona
- Visual distinction between memory-enhanced and baseline responses
- Toggle to enable/disable comparison

### Memory Import
- Import conversation history from external sources
- Support for text files, JSON, and markdown formats
- Preview before ingesting into MemMachine

### Multi-Provider LLM Support
- **OpenAI**: GPT-4.1 Mini, GPT-5, GPT-5 Mini, GPT-5 Nano
- **Anthropic (Through AWS Bedrock)**: Claude Haiku 4.5, Claude Sonnet 4.5, Claude Opus 4
- **Google Gemini**: Gemini 3 Pro (Preview), Gemini 2.5 Pro

## Architecture

```
simple_chatbot/
├── app.py              # Main Streamlit application
├── llm.py              # LLM provider integration (OpenAI, Anthropic, Google, etc.)
├── gateway_client.py   # MemMachine API client
├── model_config.py     # Model configuration and provider mappings
├── requirements.txt    # Python dependencies
├── styles.css          # Custom styling (optional)
├── assets/             # Logo and image assets
│   ├── memmachine_logo.png
│   └── memverge_logo.png
└── README.md           # This file
```

## Prerequisites

1. **Python 3.12+**
2. **MemMachine Backend Running** (see main README)
3. **LLM API Keys** (at least one):
   - OpenAI API key (for OpenAI models)
   - AWS credentials (for Bedrock models)
   - Google API key (for Gemini models)

## Installation

1. **Install dependencies**:
   ```bash
   cd examples/simple_chatbot
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   
   You can set environment variables either via:
   - **Environment variables** (export/set)
   - **`.env` file** in the `examples/simple_chatbot/` directory (automatically loaded via `python-dotenv`)
   
   ```bash
   # MemMachine backend URL
   export MEMORY_SERVER_URL="http://localhost:8080"
   
   # MemMachine organization ID (required for v2 API)
   export ORG_ID="default-org"        # Your organization ID
   # Note: Project ID is automatically set per user as "project_{user_id}"
   
   # LLM Provider API Keys (choose based on which models you want to use)
   export OPENAI_API_KEY="your-openai-api-key"  # Required for OpenAI models
   export AWS_ACCESS_KEY_ID="your-aws-key"      # Required for Bedrock models
   export AWS_SECRET_ACCESS_KEY="your-aws-secret"
   export AWS_DEFAULT_REGION="us-east-1"        # Your AWS region (default: us-east-1)
   export GOOGLE_API_KEY="your-google-api-key"  # Required for Gemini models
   
   # Optional: Provisioned throughput ARNs for Bedrock (if using)
   export BEDROCK_HAIKU_4_5_ARN="arn:aws:bedrock:..."
   export BEDROCK_SONNET_4_5_ARN="arn:aws:bedrock:..."
   export BEDROCK_OPUS_4_ARN="arn:aws:bedrock:..."
   ```
   
   **Example `.env` file** (create `examples/simple_chatbot/.env`):
   ```bash
   MEMORY_SERVER_URL=http://localhost:8080
   ORG_ID=default-org
   OPENAI_API_KEY=sk-...
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=...
   AWS_DEFAULT_REGION=us-east-1
   GOOGLE_API_KEY=...
   ```

3. **Start MemMachine backend** (if not already running):
   ```bash
   # See main README for instructions on starting MemMachine
   ```

## Running the Application

### Local Development

```bash
cd examples/simple_chatbot
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage Guide

### Basic Chat

1. **Select a Model**: Choose your preferred LLM from the sidebar
2. **Choose Persona**: Select or enter a persona name
3. **Enable MemMachine**: Toggle "Enable MemMachine" to use persistent memory
4. **Start Chatting**: Type messages and receive context-aware responses

### Session Management

- **Create Session**: Use "Create session" form in sidebar
- **Switch Sessions**: Click on session name in sidebar
- **Rename Session**: Click ⋯ menu next to session name
- **Delete Session**: Use delete option in session menu

### Comparison Mode

1. Enable "Enable MemMachine" checkbox
2. Enable "🔄 Compare with control persona" checkbox
3. Send a message to see side-by-side comparison:
   - **Left**: MemMachine-enhanced response (with memory)
   - **Right**: Control persona response (no memory)

### Memory Import

1. Expand "📋 Load Previous Memories" section
2. Paste conversation history or upload a file
3. Click "👁️ Preview" to review
4. Click "💉 Ingest into MemMachine" to import

### Profile Management

- **Delete Profile**: Removes all memories for the current persona
- **Clear Chat**: Clears current conversation history (keeps memories)

## Configuration

### Model Configuration

Edit `model_config.py` to:
- Add new models
- Change provider mappings
- Update display names
- Configure inference profile ARNs for provisioned throughput

### Customization

- **Styling**: Add CSS to `styles.css`
- **Prompt**: Modify the prompt template in `gateway_client.py`
- **UI**: Customize Streamlit components in `app.py`

## Environment Variables

The application loads environment variables from:
1. System environment variables
2. `.env` file in `examples/simple_chatbot/` directory (via `python-dotenv`)

| Variable | Description | Required | Default | Used By |
|----------|-------------|----------|---------|---------|
| `MEMORY_SERVER_URL` | MemMachine backend URL | Yes | `http://localhost:8080` | `gateway_client.py` |
| `ORG_ID` | Organization ID for v2 API | No | `default-org` | `gateway_client.py` |
| `OPENAI_API_KEY` | OpenAI API key | Yes* | - | `llm.py` |
| `AWS_ACCESS_KEY_ID` | AWS access key for Bedrock | Yes* | - | `llm.py` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for Bedrock | Yes* | - | `llm.py` |
| `AWS_DEFAULT_REGION` | AWS region for Bedrock | No | `us-east-1` | `llm.py` |
| `GOOGLE_API_KEY` | Google API key for Gemini | Yes* | - | `llm.py` |
| `BEDROCK_HAIKU_4_5_ARN` | Provisioned throughput ARN for Haiku 4.5 | No | - | `model_config.py` |
| `BEDROCK_SONNET_4_5_ARN` | Provisioned throughput ARN for Sonnet 4.5 | No | - | `model_config.py` |
| `BEDROCK_OPUS_4_ARN` | Provisioned throughput ARN for Opus 4 | No | - | `model_config.py` |

*At least one LLM provider API key is required (OpenAI, AWS Bedrock, or Google)



## Troubleshooting

### Connection Issues

**Problem**: "Failed to connect to MemMachine backend"
- **Solution**: Verify `MEMORY_SERVER_URL` is correct and backend is running
- Check firewall/network settings

### Authentication Errors

**Problem**: "Invalid token" or authentication failures
- **OpenAI**: Verify `OPENAI_API_KEY` is correct and has credits
- **AWS Bedrock**: Check `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct and have Bedrock access permissions
- **Google**: Verify `GOOGLE_API_KEY` is valid and has Gemini API access

### Memory Not Working

**Problem**: Responses don't seem personalized
- Verify "Enable MemMachine" checkbox is enabled
- Check that memories are being stored (use memory import/preview)
- Verify backend is processing requests correctly
- Check that `MEMORY_SERVER_URL` is correct and backend is accessible
- Verify `ORG_ID` is set (defaults to "default-org" if not set)

### Model Not Available

**Problem**: Selected model not working
- Verify API keys for that provider are set
- Check model ID spelling in `model_config.py`
- For Bedrock models: verify model is available in your AWS region

## Architecture Details

### Flow Diagram

```
User Input → Gateway Client → MemMachine Backend
                ↓
         Memory Search & Storage
                ↓
         Context-Enhanced Query
                ↓
         LLM Provider (OpenAI/Bedrock/Gemini)
                ↓
         Personalized Response
```

### Key Components

1. **app.py**: Main Streamlit UI and session management
2. **llm.py**: Handles communication with different LLM providers
3. **gateway_client.py**: MemMachine API integration
4. **model_config.py**: Model and provider configuration

### Memory Integration

This example uses MemMachine's **v2 API**:
- User messages are automatically ingested via `/api/v2/memories` endpoint
- Context is retrieved via `/api/v2/memories/search` endpoint
- Episodic and semantic memory types supported
- Uses `org_id` for organization scoping (set via `ORG_ID` env var)
- Project ID is dynamically generated per user, not set via environment variable

## Advanced Features

### Provisioned Throughput (AWS Bedrock)

For models with provisioned throughput, set the inference profile ARN via environment variables:
```bash
export BEDROCK_HAIKU_4_5_ARN="arn:aws:bedrock:us-east-1:..."
export BEDROCK_SONNET_4_5_ARN="arn:aws:bedrock:us-east-1:..."
export BEDROCK_OPUS_4_ARN="arn:aws:bedrock:us-east-1:..."
```

Or in your `.env` file:
```bash
BEDROCK_HAIKU_4_5_ARN=arn:aws:bedrock:us-east-1:...
BEDROCK_SONNET_4_5_ARN=arn:aws:bedrock:us-east-1:...
BEDROCK_OPUS_4_ARN=arn:aws:bedrock:us-east-1:...
```

The app will automatically use the ARN instead of the model ID when available.

### Custom Personas

You can create custom personas by:
1. Entering a custom name in the "Or enter your name" field
2. Each persona maintains isolated memory profiles 
3. Switch between personas to see different memory contexts
4. Each persona's memories are completely isolated from others

### Memory Import Formats

Supported formats for memory import:
- Plain text conversations
- Markdown files
- JSON files
- ChatGPT export formats

## Contributing

When improving this example:
1. Maintain backward compatibility with existing configurations
2. Add error handling for new features
3. Update this README with new features
4. Test with multiple LLM providers

## License

See main project LICENSE file.

## Support

For issues or questions:
- GitHub Issues: [MemMachine Repository](https://github.com/MemMachine/MemMachine)
- Discord: [MemMachine Community](https://discord.gg/usydANvKqD)
- Documentation: [MemMachine Docs](https://memmachine.ai/docs)

