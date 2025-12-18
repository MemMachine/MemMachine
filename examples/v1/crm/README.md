# CRM Agent with Slack Integration

This directory contains the CRM agent with full Slack integration capabilities, allowing real-time communication and memory management through Slack channels with AI-powered responses.

## Overview

The CRM Slack integration provides:
- **Real-time message processing** from Slack channels
- **Historical message ingestion** with user-controlled limits
- **Memory storage and retrieval** through Slack commands
- **AI-powered responses** using OpenAI integration
- **Smart deduplication** to prevent processing duplicate messages
- **Interactive setup** with user control over historical ingestion
- **Thread management** and conversation tracking
- **Webhook-based event handling** for real-time updates

## Architecture

```
crm/
├── crm_server.py          # Main CRM FastAPI server with memory backend integration
├── query_constructor.py   # CRM-specific query constructor
├── slack_server.py        # Slack integration server with historical ingestion
├── slack_service.py       # Slack service utilities and API wrapper
└── README.md             # This comprehensive guide
```

## System Flow

```
Slack Messages → Slack Server → CRM Server → Memory Backend → Database
                     ↓              ↓              ↓
                User Input    Deduplication   Episodic + Profile Memory
                     ↓              ↓              ↓
                *Q Commands   AI Search      OpenAI Responses
```

## Prerequisites

1. **Python Environment** (3.8+)
2. **PostgreSQL Database** 
3. **Slack Bot Token**
4. **OpenAI API Key** (for AI responses)
5. **Memory Backend** (MemMachine system)
6. **ngrok** for local development (free tier available)

## Step 1: Environment Configuration

Create or update `.env` file in the project root with these variables:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=your_database_name

# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_SIGNING_SECRET=your-slack-signing-secret
CRM_CHANNEL_ID=your-channel-id  # Optional: limit to specific channel

# Memory Backend
MEMORY_BACKEND_URL=http://localhost:8080

# CRM Server
CRM_SERVER_URL=http://localhost:8000
CRM_PORT=8000

# Slack Server
SLACK_PORT=8001

# OpenAI (for AI responses)
OPENAI_API_KEY=your-openai-api-key

# Historical Ingestion Control
SLACK_ENABLE_HISTORICAL=true  # Set to false to disable historical ingestion

# Logging
LOG_LEVEL=INFO
```

## Step 2: Database Setup

1. **Start PostgreSQL** and create a database
2. **Run the MemMachine schema** to set up the required tables
3. **The system automatically handles** slack message deduplication using metadata fields

## Step 3: Install Dependencies

```bash
# Install required Python packages
pip install fastapi uvicorn httpx asyncpg slack-sdk openai python-dotenv numpy
```

## Step 4: Slack Bot Setup

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App"
3. Choose "From scratch"
4. Enter app name (e.g., "CRM Memory Bot")
5. Select your workspace

### 2. Configure Bot Permissions

Navigate to **OAuth & Permissions** and ensure you have these scopes:

#### Bot Token Scopes 
```
channels:history           # View messages in public channels
channels:read              # View basic information about public channels
chat:write                 # Send messages as @CRM Bot
groups:history             # View messages in private channels
groups:read                # View basic information about private channels
users:read                 # View people in a workspace
```

#### Additional Scopes (Optional)
If you want to expand functionality, consider adding:
```
app_mentions:read          # View messages that directly mention @CRM Bot
commands                   # Add slash commands
files:read                 # Read files shared in channels
reactions:read             # Read reactions
reactions:write            # Add/remove reactions
team:read                  # View workspace name
users:read.email           # View email addresses
```

### 3. Install App to Workspace

1. Go to **OAuth & Permissions**
2. Click "Install to Workspace"
3. Review permissions and click "Allow"
4. Copy the **Bot User OAuth Token** (starts with `xoxb-`)

### 4. Get Signing Secret

1. Go to **Basic Information**
2. Copy the **Signing Secret** from the "App Credentials" section

### 5. Configure Event Subscriptions

1. Go to **Event Subscriptions**
2. Enable Events: **On**
3. Request URL: `https://your-ngrok-url.ngrok.io/slack/events`
4. Subscribe to Bot Events:
   ```
   message.channels     # Messages in public channels
   message.groups       # Messages in private channels
   ```

## Step 5: Install and Configure ngrok

### Install ngrok
```bash
# macOS (using Homebrew)
brew install ngrok

# Or download from https://ngrok.com/download
```

### Start ngrok
```bash
# Start ngrok on port 8001 (or your chosen port)
ngrok http 8001

# Note the HTTPS URL (e.g., https://abc123.ngrok.io)
```

### Update Slack App Configuration
1. Copy the ngrok HTTPS URL
2. Update your Slack app's Event Subscriptions URL
3. Update any slash command URLs

## Step 6: Run the System

### Option A: Run Everything (Recommended)

```bash
# Terminal 1: Start Memory Backend
cd memmachine/src
python -m server.app

# Terminal 2: Start CRM Server  
cd agents
python -m crm.crm_server

# Terminal 3: Start Slack Server (with interactive setup)
cd agents/crm
python slack_server.py
```

### Option B: Run Individual Components

```bash
# Memory Backend (Terminal 1)
cd memmachine/src
python -m server.app

# CRM Server (Terminal 2)  
cd agents
python -m crm.crm_server

# Slack Server (Terminal 3)
cd agents/crm
python slack_server.py
```

## Step 7: User Experience

When you start the Slack server, you'll see an interactive setup:

```
============================================================
SLACK HISTORICAL MESSAGE INGESTION
============================================================
Enter the number of historical messages to ingest per channel.
Press Enter for default (5 messages), or type a number.
============================================================
Number of messages to ingest per channel [5]: 
```

**Options:**
- Press **Enter** → Ingest 5 messages per channel (default)
- Type **50** → Ingest 50 messages per channel
- Type **0** → Skip historical ingestion
- Type **1000** → Ingest up to 1000 messages per channel

**After ingestion, you'll see:**
```
Historical ingestion complete!
Total processed: 5 messages
Skipped (duplicates): 0 messages
```

## Step 8: Verify Everything Works

1. **Check logs** for successful startup
2. **Send a test message** in your Slack channel
3. **Use `*Q` command** to test memory search:
   - Type `*Q what did I say about project X?` in Slack
   - The bot will search memory and respond with AI-generated answers
4. **Check database** for stored messages in `history` and profile tables
5. **Test deduplication** by running the server again - it should skip already processed messages

## API Endpoints

### CRM Server
- `POST /memory` - Store messages with deduplication
- `GET /memory` - Search memory and get AI-formatted responses
- `POST /memory/store-and-search` - Store and search in one operation

### Slack Server
- `POST /slack/events` - Handle Slack events and interactions

## Features

- **Historical Ingestion** - Fetches all past messages on startup with user control
- **Real-time Processing** - Processes new messages as they arrive
- **Smart Deduplication** - Prevents processing the same message twice using database lookups
- **Profile Memory Integration** - Stores messages in both episodic and profile memory
- **AI-Powered Search** - Use `*Q` commands to search memory with AI responses
- **User Control** - Choose how many messages to ingest interactively
- **Channel Filtering** - Option to limit to specific channels
- **Comprehensive Logging** - Detailed logging for troubleshooting
- **Error Handling** - Graceful failure with detailed error messages
- **Skip Counter** - Shows how many duplicate messages were skipped
- **OpenAI Integration** - AI-powered responses using GPT-4o-mini
- **Thread Support** - Responds in threads for better conversation flow

## Troubleshooting

### Common Issues:

1. **Database Connection Error**
   - Check PostgreSQL is running
   - Verify database credentials in `.env`

2. **Slack API Error**
   - Verify bot token and signing secret
   - Check bot has required scopes

3. **Memory Backend Error**
   - Ensure memory backend is running on port 8080
   - Check `MEMORY_BACKEND_URL` in `.env`

4. **OpenAI API Error**
   - Verify `OPENAI_API_KEY` is set correctly
   - Check API key has sufficient credits

5. **Port Conflicts**
   - CRM Server: port 8000
   - Slack Server: port 8001
   - Memory Backend: port 8080

### Logs to Check:

- **Slack Server**: Look for `[SLACK]` prefixed messages
- **CRM Server**: Look for `[CRM]` prefixed messages  
- **Memory Backend**: Check for database connection logs
- **Profile Memory**: Look for `[PROFILE]` and `[PROFILE_UPDATE]` messages
- **OpenAI**: Look for `[OPENAI]` prefixed messages

## How It Works

The system automatically:
1. **Fetches historical messages** when server starts (with user control)
2. **Processes new messages** in real-time with deduplication
3. **Stores everything** in both episodic and profile memory
4. **Prevents duplicates** using efficient database lookups with skip counting
5. **Provides AI search** via `*Q` commands with OpenAI integration
6. **Tracks processing stats** showing processed vs skipped messages
7. **Responds in threads** for better conversation organization