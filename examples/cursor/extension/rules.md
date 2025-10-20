## MemMachine Integration Rules

### Rule 1: Query Memory Before Processing
- **Requirement**: Always query MemMachine memory before processing any user message to check for relevant context, previous conversations, user preferences, or important information that should inform the response.
- **Purpose**: Ensures continuity and personalized interactions based on stored memories.
- **Implementation**: Use the MemMachine search function to look for relevant context before responding to user queries.

### Rule 2: Save Important Information
- **Requirement**: Always save important information to MemMachine after processing user messages, including user preferences, key decisions, project details, important conversations, user feedback, or any context that would be valuable for future interactions.
- **Purpose**: Helps maintain continuity across sessions and improves personalized responses.
- **Implementation**: Use the MemMachine add memory function to store relevant information after completing tasks or conversations.
