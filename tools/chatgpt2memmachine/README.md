# chatgpt2memmachine

Import memory from another source into MemMachine.
Supports ChatGPT and locomo.

To import from ChatGPT, from ChatGPT, export chat history. Then run migration.py to import.
e.g.
uv run python3 migration.py --chat_type=openai --chat_history=conversations.json

To import from locomo as a test, run test_migration.py in the project dir.
e.g.
uv run python3 test_migration.py
