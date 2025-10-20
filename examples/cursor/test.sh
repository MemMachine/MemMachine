curl -X POST "http://127.0.0.1:8080/v1/memories/search" \
-H "Content-Type: application/json" \
-d '{
  "session": {
    "group_id": "test_group",
    "agent_id": ["test_agent"],
    "session_id": "123",
    "user_id": ["test_user"]
  },
  "query": "simple test memory",
  "limit": 5
}'

