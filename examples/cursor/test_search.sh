curl -X POST "http://127.0.0.1:8080/v1/memories/search" \                      
-H "Content-Type: application/json" \
-d '{
  "session": {
    "group_id": "test_group",
    "agent_id": ["test_agent"],
    "user_id": ["test_user"],
    "session_id": "session_123"
  },
  "query": "simple test memory",
  "filter": {},
  "limit": 5
}'
