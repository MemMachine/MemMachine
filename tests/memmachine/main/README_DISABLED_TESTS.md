# Disabled Tests

## test_memmachine_integration.py (DISABLED)

**Status:** Temporarily disabled  
**Renamed to:** `test_memmachine_integration.py.disabled`  
**Reason:** Tests hang during semantic memory ingestion

### Why Disabled?

The integration tests in this file have the following issues:

1. **Long Wait Times** - Tests wait up to 120 seconds for background semantic ingestion
2. **External Dependencies** - Requires real LLM model (OpenAI API or Ollama)
3. **Flaky Behavior** - Background task may not start reliably
4. **Poor Feedback** - No logging during long waits (fixed but not tested)

### Tests Affected

- `test_memmachine_smoke_ingests_all_memories` - Waits 120s for semantic features
- `test_long_mem_eval_via_memmachine` - Waits 1200s (20 minutes!)
- `test_memmachine_list_set_ids_returns_details` - Generally fast but has dependencies

### To Re-enable

1. **Fix root causes:**
   ```bash
   # Set up LLM model
   export OPENAI_API_KEY="sk-..."
   # OR
   ollama pull mistral:7b
   export OLLAMA_HOST="http://localhost:11434/v1"
   ```

2. **Verify logging improvements work:**
   ```bash
   uv run pytest tests/memmachine/main/test_memmachine_integration.py.disabled \
     -m integration \
     -v \
     -s \
     --log-cli-level=INFO \
     -k "smoke"
   ```

3. **If tests pass, rename back:**
   ```bash
   mv tests/memmachine/main/test_memmachine_integration.py.disabled \
      tests/memmachine/main/test_memmachine_integration.py
   ```

### Documentation

See `FIX_HANGING_TEST.md` in the repository root for:
- Detailed root cause analysis
- Debugging steps
- Logging improvements made
- Common failure scenarios

### Alternative

Consider splitting into:
- **Unit tests** - Mock semantic service, test logic only
- **Integration tests** - Keep but mark as `@pytest.mark.slow` and `@pytest.mark.integration`
- **E2E tests** - Separate test suite with longer timeouts

Last updated: February 11, 2026
