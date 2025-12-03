# safe-to-test Label Usage Guide

## Quick Overview

The `safe-to-test` label is used to trigger full integration tests for fork PRs (requires secrets).

## For Maintainers

### When to Use

When a fork PR needs to run full integration tests that require secrets (e.g., `OPENAI_API_KEY`).

### Steps

1. **Review PR code** - Confirm the code is safe and contains no malicious behavior
2. **Add label** - On the PR page, click "Labels" on the right → search for `safe-to-test` → add it
3. **Wait for tests** - Adding the label automatically triggers the `pull_request_target` event and runs full tests

### ⚠️ Security Notes

- **Only add the label after confirming the code is safe**
- Once the label is added, the PR code will have access to secrets
- If suspicious code is found, **remove the label immediately**

## For Contributors

If your fork PR needs full tests:

1. Submit PR (basic tests will run automatically but fail due to missing secrets)
2. Wait for maintainer code review
3. After maintainer adds `safe-to-test` label, full tests will run automatically
4. After tests pass, the failed `pull_request` check will be automatically updated to success, and the PR can be merged

## How It Works

- **`pull_request` event**: Automatically triggered for fork PRs, but cannot access secrets, tests will fail
- **`pull_request_target` event**: Triggered after adding `safe-to-test` label, can access secrets, runs full tests
- **Auto-update**: After `pull_request_target` succeeds, it automatically updates the `pull_request` check status to success

## Related Documentation

- Detailed guide: [README_FORK_PR_TESTING.md](./README_FORK_PR_TESTING.md)
- Quick start: [QUICK_START_LABELS.md](./QUICK_START_LABELS.md)
