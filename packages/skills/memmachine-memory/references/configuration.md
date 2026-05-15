# MemMachine Memory Configuration

Configure MemMachine connection parameters in the first fenced `json` block in
this file. The skill reads values from this file when they are present and uses
them as explicit `mem-cli` command arguments.

```json
{
  "MEMORY_BACKEND_URL": "",
  "MEMMACHINE_API_KEY": "",
  "MEMMACHINE_ORG_ID": "",
  "MEMMACHINE_PROJECT_ID": ""
}
```

## Usage Notes

- Leave a value empty to omit it and fall back to the environment or an explicit
  command argument.
- Keep secrets out of committed files when possible. Prefer local-only edits or
  environment variables for `MEMMACHINE_API_KEY` in shared repositories.
- Command arguments provided by the user or caller override values in this file.
- Values in this file override ambient environment variables.
