## MemMachine Dify Integration (for contributors)

This folder contains the MemMachine Dify tool plugin implementation (provider + tools) used to integrate MemMachine APIs into Dify.

### What’s in here

- `manifest.yaml`: Dify plugin manifest.
- `provider/`: provider definition + credential validation.
- `tools/`: tool implementations (e.g. add/search memory).
- `templates/`: Jinja templates used to generate the provider YAML/Python during build.
- `build_plugin.py`: build + package script for contributors (not included in the packaged plugin).

### Credentials and Base URL behavior

- The plugin always requires `memmachine_api_key`.
- Optionally, the provider can expose an extra credential `memmachine_base_url`.
	- When present and non-empty, tools will use it to override the default API base URL.
	- When absent/empty, tools fall back to the default base URL (`https://api.memmachine.ai/v2`).

Whether `memmachine_base_url` exists in the provider schema is controlled at build time by a flag (see below).

### Packaging (how contributors should build)

Prerequisites:

- Dify plugin CLI installed and available as `dify` in `PATH`.
- Python available to run the build script.
- Build dependency: `jinja2` (build-only; not a runtime dependency of the plugin).

Build + package:

- Default build (no base-url credential exposed):
	- `python build_plugin.py`
- Build with optional base-url credential (`memmachine_base_url`) exposed:
	- `python build_plugin.py --enable-base-url-credential`

The build script will:

1. Verify `dify` CLI exists (otherwise it exits with an error).
2. Render provider files from templates into `provider/`.
3. Run `dify plugin package <this_plugin_dir>`.

### Repo hygiene

- Generated provider files are ignored by git (source of truth is under `templates/`).
- `templates/` and `build_plugin.py` are excluded from the packaged plugin via `.difyignore`.

