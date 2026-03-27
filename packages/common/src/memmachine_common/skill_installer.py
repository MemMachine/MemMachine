"""Provider-backed skill installation helpers shared by client and server."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
from typing import Literal, Protocol, cast

from .skill import Skill


class _AnthropicFilesAPI(Protocol):
    async def upload(self, *, file: object, betas: list[str]) -> object: ...


class _AnthropicBetaAPI(Protocol):
    files: _AnthropicFilesAPI


class _AnthropicClientProtocol(Protocol):
    beta: _AnthropicBetaAPI


class _OpenAIFilesAPI(Protocol):
    async def create(self, *, file: object, purpose: str) -> object: ...


class _OpenAIClientProtocol(Protocol):
    files: _OpenAIFilesAPI


ProviderName = Literal["anthropic", "openai"]

_ANTHROPIC_FILES_BETA = "files-api-2025-04-14"
_CACHE_FILE = ".memmachine_skill_cache.json"
_MAIN_SKILL_FILENAMES = {"skill.md", "retrieve_agent.md", "retrieve_skill.md"}


class _NamedBytesIO(io.BytesIO):
    """Bytes buffer carrying a stable filename for multipart uploads."""

    def __init__(self, data: bytes, *, name: str) -> None:
        super().__init__(data)
        self.name = name


def _require_anthropic_sdk() -> object:
    try:
        return importlib.import_module("anthropic")
    except ModuleNotFoundError as err:
        raise ImportError(
            "anthropic SDK not installed. Install it with: pip install anthropic"
        ) from err


def _require_openai_sdk() -> object:
    try:
        return importlib.import_module("openai")
    except ModuleNotFoundError as err:
        raise ImportError(
            "openai SDK not installed. Install it with: pip install openai"
        ) from err


def _new_anthropic_client() -> _AnthropicClientProtocol:
    anthropic_module = _require_anthropic_sdk()
    client_factory = getattr(anthropic_module, "AsyncAnthropic", None)
    if not callable(client_factory):
        raise TypeError(
            "anthropic SDK does not expose AsyncAnthropic(). Upgrade the anthropic package."
        )
    return cast(_AnthropicClientProtocol, client_factory())


def _new_openai_client() -> _OpenAIClientProtocol:
    openai_module = _require_openai_sdk()
    client_factory = getattr(openai_module, "AsyncOpenAI", None)
    if not callable(client_factory):
        raise TypeError(
            "openai SDK does not expose AsyncOpenAI(). Upgrade the openai package."
        )
    return cast(_OpenAIClientProtocol, client_factory())


def _resolve_skill_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Skill path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".md":
            raise ValueError("install_skill() only accepts markdown skill files")
        return [path]

    skill_files = [candidate for candidate in path.rglob("*.md") if candidate.is_file()]
    if not skill_files:
        raise ValueError(f"No markdown skill files found under: {path}")

    return sorted(
        skill_files,
        key=lambda candidate: (
            0 if candidate.name.lower() in _MAIN_SKILL_FILENAMES else 1,
            candidate.relative_to(path).as_posix(),
        ),
    )


def _normalize_skill_name(path: Path, *, skill_name: str | None) -> str:
    if skill_name is not None:
        normalized = skill_name.strip()
        if not normalized:
            raise ValueError("skill_name must not be blank when provided")
        return normalized
    return path.stem if path.is_file() else path.name


def _resolve_install_paths(
    path: str | Path,
    *,
    cache_path: str | Path | None,
) -> tuple[Path, Path]:
    skill_path = Path(path).expanduser().resolve()
    resolved_cache_path = (
        Path(cache_path).expanduser().resolve()
        if cache_path is not None
        else Path.cwd() / _CACHE_FILE
    )
    return skill_path, resolved_cache_path


def _content_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}

    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    cache: dict[str, dict[str, str]] = {}
    for provider, entries in raw.items():
        if not isinstance(provider, str) or not isinstance(entries, dict):
            continue
        normalized_entries = {
            key: value
            for key, value in entries.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if normalized_entries:
            cache[provider] = normalized_entries
    return cache


def _save_cache(cache_path: Path, cache: dict[str, dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _anthropic_upload_name(source_path: Path) -> str:
    # Anthropic document blocks support plain-text documents via file_id, but `.md`
    # filenames are currently documented as unsupported. Upload the markdown bytes as
    # plain text with a `.txt` filename while preserving the original content hash.
    return f"{source_path.stem}.txt"


async def _upload_anthropic(
    client: _AnthropicClientProtocol,
    *,
    filename: str,
    file_bytes: bytes,
) -> str:
    uploaded = await client.beta.files.upload(
        file=(filename, file_bytes, "text/plain"),
        betas=[_ANTHROPIC_FILES_BETA],
    )
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("Anthropic upload returned no file id")
    return file_id


async def _upload_openai(
    client: _OpenAIClientProtocol,
    *,
    filename: str,
    file_bytes: bytes,
) -> str:
    uploaded = await client.files.create(
        file=_NamedBytesIO(file_bytes, name=filename),
        purpose="user_data",
    )
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("OpenAI upload returned no file id")
    return file_id


async def install_skill(
    path: str | Path,
    provider: ProviderName,
    *,
    anthropic_client: _AnthropicClientProtocol | None = None,
    openai_client: _OpenAIClientProtocol | None = None,
    skill_name: str | None = None,
    cache_path: str | Path | None = None,
) -> Skill:
    """Upload one skill file or directory of skill files to a provider Files API."""
    skill_path, resolved_cache_path = _resolve_install_paths(
        path,
        cache_path=cache_path,
    )
    files = _resolve_skill_files(skill_path)
    resolved_skill_name = _normalize_skill_name(skill_path, skill_name=skill_name)
    cache = _load_cache(resolved_cache_path)
    provider_cache = cache.setdefault(provider, {})

    file_ids: list[str] = []
    content_hashes: list[str] = []
    cache_dirty = False

    if provider == "anthropic":
        if anthropic_client is None:
            anthropic_client = _new_anthropic_client()
        for file_path in files:
            file_bytes = file_path.read_bytes()
            digest = _content_hash(file_bytes)
            content_hashes.append(digest)
            cached_file_id = provider_cache.get(digest)
            if cached_file_id is None:
                cached_file_id = await _upload_anthropic(
                    anthropic_client,
                    filename=_anthropic_upload_name(file_path),
                    file_bytes=file_bytes,
                )
                provider_cache[digest] = cached_file_id
                cache_dirty = True
            file_ids.append(cached_file_id)
    elif provider == "openai":
        if openai_client is None:
            openai_client = _new_openai_client()
        for file_path in files:
            file_bytes = file_path.read_bytes()
            digest = _content_hash(file_bytes)
            content_hashes.append(digest)
            cached_file_id = provider_cache.get(digest)
            if cached_file_id is None:
                cached_file_id = await _upload_openai(
                    openai_client,
                    filename=file_path.name,
                    file_bytes=file_bytes,
                )
                provider_cache[digest] = cached_file_id
                cache_dirty = True
            file_ids.append(cached_file_id)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if cache_dirty:
        _save_cache(resolved_cache_path, cache)

    return Skill(
        provider=provider,
        skill_name=resolved_skill_name,
        file_ids=tuple(file_ids),
        content_hashes=tuple(content_hashes),
    )
