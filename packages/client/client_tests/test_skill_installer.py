"""Unit tests for provider-backed skill installation."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from memmachine_client import Skill, install_skill


class _FakeAnthropicFiles:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, object]] = []

    async def upload(self, *, file, betas):
        self.upload_calls.append({"file": file, "betas": betas})
        return SimpleNamespace(id=f"anth-file-{len(self.upload_calls)}")


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.beta = SimpleNamespace(files=_FakeAnthropicFiles())


class _FakeOpenAIFiles:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []

    async def create(self, *, file, purpose):
        file.seek(0)
        self.create_calls.append(
            {
                "filename": getattr(file, "name", None),
                "content": file.read(),
                "purpose": purpose,
            }
        )
        return SimpleNamespace(id=f"openai-file-{len(self.create_calls)}")


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.files = _FakeOpenAIFiles()


@pytest.mark.asyncio
async def test_install_skill_returns_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")
    client = _FakeOpenAIClient()

    skill = await install_skill(skill_file, "openai", openai_client=client)

    assert isinstance(skill, Skill)
    assert skill.provider == "openai"
    assert skill.skill_name == "retrieve_skill"
    assert skill.file_ids == ("openai-file-1",)
    assert len(skill.content_hashes) == 1
    assert (tmp_path / ".memmachine_skill_cache.json").exists()


@pytest.mark.asyncio
async def test_install_skill_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "retrieve_skill.md").write_text("# Top Level", encoding="utf-8")
    (skill_dir / "coq.md").write_text("# CoQ", encoding="utf-8")
    client = _FakeOpenAIClient()

    skill = await install_skill(skill_dir, "openai", openai_client=client)

    assert skill.skill_name == "skills"
    assert skill.file_ids == ("openai-file-1", "openai-file-2")
    assert [call["filename"] for call in client.files.create_calls] == [
        "retrieve_skill.md",
        "coq.md",
    ]


@pytest.mark.asyncio
async def test_anthropic_uses_text_plain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")
    client = _FakeAnthropicClient()

    await install_skill(skill_file, "anthropic", anthropic_client=client)

    upload_call = client.beta.files.upload_calls[0]
    filename, _payload, mime_type = upload_call["file"]
    assert filename == "retrieve_skill.txt"
    assert mime_type == "text/plain"
    assert upload_call["betas"] == ["files-api-2025-04-14"]


@pytest.mark.asyncio
async def test_openai_uses_user_data_purpose(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")
    client = _FakeOpenAIClient()

    await install_skill(skill_file, "openai", openai_client=client)

    assert client.files.create_calls[0]["purpose"] == "user_data"


@pytest.mark.asyncio
async def test_cache_hit_no_reupload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")
    client = _FakeOpenAIClient()

    first_skill = await install_skill(skill_file, "openai", openai_client=client)
    second_skill = await install_skill(skill_file, "openai", openai_client=client)

    assert len(client.files.create_calls) == 1
    assert first_skill.file_ids == second_skill.file_ids


@pytest.mark.asyncio
async def test_cache_miss_reuploads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")
    client = _FakeOpenAIClient()

    first_skill = await install_skill(skill_file, "openai", openai_client=client)
    skill_file.write_text("# Retrieve Updated", encoding="utf-8")
    second_skill = await install_skill(skill_file, "openai", openai_client=client)

    assert len(client.files.create_calls) == 2
    assert first_skill.file_ids != second_skill.file_ids


@pytest.mark.asyncio
async def test_anthropic_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")

    skill = await install_skill(
        skill_file,
        "anthropic",
        anthropic_client=_FakeAnthropicClient(),
    )

    assert skill.provider == "anthropic"


@pytest.mark.asyncio
async def test_openai_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")

    skill = await install_skill(skill_file, "openai", openai_client=_FakeOpenAIClient())

    assert skill.provider == "openai"


@pytest.mark.asyncio
async def test_missing_anthropic_import_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")
    original_import_module = importlib.import_module

    def _patched_import_module(name: str, package: str | None = None):
        if name == "anthropic":
            raise ModuleNotFoundError("No module named 'anthropic'")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _patched_import_module)

    with pytest.raises(ImportError, match="memmachine-client\\[anthropic\\]"):
        await install_skill(skill_file, "anthropic")


@pytest.mark.asyncio
async def test_missing_openai_import_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "retrieve_skill.md"
    skill_file.write_text("# Retrieve", encoding="utf-8")
    original_import_module = importlib.import_module

    def _patched_import_module(name: str, package: str | None = None):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _patched_import_module)

    with pytest.raises(ImportError, match="memmachine-client\\[openai\\]"):
        await install_skill(skill_file, "openai")
