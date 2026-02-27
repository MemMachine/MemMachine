import pytest
import pytest_asyncio

from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import EpisodeStorage
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.semantic_memory import SemanticService
from memmachine_server.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    SemanticCategory,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from server_tests.memmachine_server.semantic_memory.semantic_test_utils import (
    LengthEmbedder,
    SpyEmbedder,
)


@pytest.fixture
def spy_embedder() -> SpyEmbedder:
    return SpyEmbedder()


@pytest.fixture
def semantic_resource_manager(
    spy_embedder: SpyEmbedder,
    mock_llm_model,
):
    class ResourceManager:
        def __init__(
            self,
            embedder: Embedder,
            llm: LanguageModel,
        ):
            self._embedder = embedder
            self._llm = llm

        async def get_embedder(self, s: str) -> Embedder:
            if s.isdigit():
                return LengthEmbedder(int(s))

            return self._embedder

        async def get_language_model(self, _: str) -> LanguageModel:
            return self._llm

    return ResourceManager(
        embedder=spy_embedder,
        llm=mock_llm_model,
    )


@pytest.fixture
def semantic_category_retriever():
    prompt = RawSemanticPrompt(
        update_prompt="update-semantic-memory",
        consolidation_prompt="consolidate-semantic-memory",
    )
    cat = SemanticCategory(
        name="Profile",
        prompt=prompt,
    )

    def get_categories(_: str) -> list[SemanticCategory]:
        return [cat]

    return get_categories


@pytest_asyncio.fixture
async def semantic_service(
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    semantic_config_storage: SemanticConfigStorage,
    semantic_resource_manager,
    mock_llm_model,
    spy_embedder: SpyEmbedder,
    semantic_category_retriever,
):
    params = SemanticService.Params(
        semantic_storage=semantic_storage,
        episode_storage=episode_storage,
        semantic_config_storage=semantic_config_storage,
        feature_update_interval_sec=0.05,
        uningested_message_limit=10,
        resource_manager=semantic_resource_manager,
        default_embedder=spy_embedder,
        default_embedder_name="default_embedder",
        default_language_model=mock_llm_model,
        default_category_retriever=semantic_category_retriever,
    )
    service = SemanticService(params)
    await service.start()
    yield service
    await service.stop()
