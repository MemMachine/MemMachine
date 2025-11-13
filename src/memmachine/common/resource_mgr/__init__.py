from ...common.configuration import Configuration
from ...common.embedder import Embedder
from ...common.language_model import LanguageModel
from ...common.resource_mgr.embedder_mgr import EmbedderMgr
from ...common.resource_mgr.language_model_mgr import LanguageModelMgr
from ...common.resource_mgr.storage_mgr import StorageMgr
from ...episodic_memory_manager import EpisodicMemoryManager
from ...profile_memory.profile_memory import ProfileMemory
from ...profile_memory.storage.asyncpg_profile import (
    AsyncPgProfileStorage,
    AsyncPgProfileStorageParams,
)
from ...session_manager import SessionDataManagerImpl
from ...session_manager_interface import SessionDataManager
from ..configuration.episodic_config import EpisodicMemoryManagerParams
from ..reranker import Reranker
from ..vector_graph_store import VectorGraphStore
from .reranker_mgr import RerankerMgr


class ResourceMgr:
    def __init__(self, conf: Configuration):
        self._conf = conf
        self._conf.logging.apply()
        self._storage_mgr: StorageMgr = StorageMgr(self._conf.storage)
        self._embedder_mgr: EmbedderMgr = EmbedderMgr(self._conf.embeder)
        self._model_mgr: LanguageModelMgr = LanguageModelMgr(self._conf.model)
        self._reranker_mgr: RerankerMgr = RerankerMgr(self._conf.reranker)
        self._session_data_mgr: SessionDataManager | None = None
        self._episodic_memory_manager: EpisodicMemoryManager | None = None
        self._profile_memory: ProfileMemory | None = None

    def build(self):
        self._storage_mgr.build_all(validate=True)
        self._embedder_mgr.build_all()
        self._model_mgr.build_all()
        self._reranker_mgr.build_all(self._embedder_mgr.embedders)

    def close(self):
        if self._episodic_memory_manager is not None:
            self._episodic_memory_manager.close()
        if self._profile_memory is not None:
            self._profile_memory.cleanup()
        self._storage_mgr.close()

    def get_graph_store(self, name: str) -> VectorGraphStore:
        return self._storage_mgr.get_graph_store(name)

    def get_embedder(self, name: str) -> Embedder:
        return self._embedder_mgr.get_embedder(name)

    def get_model(self, name: str) -> LanguageModel:
        return self._model_mgr.get_model(name)

    def get_reranker(self, name: str) -> Reranker:
        return self._reranker_mgr.get_reranker(name)

    @property
    def session_data_manager(self) -> SessionDataManager:
        if self._session_data_mgr is not None:
            return self._session_data_mgr
        engine = self._storage_mgr.get_sql_engine(self._conf.sessiondb.storage_id)
        self._session_data_mgr = SessionDataManagerImpl(engine)
        return self._session_data_mgr

    @property
    def episodic_memory_manager(self) -> EpisodicMemoryManager:
        if self._episodic_memory_manager is not None:
            return self._episodic_memory_manager
        params = EpisodicMemoryManagerParams(
            session_storage=self.session_data_manager,
        )
        self._episodic_memory_manager = EpisodicMemoryManager(params)
        return self._episodic_memory_manager

    @property
    def profile_memory(self) -> ProfileMemory:
        if self._profile_memory is not None:
            return self._profile_memory
        conf = self._conf.profile_memory
        model = self._model_mgr.get_model(conf.llm_moel)
        embedder = self._embedder_mgr.get_embedder(conf.embedding_model)

        pg_storage_params = AsyncPgProfileStorageParams(
            pool=self._storage_mgr.get_postgres(conf.database)
        )
        pg_storage = AsyncPgProfileStorage(pg_storage_params)
        profile_prompt = self._conf.prompt.profile_prompt
        self._profile_memory = ProfileMemory(
            model=model,
            embeddings=embedder,
            prompt=profile_prompt,
            profile_storage=pg_storage,
        )
        return self._profile_memory
