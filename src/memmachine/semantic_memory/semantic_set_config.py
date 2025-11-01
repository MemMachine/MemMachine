from abc import ABC, abstractmethod

from pydantic import InstanceOf

from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_model import SemanticType


class ConfigManager(ABC):
    class SetConfig(ABC):
        @abstractmethod
        def get_language_model(self) -> LanguageModel:
            raise NotImplementedError

        @abstractmethod
        def get_embedder_model(self) -> LanguageModel:
            raise NotImplementedError

        @abstractmethod
        def get_semantic_types(self) -> SemanticType:
            raise NotImplementedError

    @abstractmethod
    def get_set_config(self, set_id: str) -> InstanceOf[SetConfig]:
        raise NotImplementedError
