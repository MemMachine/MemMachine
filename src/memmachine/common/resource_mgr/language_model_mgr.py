"""
Builder for LanguageModel instances.
"""

from memmachine.common.configuration.model_conf import LanguageModelConf
from memmachine.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockLanguageModel,
)
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.language_model.openai_compatible_language_model import (
    OpenAICompatibleLanguageModel,
)
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel


class LanguageModelMgr:
    def __init__(self, conf: LanguageModelConf):
        self.conf = conf
        self._language_models: dict[str, LanguageModel] = {}

    def build_all(self) -> dict[str, LanguageModel]:
        self._build_openai_model()
        self._build_aws_bedrock_model()
        self._build_openai_compatible_model()
        return self.language_models

    def get_language_model(self, name: str) -> LanguageModel:
        if name not in self.language_models:
            raise ValueError(f"Language model with name {name} not found.")
        return self.language_models[name]

    @property
    def language_models(self) -> dict[str, LanguageModel]:
        return self._language_models

    def get_model(self, name: str) -> LanguageModel:
        if name not in self.language_models:
            raise ValueError(f"Language model with name {name} not found.")
        return self._language_models[name]

    def _build_openai_model(self):
        for name, conf in self.conf.openai_confs.items():
            self._language_models[name] = OpenAILanguageModel(conf)

    def _build_aws_bedrock_model(self):
        for name, conf in self.conf.aws_bedrock_confs.items():
            self._language_models[name] = AmazonBedrockLanguageModel(conf)

    def _build_openai_compatible_model(self):
        for name, conf in self.conf.openai_compatible_confs.items():
            self._language_models[name] = OpenAICompatibleLanguageModel(conf)
