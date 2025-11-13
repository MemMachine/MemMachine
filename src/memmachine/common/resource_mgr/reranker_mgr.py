import boto3

from ...common.configuration.reranker_conf import RerankerConf
from ...common.reranker import Reranker
from ..embedder import Embedder


class RerankerMgr:
    def __init__(self, conf: RerankerConf):
        self.conf = conf
        self.rerankers: dict[str, Reranker] = {}

    def build_all(self, embedders: dict[str, Embedder]) -> dict[str, Reranker]:
        self._build_bm25_rerankers()
        self._build_cross_encoder_rerankers()
        self._build_amazon_bedrock_rerankers()
        self._build_embedder_rerankers(embedders)
        self._build_identity_rerankers()
        self._build_rrf_hybrid_rerankers()
        return self.rerankers

    def get_reranker(self, reranker_id: str) -> Reranker:
        if reranker_id not in self.rerankers:
            raise ValueError(f"Reranker with id {reranker_id} not found.")
        return self.rerankers[reranker_id]

    def _build_bm25_rerankers(self):
        for name, conf in self.conf.bm25.items():
            from memmachine.common.reranker.bm25_reranker import BM25Reranker

            self.rerankers[name] = BM25Reranker(conf)

    def _build_cross_encoder_rerankers(self):
        for name, conf in self.conf.cross_encoder.items():
            from memmachine.common.reranker.cross_encoder_reranker import (
                CrossEncoderReranker,
            )

            self.rerankers[name] = CrossEncoderReranker(conf)

    def _build_amazon_bedrock_rerankers(self):
        for name, conf in self.conf.amazon_bedrock.items():
            from memmachine.common.reranker.amazon_bedrock_reranker import (
                AmazonBedrockReranker,
                AmazonBedrockRerankerParams,
            )

            client = boto3.client(
                "bedrock-agent-runtime",
                region_name=conf.region,
                aws_access_key_id=conf.aws_access_key_id,
                aws_secret_access_key=conf.aws_secret_access_key,
            )
            params = AmazonBedrockRerankerParams(
                client=client,
                region=conf.region,
                model_id=conf.model_id,
                additional_model_request_fields=conf.additional_model_request_fields,
            )
            self.rerankers[name] = AmazonBedrockReranker(params)

    def _build_embedder_rerankers(self, embedders: dict[str, Embedder]):
        from ...common.reranker.embedder_reranker import (
            EmbedderReranker,
            EmbedderRerankerParams,
        )

        for name, conf in self.conf.embedder.items():
            embedder = embedders.get(conf.embedder_id, None)
            if embedder is None:
                raise ValueError(
                    f"Embedder with id {conf.embedder_id} not found for "
                    f"EmbedderReranker {name}."
                )
            params = EmbedderRerankerParams(embedder=embedder)
            self.rerankers[name] = EmbedderReranker(params)

    def _build_identity_rerankers(self):
        for name, conf in self.conf.identity.items():
            from ...common.reranker.identity_reranker import IdentityReranker

            self.rerankers[name] = IdentityReranker()

    def _build_rrf_hybrid_rerankers(self):
        """Build RRF hybrid rerankers by combining existing rerankers.

        This method must be called after all individual rerankers have been built,
        """
        for name, conf in self.conf.rrf_hybrid.items():
            from ...common.reranker.rrf_hybrid_reranker import (
                RRFHybridReranker,
                RRFHybridRerankerParams,
            )

            rerankers = []
            for reranker_id in conf.reranker_ids:
                if reranker_id not in self.rerankers:
                    raise ValueError(
                        f"Reranker with id {reranker_id} not found for "
                        f"RRFHybridReranker {name}."
                    )
                rerankers.append(self.rerankers[reranker_id])
            params = RRFHybridRerankerParams(rerankers=rerankers, k=conf.k)
            self.rerankers[name] = RRFHybridReranker(params)
