
from evaluation.semantic_memory.semantic_harness import build_run_config


def test_build_run_config_sets_cluster_threshold():
    config = {
        "semantic_memory": {"cluster_similarity_threshold": 0.3},
        "resources": {"databases": {}},
    }
    run_conf = build_run_config(config, similarity_threshold=1.0)
    assert run_conf["semantic_memory"]["cluster_similarity_threshold"] == 1.0
