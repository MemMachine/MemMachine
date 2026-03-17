from evaluation.semantic_memory.semantic_harness import apply_cluster_settings


def test_apply_cluster_settings_sets_threshold():
    conf = {
        "semantic_memory": {"cluster_similarity_threshold": 0.3},
    }
    apply_cluster_settings(conf, similarity_threshold=1.0)
    assert conf["semantic_memory"]["cluster_similarity_threshold"] == 1.0
