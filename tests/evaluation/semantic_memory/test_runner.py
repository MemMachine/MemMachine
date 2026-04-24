from evaluation.semantic_memory.runner import build_variant_plan


def test_build_variant_plan_has_two_variants():
    variants = build_variant_plan()
    assert "clustered" in variants
    assert "no_cluster" in variants
