from evaluation.semantic_memory.search import format_feature_context


def test_format_feature_context_includes_feature_value():
    feature = {
        "feature_name": "favorite_color",
        "value": "blue",
        "category": "profile_prompt",
        "tag": "facts",
        "metadata": {"citations": ["e1"]},
    }
    context = format_feature_context([feature])
    assert "favorite_color" in context
    assert "blue" in context
