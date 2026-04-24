def format_feature_context(features: list[dict]) -> str:
    lines = []
    for feature in features:
        lines.append(
            f"[{feature['category']}/{feature['tag']}] {feature['feature_name']}: {feature['value']}"
        )
    return "\n".join(lines)
