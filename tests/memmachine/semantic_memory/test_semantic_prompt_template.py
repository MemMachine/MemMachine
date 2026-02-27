"""Tests for semantic prompt template entity type classification."""

from memmachine.semantic_memory.util.semantic_prompt_template import (
    build_update_prompt,
)

ENTITY_TYPES = [
    "Person",
    "Location",
    "Event",
    "Concept",
    "Organization",
    "Temporal",
    "Preference",
    "Other",
]

SAMPLE_TAGS = {"Demographic Information": "Basic user demographics"}


def _build_prompt() -> str:
    return build_update_prompt(tags=SAMPLE_TAGS)


def test_prompt_contains_entity_type_classification_section():
    """The prompt should have an Entity Type Classification section."""
    prompt = _build_prompt()
    assert "Entity Type Classification:" in prompt


def test_prompt_contains_all_entity_types():
    """All eight entity types from the closed set must appear in the prompt."""
    prompt = _build_prompt()
    for entity_type in ENTITY_TYPES:
        assert entity_type in prompt, f"Missing entity type: {entity_type}"


def test_prompt_entity_type_field_in_add_examples():
    """ADD command examples should include the entity_type field."""
    prompt = _build_prompt()
    assert '"entity_type": "Person"' in prompt
    assert '"entity_type": "Preference"' in prompt
    assert '"entity_type": "Concept"' in prompt
    assert '"entity_type": "Temporal"' in prompt


def test_prompt_delete_example_has_no_entity_type():
    """DELETE command examples should NOT include entity_type."""
    prompt = _build_prompt()
    # Find the delete example block and verify no entity_type nearby
    delete_idx = prompt.index('"command": "delete"')
    # The next command block starts at the next "command" occurrence
    next_cmd_idx = prompt.index('"command"', delete_idx + 1)
    delete_block = prompt[delete_idx:next_cmd_idx]
    assert "entity_type" not in delete_block


def test_prompt_entity_type_is_optional():
    """The prompt should indicate entity_type is optional / can be omitted."""
    prompt = _build_prompt()
    assert "omit" in prompt.lower() or "optional" in prompt.lower()


def test_prompt_entity_type_descriptions():
    """Each entity type should have a descriptive explanation."""
    prompt = _build_prompt()
    assert "features about people" in prompt
    assert "features about places" in prompt
    assert "features about events" in prompt
    assert "features about abstract ideas" in prompt
    assert "features about companies" in prompt
    assert "time-related information" in prompt
    assert "user preferences" in prompt
    assert "do not fit any" in prompt


def test_prompt_first_add_example_has_entity_type():
    """The very first ADD example (unicode_for_math) should include entity_type."""
    prompt = _build_prompt()
    # Find the first ADD command format example
    idx = prompt.index('"feature": "unicode_for_math"')
    # entity_type should appear nearby (within the same JSON object)
    nearby = prompt[idx : idx + 200]
    assert '"entity_type"' in nearby


def test_prompt_name_example_has_person_type():
    """The 'Katara' name example should classify as Person."""
    prompt = _build_prompt()
    katara_idx = prompt.index('"value": "Katara"')
    nearby = prompt[katara_idx : katara_idx + 200]
    assert '"entity_type": "Person"' in nearby


def test_prompt_with_custom_tags():
    """Entity type instructions should appear regardless of which tags are passed."""
    prompt = build_update_prompt(
        tags={"Custom Tag": "A custom tag for testing"},
    )
    assert "Entity Type Classification:" in prompt
    for entity_type in ENTITY_TYPES:
        assert entity_type in prompt


def test_prompt_with_description():
    """Entity type instructions should appear when a description is provided."""
    prompt = build_update_prompt(
        tags=SAMPLE_TAGS,
        description="This is a custom description.",
    )
    assert "Entity Type Classification:" in prompt
    assert "This is a custom description." in prompt
