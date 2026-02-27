"""Unit tests for Neo4j utility functions and helper methods.

These tests cover:
- Identifier sanitization and desanitization
- Filter expression rendering
- Field reference resolution
- Comparison condition generation
- Value adaptation
- Entity type label extraction
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.neo4j_utils import (
    ENTITY_TYPE_PREFIX,
    desanitize_entity_type,
    sanitize_entity_type,
)
from memmachine.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
    _desanitize_identifier,
    _sanitize_identifier,
)


class TestIdentifierSanitization:
    """Test identifier sanitization for Neo4j labels and index names."""

    def test_sanitize_empty_string(self):
        """Empty string should return special marker."""
        assert _sanitize_identifier("") == "_u0_"

    def test_sanitize_alphanumeric(self):
        """Alphanumeric strings should pass through unchanged."""
        assert _sanitize_identifier("test123") == "test123"
        assert _sanitize_identifier("MyLabel") == "MyLabel"

    def test_sanitize_special_characters(self):
        """Special characters should be hex-encoded."""
        # Slash becomes _u2f_
        assert _sanitize_identifier("org/project") == "org_u2f_project"
        # Hyphen becomes _u2d_
        assert _sanitize_identifier("my-project") == "my_u2d_project"
        # Space becomes _u20_
        assert _sanitize_identifier("my project") == "my_u20_project"

    def test_sanitize_complex_string(self):
        """Complex strings with multiple special chars."""
        result = _sanitize_identifier("user@example.com")
        assert result == "user_u40_example_u2e_com"

    def test_sanitize_unicode(self):
        """Unicode letters are considered alphanumeric by isalnum()."""
        result = _sanitize_identifier("café")
        # é is alphanumeric in Python, so it passes through
        assert result == "café"

    def test_desanitize_empty_string(self):
        """Empty string should return empty."""
        assert _desanitize_identifier("") == ""

    def test_desanitize_alphanumeric(self):
        """Alphanumeric strings should pass through unchanged."""
        assert _desanitize_identifier("test123") == "test123"

    def test_desanitize_simple(self):
        """Hex-encoded chars should be restored."""
        assert _desanitize_identifier("org_u2f_project") == "org/project"
        assert _desanitize_identifier("my_u2d_project") == "my-project"
        assert _desanitize_identifier("my_u20_project") == "my project"

    def test_desanitize_complex(self):
        """Complex encoded strings should be fully restored."""
        assert _desanitize_identifier("user_u40_example_u2e_com") == "user@example.com"

    def test_sanitize_desanitize_roundtrip(self):
        """Sanitize then desanitize should return original."""
        test_cases = [
            "org/project",
            "user@example.com",
            "my-project-123",
            "test_with_underscores",
            "special!@#$%chars",
        ]
        for original in test_cases:
            sanitized = _sanitize_identifier(original)
            restored = _desanitize_identifier(sanitized)
            assert restored == original, f"Roundtrip failed for: {original}"

    def test_desanitize_invalid_hex(self):
        """Invalid hex sequences should be left as-is."""
        # Invalid hex (zzz is not valid hex)
        assert _desanitize_identifier("test_uzzz_") == "test_uzzz_"


class TestFilterExpressionRendering:
    """Test Neo4j filter expression rendering."""

    @pytest.fixture
    def storage(self):
        """Create a Neo4jSemanticStorage instance with mock driver."""
        mock_driver = MagicMock()
        storage = Neo4jSemanticStorage(driver=mock_driver, owns_driver=False)
        return storage

    def test_render_simple_equality(self, storage):
        """Test rendering simple equality comparison."""
        expr = FilterComparison(field="category", op="=", value="preferences")
        condition, params = storage._render_filter_expr("f", expr)

        assert "f.category" in condition
        assert "=" in condition
        assert "$" in condition
        assert len(params) == 1
        assert "preferences" in params.values()

    def test_render_greater_than(self, storage):
        """Test rendering greater than comparison."""
        expr = FilterComparison(field="count", op=">", value=5)
        condition, params = storage._render_filter_expr("f", expr)

        assert "f.count" in condition
        assert ">" in condition
        assert len(params) == 1
        assert 5 in params.values()

    def test_render_less_than_or_equal(self, storage):
        """Test rendering <= comparison."""
        expr = FilterComparison(field="score", op="<=", value=100)
        condition, params = storage._render_filter_expr("f", expr)

        assert "f.score" in condition
        assert "<=" in condition
        assert 100 in params.values()

    def test_render_in_operator(self, storage):
        """Test rendering IN operator with list."""
        expr = FilterComparison(field="tag", op="in", value=["food", "drink"])
        condition, params = storage._render_filter_expr("f", expr)

        assert "f.tag" in condition
        assert "IN" in condition
        assert len(params) == 1
        param_value = next(iter(params.values()))
        assert isinstance(param_value, list)
        assert "food" in param_value
        assert "drink" in param_value

    def test_render_is_null(self, storage):
        """Test rendering IS NULL operator."""
        expr = FilterComparison(field="optional_field", op="is_null", value=None)
        condition, params = storage._render_filter_expr("f", expr)

        assert "f.optional_field" in condition
        assert "IS NULL" in condition
        assert len(params) == 0  # No parameters for IS NULL

    def test_render_is_not_null(self, storage):
        """Test rendering IS NOT NULL operator."""
        expr = FilterComparison(field="required_field", op="is_not_null", value=None)
        condition, params = storage._render_filter_expr("f", expr)

        assert "f.required_field" in condition
        assert "IS NOT NULL" in condition
        assert len(params) == 0

    def test_render_and_expression(self, storage):
        """Test rendering AND expression."""
        left = FilterComparison(field="category", op="=", value="facts")
        right = FilterComparison(field="tag", op="=", value="location")
        expr = FilterAnd(left=left, right=right)

        condition, params = storage._render_filter_expr("f", expr)

        assert "AND" in condition
        assert "f.category" in condition
        assert "f.tag" in condition
        assert len(params) == 2

    def test_render_or_expression(self, storage):
        """Test rendering OR expression."""
        left = FilterComparison(field="category", op="=", value="facts")
        right = FilterComparison(field="category", op="=", value="preferences")
        expr = FilterOr(left=left, right=right)

        condition, params = storage._render_filter_expr("f", expr)

        assert "OR" in condition
        assert "f.category" in condition
        assert len(params) == 2

    def test_render_nested_and_or(self, storage):
        """Test rendering nested AND/OR expressions."""
        # (category = 'facts' AND tag = 'location') OR (category = 'preferences')
        left_and = FilterAnd(
            left=FilterComparison(field="category", op="=", value="facts"),
            right=FilterComparison(field="tag", op="=", value="location"),
        )
        right = FilterComparison(field="category", op="=", value="preferences")
        expr = FilterOr(left=left_and, right=right)

        condition, params = storage._render_filter_expr("f", expr)

        assert "AND" in condition
        assert "OR" in condition
        assert len(params) == 3

    def test_render_metadata_field(self, storage):
        """Test rendering metadata field reference."""
        expr = FilterComparison(field="m.confidence", op=">", value=0.8)
        condition, params = storage._render_filter_expr("f", expr)

        # Should use metadata__ prefix
        assert "metadata__confidence" in condition
        assert ">" in condition
        assert 0.8 in params.values()

    def test_render_metadata_field_alt_syntax(self, storage):
        """Test rendering metadata field with 'metadata.' prefix."""
        expr = FilterComparison(field="metadata.source", op="=", value="api")
        condition, params = storage._render_filter_expr("f", expr)

        assert "metadata__source" in condition
        assert "api" in params.values()

    def test_render_created_at_timestamp(self, storage):
        """Test rendering created_at field with datetime conversion."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        expr = FilterComparison(field="created_at", op=">", value=dt)
        condition, params = storage._render_filter_expr("f", expr)

        # Should use created_at_ts and convert datetime to timestamp
        assert "f.created_at_ts" in condition
        param_value = next(iter(params.values()))
        assert isinstance(param_value, (int, float))

    def test_render_updated_at_timestamp(self, storage):
        """Test rendering updated_at field."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        expr = FilterComparison(field="updated_at", op=">=", value=dt)
        condition, params = storage._render_filter_expr("f", expr)

        assert "f.updated_at_ts" in condition
        param_value = next(iter(params.values()))
        assert isinstance(param_value, (int, float))

    def test_invalid_operator_raises_error(self, storage):
        """Test that invalid operator raises ValueError."""
        expr = FilterComparison(field="field", op="invalid_op", value="value")

        with pytest.raises(ValueError, match="Unsupported operator"):
            storage._render_filter_expr("f", expr)

    def test_in_operator_with_non_list_raises_error(self, storage):
        """Test that IN operator with non-list raises ValueError."""
        expr = FilterComparison(field="field", op="in", value="not_a_list")

        with pytest.raises(ValueError, match="IN comparison requires a list"):
            storage._render_filter_expr("f", expr)

    def test_comparison_operator_with_list_raises_error(self, storage):
        """Test that comparison operators reject list values."""
        expr = FilterComparison(field="field", op="=", value=["a", "b"])

        with pytest.raises(ValueError, match="cannot accept list values"):
            storage._render_filter_expr("f", expr)

    def test_unique_filter_params(self, storage):
        """Test that filter params have unique names."""
        expr1 = FilterComparison(field="field1", op="=", value="value1")
        expr2 = FilterComparison(field="field2", op="=", value="value2")

        _, params1 = storage._render_filter_expr("f", expr1)
        _, params2 = storage._render_filter_expr("f", expr2)

        # Parameter names should be different
        param_names1 = set(params1.keys())
        param_names2 = set(params2.keys())
        assert param_names1 != param_names2


class TestFieldReferenceResolution:
    """Test field reference resolution."""

    @pytest.fixture
    def storage(self):
        """Create a Neo4jSemanticStorage instance with mock driver."""
        mock_driver = MagicMock()
        storage = Neo4jSemanticStorage(driver=mock_driver, owns_driver=False)
        return storage

    def test_resolve_regular_field(self, storage):
        """Test resolving 'category' alias to the Neo4j property 'category_name'."""
        field_ref, adapter = storage._resolve_field_reference("f", "category")
        assert field_ref == "f.category_name"
        assert adapter is None

    def test_resolve_created_at(self, storage):
        """Test resolving created_at field."""
        field_ref, adapter = storage._resolve_field_reference("f", "created_at")
        assert field_ref == "f.created_at_ts"
        assert adapter is not None  # Should have datetime adapter

    def test_resolve_created_at_ts(self, storage):
        """Test resolving created_at_ts field."""
        field_ref, adapter = storage._resolve_field_reference("f", "created_at_ts")
        assert field_ref == "f.created_at_ts"
        assert adapter is not None

    def test_resolve_updated_at(self, storage):
        """Test resolving updated_at field."""
        field_ref, adapter = storage._resolve_field_reference("f", "updated_at")
        assert field_ref == "f.updated_at_ts"
        assert adapter is not None

    def test_resolve_metadata_field_m_prefix(self, storage):
        """Test resolving metadata field with m. prefix."""
        field_ref, adapter = storage._resolve_field_reference("f", "m.confidence")
        assert field_ref == "f.metadata__confidence"
        assert adapter is None

    def test_resolve_metadata_field_metadata_prefix(self, storage):
        """Test resolving metadata field with metadata. prefix."""
        field_ref, adapter = storage._resolve_field_reference("f", "metadata.source")
        assert field_ref == "f.metadata__source"
        assert adapter is None


class TestValueAdaptation:
    """Test value adaptation for filter comparisons."""

    @pytest.fixture
    def storage(self):
        """Create a Neo4jSemanticStorage instance with mock driver."""
        mock_driver = MagicMock()
        storage = Neo4jSemanticStorage(driver=mock_driver, owns_driver=False)
        return storage

    def test_adapt_value_with_no_adapter(self, storage):
        """Test that value passes through when no adapter."""
        value = "test_value"
        result = storage._adapt_filter_value(value, None)
        assert result == value

    def test_adapt_none_value(self, storage):
        """Test that None value passes through."""
        result = storage._adapt_filter_value(None, lambda x: x * 2)
        assert result is None

    def test_adapt_value_with_adapter(self, storage):
        """Test that adapter is applied."""
        value = 5

        def double(x: int) -> int:
            return x * 2

        result = storage._adapt_filter_value(value, double)
        assert result == 10


class TestSharedEntityTypeSanitization:
    """Tests for the shared entity type sanitization utilities."""

    def test_sanitize_simple_type(self):
        assert sanitize_entity_type("Person") == "ENTITY_TYPE_Person"

    def test_sanitize_type_with_spaces(self):
        result = sanitize_entity_type("My Type")
        assert result.startswith(ENTITY_TYPE_PREFIX)
        assert "My" in result or "_u20_" in result  # space encoded

    def test_desanitize_simple_type(self):
        assert desanitize_entity_type("ENTITY_TYPE_Person") == "Person"

    def test_round_trip_all_standard_types(self):
        for etype in [
            "Person",
            "Location",
            "Event",
            "Concept",
            "Organization",
            "Temporal",
            "Preference",
            "Other",
        ]:
            sanitized = sanitize_entity_type(etype)
            assert sanitized.startswith(ENTITY_TYPE_PREFIX)
            assert desanitize_entity_type(sanitized) == etype

    def test_round_trip_special_characters(self):
        original = "My Type!"
        sanitized = sanitize_entity_type(original)
        assert desanitize_entity_type(sanitized) == original


class TestNodeToEntryEntityType:
    """Tests for _node_to_entry entity type extraction from Neo4j labels."""

    @pytest.fixture
    def storage(self):
        mock_driver = MagicMock()
        return Neo4jSemanticStorage(driver=mock_driver, owns_driver=False)

    def _make_mock_node(
        self,
        labels: frozenset[str],
        props: dict | None = None,
    ):
        """Create a mock Neo4j node with labels and properties."""
        default_props = {
            "set_id": "test-set",
            "category_name": "Profile",
            "tag": "Demographics",
            "feature": "name",
            "value": "Alice",
            "embedding": [1.0, 2.0],
            "citations": [],
            "created_at_ts": 1000.0,
            "updated_at_ts": 1000.0,
        }
        if props:
            default_props.update(props)

        node = MagicMock()
        node.labels = labels
        node.element_id = "4:abc:123"
        node.__iter__ = lambda self: iter(default_props.items())
        node.__contains__ = lambda self, key: key in default_props
        node.get = lambda key, default=None: default_props.get(key, default)
        # Make dict(node) work
        type(node).__iter__ = lambda self: iter(default_props)
        type(node).__getitem__ = lambda self, key: default_props[key]
        node.keys = lambda: default_props.keys()
        node.values = lambda: default_props.values()
        node.items = lambda: default_props.items()
        return node

    def test_node_with_entity_type_label(self, storage):
        """Node with ENTITY_TYPE_Person label should populate entity_type."""
        node = self._make_mock_node(
            frozenset({"Feature", "FeatureSet_test", "ENTITY_TYPE_Person"})
        )
        entry = storage._node_to_entry(node)
        assert entry.entity_type == "Person"

    def test_node_without_entity_type_label(self, storage):
        """Node without ENTITY_TYPE_* label should have None entity_type."""
        node = self._make_mock_node(frozenset({"Feature", "FeatureSet_test"}))
        entry = storage._node_to_entry(node)
        assert entry.entity_type is None

    def test_node_with_location_entity_type(self, storage):
        """Node with ENTITY_TYPE_Location should extract 'Location'."""
        node = self._make_mock_node(frozenset({"Feature", "ENTITY_TYPE_Location"}))
        entry = storage._node_to_entry(node)
        assert entry.entity_type == "Location"

    def test_entry_to_model_passes_entity_type(self, storage):
        """_entry_to_model should set entity_type on SemanticFeature."""
        import numpy as np

        from memmachine.semantic_memory.storage.neo4j_semantic_storage import (
            _FeatureEntry,
        )

        entry = _FeatureEntry(
            feature_id="f-1",
            set_id="test-set",
            category_name="Profile",
            tag="Demographics",
            feature_name="name",
            value="Alice",
            embedding=np.array([1.0, 2.0]),
            metadata=None,
            citations=[],
            created_at_ts=1000.0,
            updated_at_ts=1000.0,
            entity_type="Person",
        )
        model = storage._entry_to_model(entry, load_citations=False)
        assert model.entity_type == "Person"

    def test_entry_to_model_none_entity_type(self, storage):
        """_entry_to_model with no entity_type returns None."""
        import numpy as np

        from memmachine.semantic_memory.storage.neo4j_semantic_storage import (
            _FeatureEntry,
        )

        entry = _FeatureEntry(
            feature_id="f-2",
            set_id="test-set",
            category_name="Profile",
            tag="Demographics",
            feature_name="name",
            value="Bob",
            embedding=np.array([1.0, 2.0]),
            metadata=None,
            citations=[],
            created_at_ts=1000.0,
            updated_at_ts=1000.0,
        )
        model = storage._entry_to_model(entry, load_citations=False)
        assert model.entity_type is None
