"""Unit tests for LLM-friendly memory formatting functions."""

import json
from datetime import datetime, timezone

from memmachine_common.api.spec import (
    Episode,
    EpisodeResponse,
    EpisodicSearchLongTermMemory,
    EpisodicSearchResult,
    EpisodicSearchShortTermMemory,
    SearchResult,
    SearchResultContent,
    SemanticFeature,
)

from memmachine_client.format import (
    format_episodes,
    format_search_result,
    format_semantic_memories,
)


class TestFormatEpisodes:
    """Tests for format_episodes."""

    def test_empty(self):
        assert format_episodes([]) == ""

    def test_single_episode_response(self):
        ep = EpisodeResponse(
            uid="1",
            content="Hello world",
            producer_id="user_1",
            producer_role="user",
            created_at=datetime(2024, 1, 15, 13, 30, tzinfo=timezone.utc),
        )
        result = format_episodes([ep])
        assert (
            result == '[Monday, January 15, 2024 at 01:30 PM] user_1: "Hello world"\n'
        )

    def test_multiple_episodes(self):
        eps = [
            EpisodeResponse(
                uid="1",
                content="First message",
                producer_id="user_1",
                producer_role="user",
                created_at=datetime(2024, 3, 5, 9, 0, tzinfo=timezone.utc),
            ),
            EpisodeResponse(
                uid="2",
                content="Second message",
                producer_id="assistant_1",
                producer_role="assistant",
                created_at=datetime(2024, 3, 5, 9, 1, tzinfo=timezone.utc),
            ),
        ]
        result = format_episodes(eps)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "user_1" in lines[0]
        assert "assistant_1" in lines[1]

    def test_episode_without_created_at(self):
        ep = EpisodeResponse(
            uid="1",
            content="No timestamp",
            producer_id="user_1",
            producer_role="user",
            created_at=None,
        )
        result = format_episodes([ep])
        assert result == 'user_1: "No timestamp"\n'

    def test_list_episode_type(self):
        ep = Episode(
            uid="1",
            content="Listed episode",
            session_key="sess_1",
            producer_id="user_1",
            producer_role="user",
            created_at=datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc),
        )
        result = format_episodes([ep])
        assert (
            result == '[Saturday, June 01, 2024 at 02:00 PM] user_1: "Listed episode"\n'
        )

    def test_content_json_escaped(self):
        ep = EpisodeResponse(
            uid="1",
            content='She said "hello"',
            producer_id="user_1",
            producer_role="user",
            created_at=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        result = format_episodes([ep])
        assert json.dumps('She said "hello"') in result

    def test_non_ascii_content_preserved_literally(self):
        """Non-ASCII characters must appear literally in the LLM context, not
        as ``\\uXXXX`` escapes — escaping bloats token counts and degrades
        recall on multilingual content."""
        ep = EpisodeResponse(
            uid="1",
            content="寿司 café 🍕 naïve résumé Привет",
            producer_id="user_1",
            producer_role="user",
            created_at=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        result = format_episodes([ep])
        assert "寿司" in result
        assert "café" in result
        assert "🍕" in result
        assert "naïve" in result
        assert "résumé" in result
        assert "Привет" in result
        # Sanity: no escaped CJK / cyrillic / accented sequences.
        assert "\\u" not in result

    def test_non_ascii_content_lossless_roundtrip(self):
        """The JSON-encoded portion must round-trip back to the original
        string so downstream LLM consumers (and any client-side
        post-processing) still see correct text."""
        original = '日本語 — "quoted" + emoji 🎉'
        ep = EpisodeResponse(
            uid="1",
            content=original,
            producer_id="user_1",
            producer_role="user",
            created_at=None,
        )
        result = format_episodes([ep])
        json_part = result.removeprefix("user_1: ").rstrip("\n")
        assert json.loads(json_part) == original

    def test_output_is_utf8_encodable(self):
        """The LLM-visible string must be safe to send over UTF-8 transports
        (HTTP body, logging sinks). ``ensure_ascii=False`` produces
        unescaped surrogates only for malformed inputs; clean Unicode must
        encode without error."""
        ep = EpisodeResponse(
            uid="1",
            content="Mixed: ASCII + 中文 + 🚀",
            producer_id="user_1",
            producer_role="user",
            created_at=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        result = format_episodes([ep])
        encoded = result.encode("utf-8")
        assert encoded.decode("utf-8") == result


class TestFormatSemanticMemories:
    """Tests for format_semantic_memories."""

    def test_empty(self):
        result = format_semantic_memories([])
        assert result == "{}"

    def test_single_feature(self):
        feature = SemanticFeature(
            category="profile",
            tag="preferences",
            feature_name="favorite_food",
            value="pizza",
        )
        result = format_semantic_memories([feature])
        parsed = json.loads(result)
        assert parsed == {"preferences": {"favorite_food": "pizza"}}

    def test_groups_by_tag(self):
        features = [
            SemanticFeature(
                category="profile",
                tag="preferences",
                feature_name="food",
                value="pizza",
            ),
            SemanticFeature(
                category="profile",
                tag="preferences",
                feature_name="color",
                value="blue",
            ),
            SemanticFeature(
                category="profile",
                tag="background",
                feature_name="role",
                value="engineer",
            ),
        ]
        result = format_semantic_memories(features)
        parsed = json.loads(result)
        assert parsed == {
            "preferences": {"food": "pizza", "color": "blue"},
            "background": {"role": "engineer"},
        }

    def test_non_ascii_value_preserved_literally(self):
        feature = SemanticFeature(
            category="profile",
            tag="prefs",
            feature_name="favorite_food",
            value="寿司 🍣",
        )
        result = format_semantic_memories([feature])
        assert "寿司" in result
        assert "🍣" in result
        assert "\\u" not in result
        # And the JSON is still valid.
        assert json.loads(result) == {"prefs": {"favorite_food": "寿司 🍣"}}

    def test_non_ascii_tag_and_feature_name_preserved(self):
        feature = SemanticFeature(
            category="profile",
            tag="préférences",
            feature_name="種類",
            value="ramen",
        )
        result = format_semantic_memories([feature])
        assert "préférences" in result
        assert "種類" in result
        assert json.loads(result) == {"préférences": {"種類": "ramen"}}

    def test_metadata_excluded(self):
        feature = SemanticFeature(
            set_id="set_1",
            category="profile",
            tag="info",
            feature_name="name",
            value="Alice",
            metadata=SemanticFeature.Metadata(
                id="feat_1",
                citations=["ep_1", "ep_2"],
                other={"source": "conversation"},
            ),
        )
        result = format_semantic_memories([feature])
        parsed = json.loads(result)
        # Only tag/feature_name/value should appear
        assert parsed == {"info": {"name": "Alice"}}
        assert "set_id" not in result
        assert "citations" not in result
        assert "feat_1" not in result


class TestFormatSearchResult:
    """Tests for format_search_result."""

    def test_empty_result(self):
        result = SearchResult(
            status=0,
            content=SearchResultContent(
                episodic_memory=None,
                semantic_memory=None,
            ),
        )
        assert format_search_result(result) == ""

    def test_episodic_only(self):
        ep = EpisodeResponse(
            uid="1",
            content="Hello",
            producer_id="user_1",
            producer_role="user",
            created_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        result = SearchResult(
            status=0,
            content=SearchResultContent(
                episodic_memory=EpisodicSearchResult(
                    long_term_memory=EpisodicSearchLongTermMemory(episodes=[ep]),
                    short_term_memory=EpisodicSearchShortTermMemory(
                        episodes=[], episode_summary=[]
                    ),
                ),
                semantic_memory=None,
            ),
        )
        formatted = format_search_result(result)
        assert formatted.startswith("[Episodic Memory]\n")
        assert "user_1" in formatted
        assert '"Hello"' in formatted
        assert "[Semantic Memory]" not in formatted

    def test_semantic_only(self):
        feature = SemanticFeature(
            category="profile",
            tag="prefs",
            feature_name="food",
            value="pizza",
        )
        result = SearchResult(
            status=0,
            content=SearchResultContent(
                episodic_memory=None,
                semantic_memory=[feature],
            ),
        )
        formatted = format_search_result(result)
        assert formatted.startswith("[Semantic Memory]\n")
        assert "[Episodic Memory]" not in formatted
        semantic_json = formatted.removeprefix("[Semantic Memory]\n")
        assert json.loads(semantic_json) == {"prefs": {"food": "pizza"}}

    def test_combined(self):
        ep = EpisodeResponse(
            uid="1",
            content="I like pizza",
            producer_id="user_1",
            producer_role="user",
            created_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        feature = SemanticFeature(
            category="profile",
            tag="prefs",
            feature_name="food",
            value="pizza",
        )
        result = SearchResult(
            status=0,
            content=SearchResultContent(
                episodic_memory=EpisodicSearchResult(
                    long_term_memory=EpisodicSearchLongTermMemory(episodes=[ep]),
                    short_term_memory=EpisodicSearchShortTermMemory(
                        episodes=[], episode_summary=[]
                    ),
                ),
                semantic_memory=[feature],
            ),
        )
        formatted = format_search_result(result)
        assert "[Episodic Memory]" in formatted
        assert "[Semantic Memory]" in formatted
        assert formatted.index("[Episodic Memory]") < formatted.index(
            "[Semantic Memory]"
        )
