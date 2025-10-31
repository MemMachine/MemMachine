import os
import json
import re
import time, random
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from collections.abc import Collection
from typing import List, Tuple, Dict, Any, Iterable, DefaultDict
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import boto3

import asyncio
from uuid import uuid4, UUID

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from memmachine.common.utils import async_locked
from memmachine.common.vector_graph_store import Node, Edge
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import Neo4jVectorGraphStore, Neo4jVectorGraphStoreParams
from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel
from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)

os.environ["DSPY_CACHEDIR"] = "/tmp/dspy_cache"
import dspy

EPISODE_GROUP_PROMPT = """You are an expert at grouping related episodes by **content similarity**.

**Goal**:
Analyze the episodes and cluster those that are closely related in content, themes, or topics. Avoid speculative links—use only what's explicitly present.

**Episodes**:
{episodes}

**Rules** (be strict and deterministic):
    - Treat two episodes as related only if they share a central topic (same event/series, person/organization, product/model, place, study, bug/feature, or storyline).
    - Ignore superficial overlap (generic words like “update,” “today,” “news,” dates, or common verbs).
    - Prefer **precision over recall**: if unsure, keep the episode as a singleton.
    - **Transitivity**: If A is related to B and B to C, put A, B, C in the same group.
    - **No overlaps**: Each episode index must appear exactly once in exactly one group.

**Method** (concise):
    1. **Extract key features** per episode: named entities (people/orgs/places), key nouns/phrases, and specific identifiers (e.g., model numbers, issue IDs).
    2. Score pairwise similarity on a 0–10 scale:
        - **10** = near-duplicate
        - **8-9** = strongly about the same specific topic
        - **6-7** = related or semantically similar (weaker tie)
        - **0-5** = unrelated
    3. **Create edges** between episode pairs using this deterministic rule:
        - Add an edge if score ≥ 8; or
        - If score 6-7 and the pair shares at least one anchor: an exact unique identifier (e.g., “ISSUE-1234”, “M3 MacBook Pro”), or a combined anchor of (same named entity AND same event/date/place).
    4. Take the **transitive closure** (connected components) over edges to form groups. If still ambiguous, keep as a singleton.
    5. Break ties by dominant topic; if still ambiguous, use a singleton.
**Output Format** (strict JSON, no additional text before or after the JSON block):
{{
  "groups": [[0,1], [2,3,4] ...]
}}

**Examples**:
Example 1:
Episodes:
    [0][2025-05-12T09:14:03] Mia: “Should we do Yosemite for Memorial Day? We'll need a backcountry permit.”
    [1][2025-05-12T09:40:51] Raj: “I'll submit the permit lottery today—deadline is Friday.”
    [2][2025-05-15T17:40:19] Raj: “We lost the lottery.”
    [3][2025-05-15T17:41:03] Mia: “Let's try Tuolumne first-come or shift dates.”
    [4][2025-05-20T07:52:44] Raj: “If we want Half Dome, we should train for the cables and rent a bear can.”
    [5][2025-05-21T19:12:05] Mia: “The espresso machine keeps beeping—descale light won't clear.”
    [6][2025-05-21T19:13:48] Raj: “Hold both cup buttons for 5 seconds to reset.”
    [7][2025-05-16T10:03:11] Mia: “My React app won't compile—Vite can't find Tailwind classes.”
    [8][2025-05-16T11:25:33] Raj: “Add the Tailwind plugin in postcss; rebuild should fix it.”
Output:
{{
  "groups": [[0,1,2,3,4],[5,6],[7,8]]
}}

Now generate the episode groups based on the provided episodes:
"""

EPISODE_GROUP_SUMMARY_PROMPT ="""You are a precise dialogue summarizer. Write one concise sentence that captures the main topic and current status of the episodes below. All episodes are about the same topic.

**Episodes**:
{episodes}

**Instructions**:
    - Use only facts explicitly stated in the episodes; no speculation.
    - Preserve exact names/terms (places, products, IDs).
    - Focus on the overall goal, key events/changes, and present status/next steps.
    - Length: ≤ 30 words. Form: exactly one sentence.

**Output format**:
Return only the sentence—no quotes, no labels, no extra text.

**Example**:
Episodes:
[2025-05-12T09:14:03] Mia: “Should we do Yosemite for Memorial Day? We'll need a backcountry permit.”
[2025-05-12T09:40:51] Raj: “I'll submit the permit lottery today—deadline is Friday.”
[2025-05-15T17:40:19] Raj: “We lost the lottery.”
[2025-05-15T17:41:03] Mia: “Let's try Tuolumne first-come or shift dates.”
[2025-05-20T07:52:44] Raj: “If we want Half Dome, we should train for the cables and rent a bear can.”
Output (single string):
Yosemite trip planning: permit lottery submitted then lost; considering Tuolumne or new dates, with training and a bear can for a potential Half Dome attempt.

Now generate the episode summary based on the provided episodes:
"""

SUFFICIENCY_CHECK_PROMPT = """You are a meticulous and detail-oriented expert in information retrieval evaluation. Your task is to critically assess whether a set of retrieved documents contains sufficient information to provide a direct and complete answer to a user's query.

**User Query**:
{query}

**Retrieved Documents**:
{retrieved_episodes}

**Instructions**:
Follow these steps to ensure an accurate evaluation:

1.  **Deconstruct the Query**: Break down the user's query into its core informational needs (e.g., who, what, where, when, why, how). Identify all key entities and concepts.
2.  **Scan for Keywords**: Quickly scan the documents for the key entities and concepts from the query. This is a preliminary check for relevance.
3.  **Detailed Analysis**: Read the relevant parts of the documents carefully. Determine if they contain explicit facts that directly answer *all aspects* of the query. Do not rely on making significant inferences or assumptions. The answer should be explicitly stated or very easily pieced together from the text.
4.  **Sufficiency Judgment**:
    *   If the query asked for specific details (names, dates, locations, numbers), and no exact details are provided, label as **insufficient**.
    *   If all parts of the query are directly and explicitly answered, the documents are **sufficient**.
    *   If any significant part of the query is not answered, the documents are **insufficient**.
    *   If you are uncertain, err on the side of caution and label it as **insufficient**. Do not guess.
    *   If query ask for frequencies, lists, such as "how many", "how often", the return should always be **insufficient** since you don't have complete data.
5.  **Formulate Reasoning**: Based on your analysis, write a brief (1-2 sentences) explanation for your judgment.
6.  **Identify Direct Evidence**: List the indices (0-based) of the documents that are clearly needed to answering the query. Ignore documents that are unrelated or tangential.

**Output Format** (strict JSON, no additional text before or after the JSON block):
{{
  "is_sufficient": true or false,
  "reasoning": "Brief, clear explanation for your decision (1-2 sentences).",
  "indices": [index1, index2, ...]
}}

**Examples**:

Example 1 (Sufficient):
Query: "What's Alice's hobbies?"
Documents:
  [0][2022-01-23T14:01:35] "Alice mentioned she loves painting. She spends weekends at art galleries."
  [1][2022-01-23T14:05:40] "Alice's work involves creative projects. She loves to travel off work."
Output:
{{
  "is_sufficient": true,
  "reasoning": "Document 0 and 1 explicitly states that Alice's hobbies are painting and traveling.",
  "indices": [0, 1]
}}

Example 2 (Insufficient - Missing Detail):
Query: "Where did Bob go on vacation last year?"
Documents:
  [0][2024-01-23T14:01:35] "Bob talked about his work project deadline around June."
  [1][2024-01-23T14:05:40] "Bob likes to travel and explore new places. He recently came back from a trip."
Output:
{{
  "is_sufficient": false,
  "reasoning": "The documents mention Bob likes to travel and recently took a trip, but do not specify where he went or if it was last year.",
  "indices": [1]
}}

Example 3 (Insufficient - Tangential Information):
Query: "What are the specs of the new 'Galaxy Z' phone?"
Documents:
  [0][2024-01-23T14:01:35] "The Galaxy Z is rumored to be released next month."
  [1][2024-02-23T14:01:35] "Tech enthusiasts are excited about the upcoming Galaxy Z launch event."
Output:
{{
  "is_sufficient": false,
  "reasoning": "The documents mention the phone's upcoming release but provide no specific technical specifications.",
  "indices": []
}}

Now evaluate:
"""

def rrf(
    lists: list[list[Any]],
    weights: list[float] | None = None,
    k: int = 50,
) -> list[tuple[Any, float]]:
    """
    Reciprocal Rank Fusion (RRF) implementation.
    Args:
        lists (list[list[str]]): List of ranked lists to fuse.
        weights (list[float] | None): Weights for each ranked list. If None, equal weights are used.
        k (int): Constant to control the influence of rank. Smaller k gives more weight to higher ranks.
    Returns:
        list[tuple[str, float]]: List of tuples containing item and its fused score, sorted by score in descending order.
    """
    if not lists:
        return []

    n = len(lists)
    if weights is None:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError(f"RRF: weights length {len(weights)} must equal number of lists {n}")

    scores: dict[str, float] = defaultdict(float)

    for w, L in zip(weights, lists):
        for rank, _id in enumerate(L, start=1):
            if w > 0:
                scores[_id] += w * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class KnowledgeGraph:
    """
    Knowledge Graph object should be instantiated once per session_id.
    """
    def __init__(
        self,
        model_name: str,
        model: AsyncOpenAI,
        embedder: OpenAIEmbedder,
        store: Neo4jVectorGraphStore,
        reranker: AmazonBedrockReranker = None,
    ):
        # Resources
        self._embedder = embedder
        self._store = store
        self._model = model
        self._model_name = model_name
        self._reranker = reranker

        # Local caches
        self._entity_node_map = {}
        self._entity_edge_map = {}
        self._episode_cluster_edge_map = {}
        self._episode_batch: list[Node] = []
        self._processed_triple_texts = set()

        # Constants
        self._embedding_batch_size = 100

        # Performance metrics
        self._perf_episode_grouping_time = 0.0
        self._perf_episode_summary_time = 0.0
        self._perf_entity_search_time = 0.0
        self._perf_relation_extraction_time = 0.0
        self._perf_embedding_time = 0.0
        self._perf_node_creation_time = 0.0
        self._perf_edge_creation_time = 0.0
        self._entity_node_created = 0
        self._relation_edge_created = 0
        self._episode_cluster_node_created = 0

        # Setup edge_extractor
        self._edge_extractor = setup_dialogue_relation_extractor()
        self._extract_sem = asyncio.Semaphore(200)
    
    def setup_dialogue_relation_extractor(self):
        """Setup DSPy-optimized relation extractor for text with coreference resolution"""

        class EdgeExtraction(dspy.Signature):
            """
            Extract semantic relationships from input text with advanced coreference resolution. Return as JSON array.
            
            **Guidelines**:
            - Extract meaningful semantic relationships between entities in the text
            - Resolve complex coreferences in multi-speaker conversations:
            a) Identify all pronouns, definite articles, and vague references (the, this, that, it, they, etc.)
            b) Trace each reference back to its most specific antecedent in the conversation
            c) Replace vague references with the most specific, concrete entity name possible
            d) For events/actions, create descriptive names that capture the essence
            - Handle speaker-specific references:
            a) Resolve possessive pronouns and contextual references to specific person names when possible 
            b) Identify relationships between speakers and mentioned entities
            c) Track entity mentions across multiple speakers
            - Pay special attention to temporal relationships and time information:
            a) Extract when events occurred (dates, times, periods)
            b) Preserve temporal context in relationships
            c) Include time information in relation, do not include in subject/object
            - Use context clues to determine the most appropriate entity name
            - Do not emit duplicate or semantically redundant relationships
            - Return as JSON array of objects with "subject", "relation", "object" fields
            - DO NOT include any duplicate relationships
            - Make sure the resulting JSON is valid and parsable
            - No comments or '//' or explainations in the output JSON, only the JSON array
            - The JSON array should not exceed 16000 characters
            
            **Coreference Resolution Strategy**:
            - Replace any vague or ambiguous references with specific entity names based on conversation context
            - For events and actions, create descriptive names that capture what actually happened
            - For people and organizations, use their full, proper names when available
            - Always preserve temporal information and context
            - Handle speaker-specific references and possessive pronouns

            **Output Format**(strict JSON parsable, no additional text):
            [{
                "subject": "Entity1",
                "relation": "relation1",
                "object": "Entity2"
            },
            {
                "subject": "Entity1",
                "relation": "relation2",
                "object": "Entity2"
            }]

            Before returning the result, do the following step by step:
            1. Remove duplicated or semantically redundant relationships
            2. The resulting relationship dictionaries should have "subject", "relation", "object" fields, if not, fix them.
            2. Parse the resulting JSON to make sure it is valid and parsable. If not parsable, redo the extraction.
            3. Check there is no comments or '//' or explainations in the output JSON, only the JSON array
            4. The JSON array should not exceed 16000 characters, if too long, remove less important relationships.
            """
            content: str = dspy.InputField(desc="Input text to extract relationships from")
            relations: str = dspy.OutputField(desc="JSON array of relationships with subject, relation, object")
        
        # Create the edge predictor
        edge_predictor = dspy.Predict(EdgeExtraction)
        return edge_predictor

    async def extract_dialogue_relations(
        self,
        text: str
    ) -> List[Tuple[str, str, str, List[int]]]:
        """Extract relations from dialogue text using DSPy-optimized method with coreference resolution"""
        retry = 2
        cache = True
        while retry >= 0:
            lm = dspy.LM(
                model="gpt-4.1-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.2,
                max_tokens=16000,
                cache=cache,
            )

            # Extract relations
            with dspy.settings.context(lm=lm):
                async with self._extract_sem:
                    result = await self._edge_extractor.acall(
                        content=text,
                    )
            try:
                for i in range(len(result.relations)):
                    if "//" in result.relations[i]:
                        print(f"WARNING: Found '//' in relation extraction result line: {result.relations[i]}")
                        # Remove comments in any line
                        index = result.relations[i].find("//")
                        result.relations[i] = result.relations[i][:index]
                if result.relations[-1] != "]":
                    # Remove from last "}," and append "]"
                    index = result.relations.rfind("},")
                    result.relations = result.relations[:index + 1] + "]"
                    print(f"WARNING: Fixed truncated JSON array in RE from Text:\n{text}\nResult:\n{result.relations}")

                # Parse JSON response
                relations = json.loads(result.relations)
                triples = []
                seen = set()
                for rel in relations:
                    if all(key in rel for key in ['subject', 'relation', 'object']):
                        triple_text = f"{rel['subject']} {rel['relation']} {rel['object']}"
                        # Dedup result before returning
                        if triple_text in seen:
                            continue
                        triples.append((rel['subject'], rel['relation'], rel['object']))
                        seen.add(triple_text)
                return triples
            except Exception as e:
                print(f"Warning: Failed to parse relations from text:({text}), relation result({result.relations}): {str(e)}")
                if retry == 0:
                    raise e
                retry -= 1
                cache = False

    
    async def get_episode_groups(
        self,
        episodes: List[Node],
    ) -> tuple[list[list[Node]], list[Node]]:
        """Group episodes into clusters based on content similarity"""
        episode_content_str = ""
        for i, e in enumerate(episodes):
            episode_content_str += f"[{i}][{e.properties['timestamp']}] {e.properties['content']}\n"
        episode_group_prompt = EPISODE_GROUP_PROMPT.format(episodes=episode_content_str)
        res = await self._model.responses.create(
            model=self._model_name,
            max_output_tokens=4096,
            temperature=0.0,
            input=[{"role": "user", "content": episode_group_prompt}],
        )
        try:
            json_parsable_str = ""
            for line in res.output_text.split("\n"):
                if line == "```json" or line == "```":
                    continue
                json_parsable_str += line + "\n"
            response = json.loads(json_parsable_str)
        except Exception as e:
            print(f"WARNING: Failed to parse episode grouping response JSON: {json_parsable_str}, error: {e}")
            raise e

        # If last two indices forms a group or forms singleton, do not group them and pass to next grouping
        filtered_groups = []
        left_over_episodes = []
        for group in response.get("groups", []):
            add = True
            last_idx = len(episodes) - 1
            second_last_idx = len(episodes) - 2
            if len(group) == 1:
                if last_idx in group or second_last_idx in group:
                    add = False
            elif len(group) == 2:
                if last_idx in group and second_last_idx in group:
                    add = False
            if add:
                filtered_groups.append(group)
            else:
                for idx in group:
                    left_over_episodes.append(episodes[idx])
        
        groups = []
        for group in filtered_groups:
            cur_group = []
            for idx in group:
                if idx < 0 or idx >= len(episodes):
                    print(f"WARNING: episode grouping returned invalid episode index: {idx}, len(episodes)={len(episodes)}")
                    continue
                cur_group.append(episodes[idx])
            groups.append(cur_group)

        return groups, left_over_episodes

    async def get_episode_group_summary(
        self,
        episodes: List[Node],
    ) -> str:
        """Generate a concise summary for a group of related episodes"""
        episode_content_str = ""
        for e in episodes:
            episode_content_str += f"[{e.properties['timestamp']}] {e.properties['content']}\n"
        episode_summary_prompt = EPISODE_GROUP_SUMMARY_PROMPT.format(episodes=episode_content_str)
        res = await self._model.responses.create(
            model=self._model_name,
            max_output_tokens=512,
            temperature=0.0,
            input=[{"role": "user", "content": episode_summary_prompt}],
        )
        summary = res.output_text.strip()
        return summary

    async def get_related_episode_cluster(
        self,
        uuid: UUID,
        session_id: str,
        index_search_label: str,
    ) -> set[Node]:
        return await self._store.search_related_nodes(
            node_uuid=uuid,
            allowed_relations={"INCLUDE"},
            find_sources=True,
            find_targets=False,
            limit=None,
            required_labels={"EpisodeCluster"},
            required_properties={"session_id": session_id},
        )
    
    async def batch_ingest_embed(
        self,
        content: list[str],
    ) -> dict[str, list[float]]:
        # Dedup and remove empty strings
        tmp_set = set()
        dedup_content = []
        for c in content:
            if c == "":
                continue
            if c in tmp_set:
                continue
            dedup_content.append(c)
            tmp_set.add(c)

        num_batch = self._embedding_batch_size
        if len(dedup_content) < num_batch:
            num_batch = len(dedup_content) if len(dedup_content) > 0 else 1
        embeddings = []
        for j in range(0, len(dedup_content), num_batch):
            if j + num_batch > len(dedup_content):
                num_batch = len(dedup_content) - j
            batch = [c for c in dedup_content[j:j+num_batch]]
            batch_embeddings = await self._embedder.ingest_embed(batch, max_attempts=3)
            embeddings.extend(batch_embeddings)
        return {
            c: emb for c, emb in zip(dedup_content, embeddings)
        }

    async def generate_edges_nodes(
        self,
        episode_cluster: Node,
        source: str,
        relation: str,
        target: str,
        triple_text: str,
        source_embedding: list[float],
        target_embedding: list[float],
        triple_text_embedding: list[float],
    ) -> tuple[list[Node], list[Edge]]:
        nodes: list[Node] = []
        edges: list[Edge] = []
        episode_cluster_edges: list[Edge] = []
        t = time.perf_counter()
        # 1. Search for existing source and target entity nodes
        # TODO: currently assume source/target entity in same batch of episodes are the same
        if source not in self._entity_node_map:
            # TODO: Optimize by batch processing with asyncio.gather
            search_node = await self._store.search_similar_nodes(
                query_embedding=source_embedding,
                embedding_property_name="name_embedding",
                limit=2,
                required_labels={"Entity"},
                required_properties={"session_id": episode_cluster.properties["session_id"]},
            )
            for n in search_node:
                if source == n.properties["name"]:
                    self._entity_node_map[source] = n.uuid
                    break
        
        if target != "" and target not in self._entity_node_map:
            search_node = await self._store.search_similar_nodes(
                query_embedding=target_embedding,
                embedding_property_name="name_embedding",
                limit=2,
                required_labels={"Entity"},
                required_properties={"session_id": episode_cluster.properties["session_id"]},
            )
            for n in search_node:
                if target == n.properties["name"]:
                    self._entity_node_map[target] = n.uuid
                    break
        self._perf_entity_search_time += time.perf_counter() - t
        
        # 2. Create source and target entity nodes if not exist
        if source not in self._entity_node_map:
            node = Node(
                uuid=uuid4(),
                labels={"Entity"},
                properties={
                    "name": source,
                    "session_id": episode_cluster.properties["session_id"],
                    "name_embedding": source_embedding,
                },
            )
            nodes.append(node)
            self._entity_node_map[source] = node.uuid
            # print(f"Created source node: {source} with UUID {source_node.uuid}")
        
        if target != "" and target not in self._entity_node_map:
            node = Node(
                uuid=uuid4(),
                labels={"Entity"},
                properties={
                    "name": target,
                    "session_id": episode_cluster.properties["session_id"],
                    "name_embedding": target_embedding,
                },
            )
            nodes.append(node)
            self._entity_node_map[target] = node.uuid
            # print(f"Created target node: {target} with UUID {target_node.uuid}")

        # 3. Create edge from source to target entity. This edge does not contain any data.
        #    This edge ensures related episode clusters can be connected via entity nodes.
        edge_id = source + "-" + target
        if target != "" and edge_id not in self._entity_edge_map:
            edge = Edge(
                uuid=uuid4(),
                source_uuid=self._entity_node_map[source],
                target_uuid=self._entity_node_map[target],
                relation="RELATED_TO",
                properties={
                    "session_id": episode_cluster.properties["session_id"],
                },
            )
            edges.append(edge)
            self._entity_edge_map[edge_id] = edge.uuid

        # 4. Create edge from episode cluster to source entity
        e_to_source_id = str(episode_cluster.uuid) + "-" + triple_text
        if e_to_source_id not in self._episode_cluster_edge_map:
            # Add edge from episode cluster to source node
            edges.append(
                Edge(
                    uuid=uuid4(),
                    source_uuid=episode_cluster.uuid,
                    target_uuid=self._entity_node_map[source],
                    relation="HAS_RELATION",
                    # No 'timestamp' field because we don't know which exact episode(s) the relation comes from
                    properties={
                        "triple_text": triple_text,
                        "triple_text_embedding": triple_text_embedding,
                        "session_id": episode_cluster.properties["session_id"],
                    },
                )
            )
            self._episode_cluster_edge_map[e_to_source_id] = True
            self._relation_edge_created += 1

        return nodes, edges
    
    # Have to be locked to avoid duplicate edge/node because the local caches are not async safe.
    @async_locked
    async def add_episode_bulk(self,
        episodes: List[Node],
        flush: bool = False,
    ) -> None:
        if len(episodes) == 0:
            return
        
        # All episodes must belong to the same session
        session_id = episodes[0].properties["session_id"]
        for e in episodes:
            if e.properties["session_id"] != session_id:
                raise ValueError(f"All episodes must belong to the same session for bult adding, found episode with session_id {e.properties['session_id']} instead of {session_id}")
        
        cluster_nodes: list[Node] = []
        cluster_edges: list[Edge] = []
        episode_groups = []

        # 1. Chunking episodes into batches and generating episode clusters
        e_queue = deque(episodes)
        while len(e_queue) > 0:
            # i). Fill episode_batch to 10
            if len(self._episode_batch) < 10:
                self._episode_batch.append(e_queue.popleft())
                # Leftover episodes less than 10, wait for next bulk add if not flushing
                if len(self._episode_batch) < 10:
                    if flush and len(e_queue) == 0:
                        # continue to process leftover episodes
                        pass
                    # Otherwise, continue to accumulate episodes until reached batch size
                    continue
            
            # ii). Use LLM to generate episode cluster that groups epsisodes
            t = time.perf_counter()
            groups, left_over_episodes = await self.get_episode_groups(self._episode_batch)
            self._perf_episode_grouping_time += time.perf_counter() - t
            
            # Update episode_batch with leftover episodes
            self._episode_batch = left_over_episodes

            # iii). Generate summary and create episode cluster nodes
            for group in groups:
                episode_groups.append(group)
                t = time.perf_counter()
                summary = await self.get_episode_group_summary(group)
                self._perf_episode_summary_time += time.perf_counter() - t
                cluster_node = Node(
                    uuid=uuid4(),
                    labels={"EpisodeCluster"},
                    properties={
                        "summary": summary,
                        "session_id": group[0].properties["session_id"],
                    },
                )
                cluster_nodes.append(cluster_node)
                self._episode_cluster_node_created += 1
                # Create edges from cluster node to episodes
                for episode in group:
                    edge = Edge(
                        uuid=uuid4(),
                        source_uuid=cluster_node.uuid,
                        target_uuid=episode.uuid,
                        relation="INCLUDES",
                        properties={
                            "session_id": episode.properties["session_id"],
                            "timestamp": episode.properties["timestamp"],
                        },
                    )
                    cluster_edges.append(edge)
        
        if len(episode_groups) == 0:
            return

        if len(episode_groups) != len(cluster_nodes):
            raise ValueError(f"Episode groups length {len(episode_groups)} does not match cluster nodes length {len(cluster_nodes)}")
            
        # 2. Batch extracting relations from each group of episodes
        relation_extraction_tasks = []
        for group in episode_groups:
            group_content = ""
            for e in group:
                group_content += f"[{e.properties['timestamp']}] {e.properties['content']}\n"
            relation_extraction_tasks.append(self.extract_dialogue_relations(group_content))
        
        t = time.perf_counter()
        relations_list = await asyncio.gather(*relation_extraction_tasks)
        self._perf_relation_extraction_time += time.perf_counter() - t

        if len(relations_list) != len(cluster_nodes):
            raise ValueError(f"Relations list length {len(relations_list)} does not match cluster nodes length {len(cluster_nodes)}")

        # 3. Get embeddings for triples, sources, targes, and episode cluster summarys
        sources = []
        targets = []
        triple_texts = []
        cluster_summaries = []
        for relations in relations_list:
            for source, relation, target in relations:
                triple_text = f"{source} {relation} {target}" if target != "" else f"{source} {relation}"
                sources.append(source)
                targets.append(target)
                triple_texts.append(triple_text)
        
        for n in cluster_nodes:
            cluster_summaries.append(n.properties["summary"])

        t = time.perf_counter()
        sources_embedding = await self.batch_ingest_embed(sources)
        targets_embedding = await self.batch_ingest_embed(targets)
        triple_texts_embedding = await self.batch_ingest_embed(triple_texts)
        cluster_summaries_embedding = await self.batch_ingest_embed(cluster_summaries)
        self._perf_embedding_time += time.perf_counter() - t

        # 4. Generate entity edges and nodes, assign embeddings
        nodes = []
        edges = []
        for relations, episode_cluster in zip(relations_list, cluster_nodes):
            # Assign summary embedding to episode cluster
            episode_cluster.properties["summary_embedding"] = cluster_summaries_embedding[episode_cluster.properties["summary"]]
            for source, relation, target in relations:
                triple_text = f"{source} {relation} {target}" if target != "" else f"{source} {relation}"
                if triple_text in self._processed_triple_texts:
                    continue
                self._processed_triple_texts.add(triple_text)

                n_res, e_res = await self.generate_edges_nodes(
                    episode_cluster=episode_cluster,
                    source=source,
                    relation=relation,
                    target=target,
                    triple_text=triple_text,
                    source_embedding=sources_embedding[source],
                    target_embedding=targets_embedding[target] if target != "" else [],
                    triple_text_embedding=triple_texts_embedding[triple_text],
                )
                nodes.extend(n_res)
                edges.extend(e_res)
        
        # 5. Bulk add all nodes and edges to graph store
        t = time.perf_counter()
        await self._store.add_nodes(episodes)
        await self._store.add_nodes(cluster_nodes)
        await self._store.add_nodes(nodes)
        self._entity_node_created += len(nodes)
        self._episode_cluster_node_created += len(cluster_nodes)
        self._perf_node_creation_time = time.perf_counter() - t

        t = time.perf_counter()
        await self._store.add_edges(cluster_edges)
        await self._store.add_edges(edges)
        self._perf_edge_creation_time = time.perf_counter() - t
    
    def print_ingest_perf_matrix(self):
        print(f"Ingestion Performance Matrics:")
        print(f"  Episode Grouping Time: {self._perf_episode_grouping_time:.2f} seconds")
        print(f"  Episode Summary Time: {self._perf_episode_summary_time:.2f} seconds")
        print(f"  Entity Search Time: {self._perf_entity_search_time:.2f} seconds")
        print(f"  Relation Extraction Time: {self._perf_relation_extraction_time:.2f} seconds")
        print(f"  Embedding Time: {self._perf_embedding_time:.2f} seconds")
        print(f"  Node Creation Time: {self._perf_node_creation_time:.2f} seconds")
        print(f"  Edge Creation Time: {self._perf_edge_creation_time:.2f} seconds")
        print(f"  Entity Nodes Created: {self._entity_node_created}")
        print(f"  Relation Edges Created: {self._relation_edge_created}")
        print(f"  Episode Cluster Nodes Created: {self._episode_cluster_node_created}")

    async def cohere_rerank(
        self,
        items: list[Node] | list[Edge],
        score_threshold: float,
        query: str,
        limit: int | None,
    ) -> list[tuple[Node | Edge, float]]:
        if len(items) == 0:
            return []
        content_list = []
        if isinstance(items[0], Node):
            for e in items:
                if 'content' in e.properties:
                    content_list.append(e.properties['content'])
                elif 'summary' in e.properties:
                    content_list.append(e.properties['summary'])
        elif isinstance(items[0], Edge):
            for e in items:
                content_list.append(e.properties['triple_text'])
        else:
            raise Exception(f"Unknown item type for reranking: {type(items)}")

        num_max = 1000
        processed = 0
        scores = []
        while processed < len(content_list):
            batch_contents = content_list[processed:processed+num_max]
            success = False
            max_retry = 60
            batch_scores = []
            while not success:
                try:
                    batch_scores = await self._reranker.score(query, batch_contents)
                    success = True
                except Exception as e:
                    max_retry -= 1
                    if max_retry == 0:
                        print(f"ERROR: Reranker failed after maximum retries.")
                        raise e
                    if "ThrottlingException" in str(e):
                        print(f"WARNING: Reranker throttling exception, retrying after 60 second...")
                        time.sleep(60)
                    else:
                        raise e
            scores.extend(batch_scores)
            processed += len(batch_contents)
        
        
        scored = sorted(
            zip(items, scores),
            key=lambda x: x[1],   # sort by score
            reverse=True          # highest score first
        )

        result = []
        for e, s in scored:
            if s < score_threshold and limit is not None and len(result) >= limit:
                break
            if limit is not None and len(result) >= limit:
                break
            result.append((e, s))
        
        return result
    
    async def check_sufficiency(
        self,
        query: str,
        episodes: list[Node],
    ) -> tuple[dict[str, Any], list[Node], int, int]:
        episode_content = ""
        for idx, e in enumerate(episodes):
            episode_content += f"[{idx}][{e.properties['timestamp']}] {e.properties['content']}\n"
        
        sufficient_check_prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query,
            retrieved_episodes=episode_content,
        )
        res = await self._model.responses.create(
            model=self._model_name,
            max_output_tokens=4096,
            temperature=0.0,
            # reasoning={"effort": "none"},
            input=[{"role": "user", "content": sufficient_check_prompt}],
        )
        json_parsable_str = ""
        try:
            for line in res.output_text.split("\n"):
                if line == "```json" or line == "```":
                    continue
                if line.strip().startswith("\"reasoning\":"):
                    start_pos = line.find(":")
                    # Find first and last quote
                    first_quote = line.find("\"", start_pos)
                    last_quote = line.rfind("\"")
                    # For any extra quote in between without epsace, replace with single quote
                    # Find quotes in between
                    for i in range(first_quote+1, last_quote):
                        if line[i] == "\"":
                            # Check if escaped
                            if i > 0 and line[i-1] == "\\":
                                continue
                            # Add the escape char
                            line = line[:i] + "\\" + line[i:]
                            # Move i and last_quote forward by 1
                            i += 1
                            last_quote += 1
                json_parsable_str += line + "\n"
            response = json.loads(json_parsable_str)
        except Exception as e:
            print(f"WARNING: Failed to parse sufficiency check response JSON: {res.output_text}\nFinal string used: {json_parsable_str} error: {e}")
            response = {"is_sufficient": False}

        res_episodes = []
        for idx in response.get("indices", []):
            if idx < 0 or idx >= len(episodes):
                print(f"WARNING: sufficiency check returned invalid episode index: {idx}, len(episodes)={len(episodes)}")
                continue
            res_episodes.append(episodes[idx])

        return response, res_episodes, res.usage.input_tokens, res.usage.output_tokens

    async def check_sufficiency_batch(
        self,
        episodes: list[Node],
        possible_relevant_episodes: Collection[Node],
        query: str,
    ) -> tuple[bool, list[Node], set[Node], int, int]:
        input_tokens = 0
        output_tokens = 0
        episode_batch = []
        possible_relevant = set(possible_relevant_episodes)
        for e in episodes:
            if e in possible_relevant:
                continue
            episode_batch.append(e)
            if len(episode_batch) >= 10 or e == episodes[-1]:
                res, suff_episodes, it, ot = await self.check_sufficiency(query, episode_batch)
                input_tokens += it
                output_tokens += ot
                if res['is_sufficient']:
                    sorted_episodes = sorted(
                        suff_episodes,
                        key=lambda e: (e.properties.get('timestamp') is None,
                                    e.properties.get('timestamp'))
                    )
                    reasoning_str = "Inputs:\n"
                    for idx, e in enumerate(episode_batch):
                        reasoning_str += f"[{idx}][{e.properties['timestamp']}] {e.properties['content']}\n"
                    reasoning_str += f"Reasoning: {res['reasoning']}\n"
                    return True, sorted_episodes, possible_relevant, reasoning_str, input_tokens, output_tokens
                episode_batch =[]
                possible_relevant.update(suff_episodes)
        return False, [], possible_relevant, "", input_tokens, output_tokens
    
    async def relation_and_summary_search(
        self,
        query: str,
        session_id: str,
        limit: int = 10,
        perf_matrix: dict[str, Any] = {},
    ):
        # 1. Similarity and fulltext search the query on relation edges
        t = time.perf_counter()
        q_embedding = (await self._embedder.search_embed([query], max_attempts=3))[0]
        perf_matrix["embedding_time"] += time.perf_counter() - t

        t = time.perf_counter()
        edges, res_ec_nodes = await self._store.search_similar_edges(
            query_text=query,
            query_embedding=q_embedding,
            embedding_property_name="triple_text_embedding",
            limit=max(5, limit * 3),
            allowed_relations={"HAS_RELATION"},
            required_properties={"session_id": session_id},
        )
        perf_matrix["edge_search_time"] += time.perf_counter() - t

        ec_node_map = {
            n.uuid: n for n in res_ec_nodes
        }
        # 2. Rerank and get top edges
        t = time.perf_counter()
        cohere_res = await self.cohere_rerank(edges, score_threshold=0.0, query=query, limit=limit)
        perf_matrix["rerank_time"] += time.perf_counter() - t

        added_ec_uuids = set()
        edge_search_ec_nodes = []
        for e, _ in cohere_res:
            # Source node of relation edge is always the episode cluster
            if e.source_uuid not in added_ec_uuids:
                edge_search_ec_nodes.append(ec_node_map[e.source_uuid])
                added_ec_uuids.add(e.source_uuid)

        # TODO: Extracr entities from query and do entity node search?

        # # 3. Similarity and fulltext search the query on episode clusters 
        # t = time.perf_counter()
        # res_ec_nodes = await self._store.search_similar_nodes(
        #     query_embedding=q_embedding,
        #     embedding_property_name="summary_embedding",
        #     limit=max(5, limit),
        #     required_labels={"EpisodeCluster"},
        #     required_properties={"session_id": session_id},
        # )
        # perf_matrix["episode_cluster_node_search_time"] += time.perf_counter() - t

        # # 4. Rerank the directly searched episode clusters, then RRF rerank with edge searched episode clusters
        # t = time.perf_counter()
        # cohere_res = await self.cohere_rerank(res_ec_nodes, score_threshold=0.0, query=query, limit=limit)
        # perf_matrix["rerank_time"] += time.perf_counter() - t

        # vec_search_ec = [e for e, _ in cohere_res]

        # # vec_search_ec = [e for e, _ in cohere_res]
        # fused = rrf([edge_search_ec_nodes, vec_search_ec], k=50)

        # # RRF returns unique items, return the result directly
        # result_episode_clusters = [e for e, _ in fused]
        return edge_search_ec_nodes
    
    async def get_relation_edges_and_episode_clusters(
        self,
        entity_node_uuids: list[UUID],
        session_id: str,
    ) -> tuple[list[Node], list[Edge]]:
        return await self._store.search_related_nodes_edges_batch(
            node_uuids=entity_node_uuids,
            index_search_label=":Entity",
            allowed_relations={"HAS_RELATION"},
            find_sources=True,
            find_targets=False,
            limit=None,
            required_labels={"EpisodeCluster"},
            required_properties={"session_id": session_id},
        )

    async def get_included_episodes_in_order_from_clusters(
        self,
        episode_cluster_uuids: list[UUID],
        session_id: str,
    ) -> list[Node]:
        related_episodes = []
        for uuid in episode_cluster_uuids:
            related_episodes.extend(
                await self._store.search_related_nodes(
                    node_uuid=uuid,
                    index_search_label=":EpisodeCluster",
                    allowed_relations={"INCLUDES"},
                    find_sources=False,
                    find_targets=True,
                    limit=None,
                    required_labels={"Episode"},
                    required_properties={"session_id": session_id},
                )
            )
        return related_episodes
    
    def init_perf_matrix(self) -> dict[str, Any]:
        return {
            "query": "",
            "msg": "",
            "embedding_time": 0.0,
            "entity_extraction_time": 0.0,
            "edge_search_time": 0.0,
            "entity_node_search_time": 0.0,
            "episode_cluster_node_search_time": 0.0,
            "related_node_search_time": 0.0,
            "sufficiency_check_time": 0.0,
            "rerank_time": 0.0,
            "num_sufficiency_checks": 0,
            "num_sufficiency_check_episodes": 0,
            "num_bfs_iteration": 0,
            "num_llm_input_tokens": 0,
            "num_llm_output_tokens": 0,
            "total_time": 0.0,
            "total_return_episodes": 0,
        }

    def print_search_perf_matrix(
        self,
        perf_matrix: dict[str, Any],
    ):
        print(f"Search Performance Matrics:")
        for key, value in perf_matrix.items():
            if type(value) == float:
                value = f"{value:.2f}"
            print(f"  {key}: {value}")
        print("=================================================================\n")

    async def search(
        self,
        query: str,
        possible_episodes: list[Node],
        session_id: int,
        limit: int = 10
    ) -> tuple[list[str], dict[str, Any], bool]:
        entity_node_local_cache = set()
        episode_cluster_local_cache = set()

        possible_relevant_episodes = set(possible_episodes)

        # Initialize performance matrix for current search
        perf_matrix = self.init_perf_matrix()
        perf_matrix["query"] = query
        
        search_start = time.perf_counter()
        
        # 1. Get initial episode clusters by searching on relation triples and summaries
        episode_clusters = await self.relation_and_summary_search(query, session_id, limit, perf_matrix)
        if len(episode_clusters) == 0:
            perf_matrix["msg"] += "No related episodes clusters found from initial search.\n"
            perf_matrix["total_return_episodes"] = 0
            perf_matrix["total_time"] = time.perf_counter() - search_start
            self.print_search_perf_matrix(perf_matrix)
            return [], perf_matrix, False
        
        episode_cluster_uuids = [e.uuid for e in episode_clusters]
        episode_cluster_local_cache.update(episode_cluster_uuids)
        
        # 2. Get related episodes from the episode clusters, order by episode cluster orders.
        #    There should be no duplicates since each episode only belongs to one episode cluster.
        t = time.perf_counter()
        related_episodes = await self.get_included_episodes_in_order_from_clusters(
            episode_cluster_uuids,
            session_id,
        )
        perf_matrix["related_node_search_time"] += time.perf_counter() - t

        perf_matrix["msg"] += f"Found {len(episode_clusters)} initial related episode clusters and expands to {len(related_episodes)} episodes.\n"
        
        # DEBUG: check duplication
        if len(related_episodes) != len(set([e.uuid for e in related_episodes])):
            raise ValueError(f"Related episodes from episode clusters contain duplicates, total {len(related_episodes)} vs unique {len(set([e.uuid for e in related_episodes]))}")

        # 3. Rerank the related episodes
        t = time.perf_counter()
        cohere_res = await self.cohere_rerank(related_episodes, score_threshold=0.0, query=query, limit=None)
        perf_matrix["rerank_time"] += time.perf_counter() - t
        related_episodes = [e for e, _ in cohere_res]

        # 3.Check sufficiency, on all related episodes, using 'limit' as batch size
        t = time.perf_counter()
        is_sufficient, sorted_suff_episodes, possible_relevant, _, itoken, otoken = await self.check_sufficiency_batch(
            related_episodes,
            possible_relevant_episodes,
            query,
        )
        perf_matrix["num_llm_input_tokens"] += itoken
        perf_matrix["num_llm_output_tokens"] += otoken
        perf_matrix["sufficiency_check_time"] += time.perf_counter() - t
        perf_matrix["num_sufficiency_checks"] += len(related_episodes) // limit + (1 if len(related_episodes) % limit != 0 else 0)
        perf_matrix["num_sufficiency_check_episodes"] += len(related_episodes)
        if is_sufficient:
            if len(sorted_suff_episodes) > limit:
                perf_matrix["msg"] += f"Number of sufficient episodes exceed limit: {limit}/{len(suff_episodes)}. Result truncated.\n"
                # Rerank sufficient episodes to get top 'limit' episodes
                t = time.perf_counter()
                cohere_res = await self.cohere_rerank(sorted_suff_episodes, score_threshold=0.0, query=query, limit=limit)
                perf_matrix["rerank_time"] += time.perf_counter() - t
                sorted_suff_episodes = sorted(
                    list([e for e, _ in cohere_res]),
                    key=lambda e: (e.properties.get('timestamp') is None,
                                e.properties.get('timestamp'))
                )
            perf_matrix["msg"] += f"Sufficient from initial retrieved episodes\n"
            perf_matrix["total_return_episodes"] = len(sorted_suff_episodes)
            perf_matrix["total_time"] = time.perf_counter() - search_start
            self.print_search_perf_matrix(perf_matrix)
            return sorted_suff_episodes, perf_matrix, is_sufficient

        possible_relevant_episodes.update(possible_relevant)

        # 4. Get total number of episode cluster in current session
        ec_all = await self._store.search_matching_nodes(
            limit=None,
            required_labels={"EpisodeCluster"},
            required_properties={"session_id": session_id},
        )

        # 5. Start BFS, expanding from initial episode clusters
        num_all_ec = len(ec_all)
        episode_cluster_queue = deque(episode_clusters)
        while len(episode_cluster_queue) > 0:
            if len(episode_cluster_local_cache) >= num_all_ec:
                if (len(episode_cluster_local_cache) > num_all_ec):
                    perf_matrix["msg"] += f"WARNING: total related episode cluster {len(episode_cluster_local_cache)} greater than session total {num_all_ec}\n"
                break

            perf_matrix["num_bfs_iteration"] += 1
            episode_cluster = episode_cluster_queue.popleft()

            # perf_matrix["msg"] += f"  DEBUG: curr iter: {perf_matrix["num_bfs_iteration"]}, queue size: {len(episode_cluster_queue)}, possible relevant episodes size: {len(possible_relevant_episodes)}\n"

            # i). Get related entity nodes
            t = time.perf_counter()
            res = await self._store.search_related_nodes(
                node_uuid=episode_cluster.uuid,
                index_search_label=":EpisodeCluster",
                allowed_relations={"HAS_RELATION"},
                find_sources=False,
                find_targets=True,
                limit=None,
                required_labels={"Entity"},
                required_properties={"session_id": session_id},
            )
            perf_matrix["related_node_search_time"] += time.perf_counter() - t
            entity_nodes_uuids = []
            for n in res:
                if n.uuid in entity_node_local_cache:
                    continue
                entity_nodes_uuids.append(n.uuid)
                entity_node_local_cache.add(n.uuid)

            # perf_matrix["msg"] += f"  DEBUG: Found {len(entity_nodes_uuids)} new source entity nodes\n"
            if len(entity_nodes_uuids) == 0:
                continue
            
            # ii). Get all connected episode clusters and relation edges from the entity nodes above.
            t = time.perf_counter()
            res_ec, res_edges = await self.get_relation_edges_and_episode_clusters(
                entity_nodes_uuids,
                session_id,
            )
            perf_matrix["related_node_search_time"] += time.perf_counter() - t
            # Dedup search result(do not move this inside get_relation_edges_and_episode_clusters() to
            # be more clear). Notice need to dedup relation edges first. The order matters because
            # we append nodes to episode_cluster_local_cache when deduping new episode clusters.
            direct_relation_edges = []
            for re in res_edges:
                if re.source_uuid in episode_cluster_local_cache:
                    continue
                direct_relation_edges.append(re)

            direct_episode_clusters = []
            for e in res_ec:
                if e.uuid in episode_cluster_local_cache:
                    continue
                direct_episode_clusters.append(e)
                episode_cluster_local_cache.add(e.uuid)

            # perf_matrix["msg"] += f"  DEBUG: Get {len(direct_episode_clusters)} direct episode clusters and {len(direct_relation_edges)} direct relation edges\n"
            
            # iii). Get connected Entity node via RELATED_TO edges from the entity nodes above
            t = time.perf_counter()
            res_nodes, _ = await self._store.search_related_nodes_edges_batch(
                node_uuids=entity_nodes_uuids,
                index_search_label=":Entity",
                allowed_relations={"RELATED_TO"},
                find_sources=False,
                find_targets=True,
                limit=None,
                required_labels={"Entity"},
                required_properties={"session_id": session_id},
            )
            perf_matrix["related_node_search_time"] += time.perf_counter() - t

            indirect_entity_nodes_uuids = []
            for n in res_nodes:
                if n.uuid in entity_node_local_cache:
                    continue
                indirect_entity_nodes_uuids.append(n.uuid)
                entity_node_local_cache.add(n.uuid)
            
            # iv). Get the connected episode clusters and relation edges from the target entity nodes above.
            #.     These episode clusters are the '1st level of indirectly related' ecs of the episode
            #      cluster of current BFS iteration.
            t = time.perf_counter()
            res_ec, res_edges = await self.get_relation_edges_and_episode_clusters(
                indirect_entity_nodes_uuids,
                session_id,
            )
            perf_matrix["related_node_search_time"] += time.perf_counter() - t
            # Dedup search result, see comments above for details
            indirect_relation_edges = []
            for re in res_edges:
                if re.source_uuid in episode_cluster_local_cache:
                    continue
                indirect_relation_edges.append(re)

            indirect_episode_clusters = []
            for e in res_ec:
                if e.uuid in episode_cluster_local_cache:
                    continue
                indirect_episode_clusters.append(e)
                episode_cluster_local_cache.add(e.uuid)

            # perf_matrix["msg"] += f"  DEBUG: Get {len(indirect_episode_clusters)} indirect episode clusters and {len(indirect_relation_edges)} indirect relation edges\n"

            # DEBUG: check relation edge duplication
            if len(direct_relation_edges) + len(indirect_relation_edges) != len(set([e.uuid for e in direct_relation_edges + indirect_relation_edges])):
                raise ValueError(f"Relation edges from unique episode cluster to entity nodes contain duplicates, total {len(direct_relation_edges) + len(indirect_relation_edges)} vs unique {len(set([e.uuid for e in direct_relation_edges + indirect_relation_edges]))}")
            
            new_ec_map = {
                e.uuid: e for e in direct_episode_clusters + indirect_episode_clusters
            }

            if len(new_ec_map) == 0:
                continue
            
            # # Rrerank relations to get ranked episode clusters
            # t = time.perf_counter()
            # cohere_res = await self.cohere_rerank(direct_relation_edges + indirect_relation_edges, score_threshold=0.0, query=query, limit=None)
            # perf_matrix["rerank_time"] += time.perf_counter() - t

            # relation_ranked_episode_clusters = []
            # added_ec_uuids = set()
            # for e, _ in cohere_res:
            #     # Source node of relation edge is always the episode cluster
            #     if e.source_uuid in added_ec_uuids:
            #         continue
            #     added_ec_uuids.add(e.source_uuid)
            #     relation_ranked_episode_clusters.append(new_ec_map[e.source_uuid])

            # # iv). Rerank new episode clusters based on summary
            # t = time.perf_counter()
            # cohere_res = await self.cohere_rerank(direct_episode_clusters + indirect_episode_clusters, 0, query, limit=None)
            # perf_matrix["rerank_time"] += time.perf_counter() - t
            # summary_reranked_episode_clusters = [e for e, _ in cohere_res]

            # # v). RRF fuse the two reranked episode cluster lists
            # fused = rrf([relation_ranked_episode_clusters, summary_reranked_episode_clusters], k=50)

            # # RRF returns unique items
            # episode_cluster_uuids = [e.uuid for e, _ in fused]

            # vi). Get related episodes from the fused episode clusters, order by episode cluster orders.
            t = time.perf_counter()
            related_episodes = await self.get_included_episodes_in_order_from_clusters(
                [uuid for uuid in new_ec_map.keys()],
                session_id,
            )
            # perf_matrix["msg"] += f" DEBUG: BFS iteration {perf_matrix['num_bfs_iteration']}: Retrieved {len(related_episodes)} related episodes from {len(new_ec_map)} new episode clusters.\n"
            perf_matrix["related_node_search_time"] += time.perf_counter() - t

            # 3. Rerank the related episodes
            t = time.perf_counter()
            cohere_res = await self.cohere_rerank(related_episodes, score_threshold=0.0, query=query, limit=None)
            perf_matrix["rerank_time"] += time.perf_counter() - t
            related_episodes = [e for e, _ in cohere_res]

            # perf_matrix["msg"] += f"  DEBUG: BFS iteration {perf_matrix['num_bfs_iteration']}: Check sufficiency on {len(related_episodes)} new related episodes from fused episode clusters\n"
            
            # vii). Check sufficiency on all new related episodes
            t = time.perf_counter()
            is_sufficient, sorted_suff_episodes, possible_relevant, _, itoken, otoken = await self.check_sufficiency_batch(
                related_episodes,
                possible_relevant_episodes,
                query,
            )
            perf_matrix["num_llm_input_tokens"] += itoken
            perf_matrix["num_llm_output_tokens"] += otoken
            perf_matrix["sufficiency_check_time"] += time.perf_counter() - t
            perf_matrix["num_sufficiency_checks"] += len(related_episodes) // limit + (1 if len(related_episodes) % limit != 0 else 0)
            perf_matrix["num_sufficiency_check_episodes"] += len(related_episodes)
            if is_sufficient:
                if len(sorted_suff_episodes) > limit:
                    perf_matrix["msg"] += f"Number of sufficient episodes exceed limit from BFS retrival: {limit}/{len(suff_episodes)}. Result truncated.\n"
                    # Rerank sufficient episodes to get top 'limit' episodes
                    t = time.perf_counter()
                    cohere_res = await self.cohere_rerank(sorted_suff_episodes, score_threshold=0.0, query=query, limit=limit)
                    perf_matrix["rerank_time"] += time.perf_counter() - t
                    sorted_suff_episodes = sorted(
                        list([e for e, _ in cohere_res]),
                        key=lambda e: (e.properties.get('timestamp') is None,
                                    e.properties.get('timestamp'))
                    )
                perf_matrix["msg"] += f"Sufficient from BFS retrieved episodes.\n"
                perf_matrix["total_return_episodes"] = len(sorted_suff_episodes)
                perf_matrix["total_time"] = time.perf_counter() - search_start
                self.print_search_perf_matrix(perf_matrix)
                return sorted_suff_episodes, perf_matrix, is_sufficient
            possible_relevant_episodes.update(possible_relevant)
            
            # viii). Not sufficient, add to queue
            episode_cluster_queue.extend(direct_episode_clusters + indirect_episode_clusters)
        
        # 5. BFS finished but still not sufficient, rerank possible_relevant_episodes if there are more than limit, otherwise return all
        if len(possible_relevant_episodes) > limit:
            print(f"BFS finished but not sufficient, rerank possible relevant episodes from {len(possible_relevant_episodes)} to {limit}.")
            t = time.perf_counter()
            cohere_res = await self.cohere_rerank(list(possible_relevant_episodes), 0, query, limit=limit)
            possible_relevant_episodes = set([e for e, _ in cohere_res])
        
        sorted_episodes = sorted(
            list(possible_relevant_episodes),
            key=lambda e: (e.properties.get('timestamp') is None,
                        e.properties.get('timestamp'))
        )
        perf_matrix["msg"] += f"BFS finished but not sufficient, return possible relevant episodes.\n"
        perf_matrix["total_return_episodes"] = len(sorted_episodes)
        perf_matrix["total_time"] = time.perf_counter() - search_start
        self.print_search_perf_matrix(perf_matrix)
        return sorted_episodes, perf_matrix, False
