"""Multi-Hop Question Decomposer - Properly fixed version."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

import spacy
from spacy.tokens import Doc, Token


@dataclass
class DecomposedHop:
    """A single decomposed hop of a multi-hop question."""

    question: str
    entity: str | None = None
    relation: str | None = None
    answer_type: str | None = None
    hop_order: int = 0


@dataclass
class DecompositionResult:
    """Result of decomposing a question into hops."""

    original_question: str
    first_hop: str
    second_hop_template: str
    hops: list = field(default_factory=list)
    is_multi_hop: bool = True


# Verbs used to delimit the entity after "of".
_VERBS = (
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "won",
    "find",
    "found",
    "work",
    "earned",
    "born",
    "die",
)

# Relative-clause verbs that should NOT be used as entity boundaries.
_THAT_CLAUSE_VERBS = re.compile(
    r"\bthat\s+(inspired|won|lost|created|made|directed|wrote)\b", re.IGNORECASE
)

# Nouns treated as property words, not extracted as the possessive relation.
_SKIP_NOUNS = frozenset(
    {
        "birthday",
        "birth",
        "death",
        "anniversary",
        "name",
        "age",
        "height",
        "weight",
    }
)


class MultiHopDecomposer:
    """Rule-based decomposer for compositional multi-hop questions."""

    WH_PATTERNS: ClassVar[dict[str, str]] = {
        "when": "date/time",
        "where": "location",
        "who": "person",
        "whom": "person",
        "whose": "person",
        "what": "entity",
        "which": "entity",
        "why": "reason",
        "how": "manner",
    }

    # Property expressions: excluded from first_hop, preserved in second_hop.
    PROPERTY_PATTERNS: ClassVar[tuple[str, ...]] = (
        "cause of death",
        "date of death",
        "place of death",
        "cause of birth",
        "date of birth",
        "place of birth",
        "award that",
        "award which",
        "prize that",
        "prize which",
        # In "the X of the Y of" patterns, treat X as a property.
        "population of",
        "capital of",
        "CEO of",
        "coach of",
        "profession of",
        "grandfather of",
        "grandmother of",
        "father of",
        "mother of",
        "brother of",
        "sister of",
        "author of",
        "book that",
        "song that",
        "film that",
        "character of",
        "role of",
    )

    # Verb + preposition phrases: excluded from first_hop, preserved in second_hop.
    VERB_PREP_PATTERNS: ClassVar[tuple[str, ...]] = (
        "work at",
        "work for",
        "work with",
        "is from",
        "are from",
        "was from",
        "were from",
        "study at",
        "study in",
        "studied at",
        "studied in",
        "graduate from",
        "graduated from",
        "live in",
        "live at",
        "born in",
        "born at",
        "study",
        "studied",
    )

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """Load the spaCy pipeline, downloading it on first use."""
        try:
            self.nlp = spacy.load(model)
        except OSError:
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

    def decompose(self, question: str) -> DecompositionResult:
        doc = self.nlp(question)
        _wh_word, answer_type = self._extract_wh_word(question)

        # 1. Try the possessive ("'s") pattern first.
        possessive_result = self._try_possessive(doc, question)
        first_hop, second_hop = possessive_result or (None, None)

        # 2. If no possessive, try the "the X of Y" pattern.
        if first_hop is None:
            xofy_result = self._try_the_x_of_y(question)
            first_hop, second_hop = xofy_result or (None, None)

        # 3. If neither matches, return the original question (failure).
        if first_hop is None or second_hop is None:
            first_hop = question
            second_hop = question
        else:
            # Post-process property expressions and verb + preposition phrases.
            first_hop, second_hop = self._fix_property_and_verb_prep(
                question, first_hop, second_hop
            )

        hops = [
            DecomposedHop(
                question=first_hop,
                entity=self._extract_entity(first_hop),
                relation=self._extract_relation(first_hop),
                hop_order=0,
            ),
            DecomposedHop(
                question=second_hop.replace("[HOP]", first_hop),
                answer_type=answer_type,
                hop_order=1,
            ),
        ]

        return DecompositionResult(
            original_question=question,
            first_hop=first_hop,
            second_hop_template=second_hop,
            hops=hops,
            is_multi_hop=(first_hop != question),
        )

    def _extract_wh_word(self, question: str) -> tuple[str | None, str]:
        q = question.lower().strip()
        for wh, atype in self.WH_PATTERNS.items():
            if q.startswith((wh + " ", wh + "'")):
                return wh, atype
        return None, "unknown"

    def _restore_property_in_second_hop(
        self, question: str, first_hop: str, second_hop: str
    ) -> str:
        """Restore a leading property expression into second_hop.

        If a "the <prop> of" expression precedes first_hop in the question,
        splice it into second_hop: "What is [HOP]?" -> "What is the cause of death of [HOP]?".
        """
        for prop in self.PROPERTY_PATTERNS:
            prop_pattern = rf"\bthe\s+{re.escape(prop)}\s+of\s+"
            prop_match = re.search(prop_pattern, question, re.IGNORECASE)
            if prop_match and prop_match.start() < question.find(first_hop):
                wh_word, _ = self._extract_wh_word(question)
                if wh_word:
                    # "What is [HOP]?" -> "What is the cause of death of [HOP]?"
                    return re.sub(
                        rf"\b{re.escape(wh_word)}\s+(?:is|are|was|were)?\s*\[HOP\]",
                        f"{wh_word} is the {prop} of [HOP]",
                        second_hop,
                        flags=re.IGNORECASE,
                    )
        return second_hop

    def _remove_property_from_first_hop(
        self, question: str, first_hop: str, second_hop: str
    ) -> tuple[str, str]:
        """Remove a property expression from first_hop and add it to second_hop."""
        for prop in self.PROPERTY_PATTERNS:
            prop_pattern = rf"\bthe\s+{re.escape(prop)}\s+of\s+"
            prop_match = re.search(prop_pattern, first_hop, re.IGNORECASE)
            if prop_match:
                first_hop = first_hop.replace(prop_match.group(0), "", 1)
                orig_wh_word, _ = self._extract_wh_word(question)
                if orig_wh_word:
                    # "What is [HOP]?" -> "What is the cause of death of [HOP]?"
                    second_hop = re.sub(
                        rf"\b{re.escape(orig_wh_word)}\s+(?:is|are|was|were)?\s*\[HOP\]",
                        f"{orig_wh_word.capitalize()} is the {prop} of [HOP]",
                        second_hop,
                        flags=re.IGNORECASE,
                    )
                break
        return first_hop, second_hop

    def _fix_verb_prep_phrases(
        self, first_hop: str, second_hop: str
    ) -> tuple[str, str]:
        """Strip trailing verb+preposition phrases from first_hop into second_hop."""
        for verb_prep in self.VERB_PREP_PATTERNS:
            # Word-order fix for "from" phrases in second_hop.
            if verb_prep in ("is from", "are from", "was from", "were from"):
                # "Which country [HOP] is from?" -> "Which country is [HOP] from?"
                pattern = rf"(\S+)\s+\[HOP\]\s+{re.escape(verb_prep)}"
                if re.search(pattern, second_hop, re.IGNORECASE):
                    second_hop = re.sub(
                        pattern,
                        r"\1 " + verb_prep.replace(" from", "") + " [HOP] from",
                        second_hop,
                        flags=re.IGNORECASE,
                    )
            # Remove a trailing verb+preposition from first_hop.
            vp_pattern = rf"\s+{re.escape(verb_prep)}\s*\??$"
            vp_match = re.search(vp_pattern, first_hop, re.IGNORECASE)
            if vp_match:
                first_hop = first_hop[: vp_match.start()].strip()
                if not re.search(
                    rf"\b{re.escape(verb_prep)}\s*\??$", second_hop, re.IGNORECASE
                ):
                    second_hop = second_hop.rstrip("?").rstrip() + " " + verb_prep + "?"
                break

        # Verbs not in VERB_PREP_PATTERNS: "got", "won", "received", "earned".
        for verb in ("got", "won", "received", "earned"):
            verb_pattern = rf"\s+{verb}\s*\??$"
            verb_match = re.search(verb_pattern, first_hop, re.IGNORECASE)
            if verb_match:
                first_hop = first_hop[: verb_match.start()].strip()
                if not re.search(
                    rf"\b{re.escape(verb)}\s*\??$", second_hop, re.IGNORECASE
                ):
                    second_hop = second_hop.rstrip("?").rstrip() + " " + verb + "?"
                break

        return first_hop, second_hop

    def _fix_property_and_verb_prep(
        self, question: str, first_hop: str, second_hop: str
    ) -> tuple[str, str]:
        """Fix property and verb+preposition phrases.

        Remove property expressions and verb+preposition phrases from first_hop
        and add them to second_hop.

        Example: "What is the cause of death of X's father?"
        first_hop: "the cause of death of X's father" -> "X's father"
        second_hop: "What is [HOP]?" -> "What is the cause of death of [HOP]?"
        """
        first_hop, second_hop = self._remove_property_from_first_hop(
            question, first_hop, second_hop
        )
        first_hop, second_hop = self._fix_verb_prep_phrases(first_hop, second_hop)
        return first_hop.strip(), self._clean_template(second_hop)

    def _possessive_relation(self, doc: Doc, token: Token) -> Token | None:
        """First noun to the right of "'s" (skipping property words)."""
        for i in range(token.i + 1, len(doc)):
            t = doc[i]
            if t.pos_ in ("NOUN", "PROPN"):
                if t.text.lower() in _SKIP_NOUNS:
                    # Property words are not extracted as the relation.
                    return None
                return t
            # Stop at a verb or WH word (no relation).
            if t.pos_ in ("VERB", "AUX") or t.text.lower() in self.WH_PATTERNS:
                return None
        return None

    def _possessive_owner_start(self, doc: Doc, token: Token) -> int:
        """Index in doc where the owner (left of "'s") begins."""
        for i in range(token.i - 1, -1, -1):
            t = doc[i]
            # Start after a WH word.
            if t.text.lower() in self.WH_PATTERNS:
                return i + 1
            # Start after an aux verb (did, does, is, etc.).
            if t.pos_ == "AUX":
                return i + 1
            # Start after a ROOT verb.
            if t.pos_ == "VERB" and t.dep_ == "ROOT":
                return i + 1
        return 0

    def _clean_possessive_owner(self, question: str, owner: str) -> str:
        """Strip leading "country" and "the place of X of" prefixes from the owner."""
        wh_word, _ = self._extract_wh_word(question)
        if wh_word:
            country_match = re.search(
                rf"^{re.escape(wh_word)}\s+country\s+(.+)$", owner, re.IGNORECASE
            )
            if country_match:
                return country_match.group(1).strip()
            if owner.lower().startswith("country "):
                return owner[8:].strip()
        # "the place of X of Y's Z": keep only the Y part.
        place_of_match = re.search(
            r"\bthe place of (?:death|birth|origin|residence|burial|interment)\s+of\s+(.+)$",
            owner,
            re.IGNORECASE,
        )
        if place_of_match:
            return place_of_match.group(1).strip()
        return owner

    def _try_possessive(self, doc: Doc, question: str) -> tuple[str, str] | None:
        """Handle the possessive ("'s") pattern.

        Extract the full "'s" relation expression as first_hop.

        Example: "When did John V's father die?"
        -> first_hop: "John V's father"
        -> second_hop: "When did [HOP] die?"
        """
        for token in doc:
            if token.text != "'s":
                continue
            relation = self._possessive_relation(doc, token)
            if not relation:
                continue

            start_idx = self._possessive_owner_start(doc, token)
            owner = question[doc[start_idx].idx : token.idx].strip().rstrip("'").strip()
            owner = self._clean_possessive_owner(question, owner)
            if not owner:
                continue

            first_hop = self._fix_parentheses(f"{owner}'s {relation.text}", question)
            if first_hop not in question:
                continue

            # second_hop: replace the whole first_hop with [HOP], restoring any
            # leading property expression.
            second_hop = question.replace(first_hop, "[HOP]", 1)
            second_hop = self._restore_property_in_second_hop(
                question, first_hop, second_hop
            )
            return first_hop, self._clean_template(second_hop)

        return None

    def _try_who_played(self, question: str) -> tuple[str, str] | None:
        """Handle "Who played the character of X"."""
        who_played_match = re.search(
            r"\b(W|w)ho\s+(played|plays)\s+the\s+(character|role)\s+of\s+(.+)\??",
            question,
            re.IGNORECASE,
        )
        if not who_played_match:
            return None
        entity_part = who_played_match.group(4).strip().rstrip("?")
        first_hop = entity_part.strip()
        second_hop = "Who played the character of [HOP]?"
        if first_hop and first_hop in question:
            return first_hop, second_hop
        return None

    def _try_triple_nested_of(self, question: str) -> tuple[str, str] | None:
        """Handle 3+ level nested "the X of the Y of the Z of" patterns."""
        all_of_matches = list(
            re.finditer(r"\bthe\s+[\w\s]+?\s+of\s+", question, re.IGNORECASE)
        )
        if len(all_of_matches) < 3:
            return None

        second_innermost = all_of_matches[-2]
        innermost_match = all_of_matches[-1]

        # Use the whole "the X of Y" as first_hop (X = innermost_match, Y = the rest).
        first_hop_end = question.find("?")
        if first_hop_end < 0:
            first_hop_end = len(question)
        first_hop = (
            question[innermost_match.start() : first_hop_end].strip().rstrip("?")
        )

        # Include year expressions (in 2018).
        year_match = re.search(r"\s+in\s+\d{4}", first_hop)
        if year_match:
            first_hop = first_hop[: year_match.end()].strip()

        # second_hop: outer relations + [HOP].
        outer_relations = question[: second_innermost.end()].strip()
        wh_word, _ = self._extract_wh_word(outer_relations)
        if wh_word:
            # "Who is the grandfather of the son of " -> "the grandfather of the son of"
            outer_relations = re.sub(
                r"^is\s+",
                "",
                outer_relations[len(wh_word) :].strip(),
                flags=re.IGNORECASE,
            )

        wh_word_orig, _ = self._extract_wh_word(question)
        if wh_word_orig:
            second_hop = f"{wh_word_orig.capitalize()} is {outer_relations} [HOP]?"
        else:
            second_hop = f"What is {outer_relations} [HOP]?"

        if first_hop and first_hop in question:
            return first_hop, self._clean_template(second_hop)
        return None

    def _try_nested_outer_property(self, question: str) -> tuple[str, str] | None:
        """Handle "the X of the Y of" where the outer "the X of" is a property."""
        nested_of_match = re.search(
            r"\b(the\s+(?:population|capital|CEO|coach|profession|grandfather|grandmother|father|mother|brother|sister|author|book)\s+of)\s+(the\s+.+)$",
            question,
            re.IGNORECASE,
        )
        if not nested_of_match:
            return None
        outer_prop = nested_of_match.group(1).strip()  # "the population of"
        inner_part = nested_of_match.group(2).strip()  # "the capital of France"

        # Extract from inner_part up to a question mark.
        inner_end = inner_part.find("?")
        if inner_end < 0:
            inner_end = len(inner_part)
        first_hop = inner_part[:inner_end].strip().rstrip("?")

        wh_word, _ = self._extract_wh_word(question)
        if wh_word:
            second_hop = f"{wh_word.capitalize()} is {outer_prop} [HOP]?"
        else:
            second_hop = f"What is {outer_prop} [HOP]?"

        if first_hop and first_hop in question:
            return first_hop, self._clean_template(second_hop)
        return None

    def _strip_trailing_verb(self, entity: str) -> str:
        """Remove a trailing verb from the entity (double check)."""
        if not entity:
            return entity
        last_word = entity.rsplit(maxsplit=1)[-1].lower().rstrip(".,!?")
        if last_word in _VERBS:
            return " ".join(entity.split()[:-1]).strip()
        return entity

    def _preserve_proper_nouns(self, entity: str) -> str:
        """Best-effort preservation of multi-word proper nouns in the entity.

        Currently a no-op aside from validating spaCy entities; kept for
        symmetry with the upstream decomposer behavior.
        """
        if not entity:
            return entity
        entity_doc = self.nlp(entity)
        for ent in entity_doc.ents:
            if ent.label_ not in ("PERSON", "WORK_OF_ART", "GPE", "ORG"):
                continue
            ent_text = ent.text
            if " " in ent_text and ent_text in entity:
                ent_end_idx = entity.find(ent_text) + len(ent_text)
                remaining = entity[ent_end_idx:].strip()
                if remaining and not re.match(
                    r"\b(in|at|on|of|the|a|an)\b", remaining, re.IGNORECASE
                ):
                    # If not a year or preposition, do not include.
                    pass
        return entity

    def _try_that_relative_clause(self, question: str) -> tuple[str, str] | None:
        """Handle "that the X of" relative clauses."""
        that_match = re.search(
            r"\bthat\s+(the\s+[\w\s]+?\s+of\s+)", question, re.IGNORECASE
        )
        if not that_match:
            return None
        after_that_start = that_match.start(1)
        matches = list(
            re.finditer(
                r"\bthe\s+[\w\s]+?\s+of\s+", question[after_that_start:], re.IGNORECASE
            )
        )
        if not matches:
            return None
        first_match = matches[0]
        actual_end = after_that_start + first_match.end()

        # Extract the entity after "of", up to a verb/question-mark boundary.
        rest = question[actual_end:]
        end_idx = rest.find("?")
        if end_idx < 0:
            end_idx = len(rest)
        if not _THAT_CLAUSE_VERBS.search(rest):
            for verb in _VERBS:
                m = re.search(r"\b" + verb + r"\b", rest, re.IGNORECASE)
                if m and m.start() < end_idx:
                    end_idx = m.start()
        entity = self._strip_trailing_verb(rest[:end_idx].strip())
        entity = self._preserve_proper_nouns(entity)

        first_hop = " ".join((first_match.group().strip() + " " + entity).split())
        first_hop = self._fix_parentheses(first_hop, question)
        if first_hop and first_hop in question:
            second_hop = question.replace(first_hop, "[HOP]", 1)
            return first_hop, self._clean_template(second_hop)
        return None

    def _try_that_verb_clause(self, question: str) -> tuple[str, str] | None:
        """Handle "the X that + verb + Y" relative clauses."""
        that_verb_match = re.search(
            r"\b(the\s+\w+)\s+that\s+(\w+)\s+([\w\'\s]+?)\s+\w+\?",
            question,
            re.IGNORECASE,
        )
        if not that_verb_match:
            return None
        noun_phrase = that_verb_match.group(1)  # the company
        verb = that_verb_match.group(2)  # published
        entity = that_verb_match.group(3).strip()  # American Scientist

        first_hop = self._fix_parentheses(
            f"{noun_phrase} that {verb} {entity}", question
        )
        if first_hop and first_hop in question:
            second_hop = question.replace(first_hop, "[HOP]", 1)
            return first_hop, self._clean_template(second_hop)
        return None

    def _try_place_cause_pattern(
        self, question: str, wh_word: str | None
    ) -> tuple[str, str] | None:
        """Handle "the place/cause of X of Y" with Where/What/Who."""
        if wh_word not in ("where", "what", "who"):
            return None
        place_pattern = re.search(
            r"\bthe (?:place|cause) of (?:death|birth|origin|residence|burial|interment)\s+of\s+(.+?)\s*\?",
            question,
            re.IGNORECASE,
        )
        if not place_pattern:
            return None
        inner_content = place_pattern.group(1).rstrip("?").strip()
        inner_match = re.search(
            r"\bthe\s+[\w\s]+?\s+of\s+([\w\'\s]+?)$", inner_content, re.IGNORECASE
        )
        first_hop = (
            inner_match.group(0).strip() if inner_match else inner_content.strip()
        )

        if not first_hop:
            return None
        first_hop = self._strip_trailing_verb(first_hop)
        first_hop = self._fix_parentheses(first_hop, question)
        if first_hop and first_hop in question:
            second_hop = question.replace(first_hop, "[HOP]", 1)
            return first_hop, self._clean_template(second_hop)
        return None

    def _strip_verb_prep_from_entity(self, entity: str) -> str:
        """Remove a trailing verb+preposition phrase from the entity."""
        for verb_prep in self.VERB_PREP_PATTERNS:
            vp_match = re.search(
                rf"\s+{re.escape(verb_prep)}\s*$", entity, re.IGNORECASE
            )
            if vp_match:
                return entity[: vp_match.start()].strip()
        return entity

    def _extract_entity_after_of(
        self,
        question: str,
        matches: list[re.Match[str]],
        first_match: re.Match[str],
    ) -> tuple[str, re.Match[str]]:
        """Extract the entity after "of" and the (possibly updated) first match."""
        after_of_idx = first_match.end()
        rest = question[after_of_idx:]

        # "the X of the Y of film/song": use only the inner "the Y of film Z" as first_hop.
        nested_match = re.search(
            r"\bthe\s+[\w\s]+?\s+of\s+(?:film|song)\s+([^\?]+)\??", rest, re.IGNORECASE
        )
        if nested_match:
            film_song_word = re.search(
                r"\b(film|song)\b", nested_match.group(0), re.IGNORECASE
            )
            if film_song_word:
                entity = f"{film_song_word.group(1)} {nested_match.group(1)}".rstrip(
                    "?"
                ).strip()
            else:
                entity = nested_match.group(1).rstrip("?").strip()
            entity = self._strip_verb_prep_from_entity(entity)
            # Remove a trailing year expression (leave it at the end of the question).
            year_match = re.search(r"\s+in\s+\d{4}\s*$", entity, re.IGNORECASE)
            if year_match:
                entity = entity[: year_match.start()].strip()
            # Re-anchor first_match to the nested match.
            actual_nested_start = after_of_idx + nested_match.start()
            for m in matches:
                if m.start() == actual_nested_start:
                    first_match = m
                    break
            return entity, first_match

        # Film/song title only (ignore verb boundaries).
        film_match = re.search(r"\b(film|song)\s+([^\?]+)\??", rest, re.IGNORECASE)
        if film_match:
            entity = film_match.group(0).strip().rstrip("?")
            return self._strip_verb_prep_from_entity(entity), first_match

        # General logic: entity up to a verb or question mark.
        end_idx = rest.find("?")
        if end_idx < 0:
            end_idx = len(rest)
        for verb in _VERBS:
            m = re.search(r"\b" + verb + r"\b", rest, re.IGNORECASE)
            if m and m.start() < end_idx:
                end_idx = m.start()
        return rest[:end_idx].strip(), first_match

    def _try_the_x_of_y(self, question: str) -> tuple[str, str] | None:
        """Handle the "the X of Y" pattern.

        Find a "the ... of ..." pattern in the question
        and extract the full relation chain.

        Example: "Who is the mother of the director of film X?"
        -> first_hop: "the mother of the director of film X" (full chain)
        -> second_hop: "Who is [HOP]?"
        """
        for handler in (
            self._try_who_played,
            self._try_triple_nested_of,
            self._try_nested_outer_property,
            self._try_that_relative_clause,
            self._try_that_verb_clause,
        ):
            result = handler(question)
            if result is not None:
                return result

        matches = list(
            re.finditer(r"\bthe\s+[\w\s]+?\s+of\s+", question, re.IGNORECASE)
        )
        if not matches:
            return None

        wh_word, _ = self._extract_wh_word(question)
        place_result = self._try_place_cause_pattern(question, wh_word)
        if place_result is not None:
            return place_result

        # General "the + (noun phrase) + of" pattern (minimal match).
        first_match = matches[0]
        entity, first_match = self._extract_entity_after_of(
            question, matches, first_match
        )
        entity = self._strip_trailing_verb(entity)

        # Remove the "'s + property noun" pattern from the entity (e.g. "Members's birthday").
        if entity:
            possessive_attr_match = re.search(
                r"^(.+?)'s\s+(birthday|birth|death|anniversary|name|age|height|weight)\s*$",
                entity,
                re.IGNORECASE,
            )
            if possessive_attr_match:
                entity = possessive_attr_match.group(1).strip()

        first_hop = self._fix_parentheses(
            " ".join((first_match.group().strip() + " " + entity).split()), question
        )
        if not first_hop or first_hop not in question:
            return None

        second_hop = question.replace(first_hop, "[HOP]", 1)
        second_hop = self._restore_property_in_second_hop(
            question, first_hop, second_hop
        )
        return first_hop, self._clean_template(second_hop)

    def _get_full_noun_phrase(self, head_token: Token, doc: Doc) -> str:
        """Extract the full noun phrase from a noun token.

        Includes modifiers, apposition, and parentheses.

        e.g. returns the whole "Polish-Russian War (Film)".
        """
        # Find the noun chunk that contains head_token.
        for chunk in doc.noun_chunks:
            if head_token.i >= chunk.start and head_token.i < chunk.end:
                return chunk.text

        # If no chunk, assemble from tokens.
        tokens = [head_token.text]

        # Left modifiers.
        for left in head_token.lefts:
            if left.dep_ in ("amod", "compound", "nmod"):
                tokens.insert(0, left.text)
                for left2 in left.lefts:
                    if left2.dep_ in ("amod", "compound"):
                        tokens.insert(0, left2.text)

        # Right modifiers.
        tokens.extend(
            right.text
            for right in head_token.rights
            if right.dep_ in ("amod", "compound", "appos")
        )
        return " ".join(tokens)

    def _fix_parentheses(self, text: str, question: str) -> str:
        """Match an unbalanced open parenthesis.

        e.g. "the director of film Thomas Jefferson (Film" ->
             "the director of film Thomas Jefferson (Film)"
        """
        # Check whether text is part of the question.
        if text not in question:
            return text

        # If text ends with an open parenthesis, find the closing one in the question.
        if text.endswith("(") or (text.count("(") > text.count(")")):
            idx = question.find(text)
            if idx >= 0:
                rest = question[idx + len(text) :]
                close_idx = rest.find(")")
                if close_idx >= 0:
                    text = text + rest[: close_idx + 1]

        return text

    def _clean_template(self, template: str) -> str:
        template = re.sub(r"\s+\?", "?", template)
        template = re.sub(r"\s+", " ", template)
        template = re.sub(r"\[\s*HOP\s*\]", "[HOP]", template)
        return template.strip()

    def _extract_entity(self, text: str) -> str | None:
        doc = self.nlp(text)
        ents = [
            ent.text
            for ent in doc.ents
            if ent.label_ in ("PERSON", "GPE", "ORG", "WORK_OF_ART")
        ]
        return ents[-1] if ents else None

    def _extract_relation(self, text: str) -> str | None:
        match = re.search(r"\bthe\s+(\w+)\s+of\b", text, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"'s\s+(\w+)", text)
        if match:
            return match.group(1)
        return None


def decompose(question: str, model: str = "en_core_web_sm") -> DecompositionResult:
    """Decompose a question using a fresh MultiHopDecomposer instance."""
    return MultiHopDecomposer(model=model).decompose(question)
