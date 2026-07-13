"""
Multi-Hop Question Decomposer - Properly fixed version
"""

import re
import spacy
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class DecomposedHop:
    question: str
    entity: Optional[str] = None
    relation: Optional[str] = None
    answer_type: Optional[str] = None
    hop_order: int = 0


@dataclass
class DecompositionResult:
    original_question: str
    first_hop: str
    second_hop_template: str
    hops: list = field(default_factory=list)
    is_multi_hop: bool = True


class MultiHopDecomposer:
    WH_PATTERNS = {
        'when': 'date/time', 'where': 'location', 'who': 'person',
        'whom': 'person', 'whose': 'person', 'what': 'entity',
        'which': 'entity', 'why': 'reason', 'how': 'manner'
    }

    # Property expressions: excluded from first_hop, preserved in second_hop.
    PROPERTY_PATTERNS = [
        'cause of death', 'date of death', 'place of death',
        'cause of birth', 'date of birth', 'place of birth',
        'award that', 'award which', 'prize that', 'prize which',
        # In "the X of the Y of" patterns, treat X as a property.
        'population of', 'capital of', 'CEO of', 'coach of',
        'profession of', 'grandfather of', 'grandmother of',
        'father of', 'mother of', 'brother of', 'sister of',
        'author of', 'book that', 'song that', 'film that',
        'character of', 'role of',
    ]

    # Verb + preposition phrases: excluded from first_hop, preserved in second_hop.
    VERB_PREP_PATTERNS = [
        'work at', 'work for', 'work with',
        'is from', 'are from', 'was from', 'were from',
        'study at', 'study in', 'studied at', 'studied in',
        'graduate from', 'graduated from',
        'live in', 'live at',
        'born in', 'born at',
        'study', 'studied',
    ]

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

    def decompose(self, question: str) -> DecompositionResult:
        doc = self.nlp(question)
        wh_word, answer_type = self._extract_wh_word(question)

        # 1. Try the possessive ("'s") pattern first.
        possessive_result = self._try_possessive(doc, question)
        first_hop, second_hop = possessive_result if possessive_result else (None, None)

        # 2. If no possessive, try the "the X of Y" pattern.
        if first_hop is None:
            xofy_result = self._try_the_x_of_y(doc, question)
            first_hop, second_hop = xofy_result if xofy_result else (None, None)

        # 3. If neither matches, return the original question (failure).
        if first_hop is None:
            first_hop = question
            second_hop = question
        else:
            # Post-process property expressions and verb + preposition phrases.
            first_hop, second_hop = self._fix_property_and_verb_prep(question, first_hop, second_hop)

        hops = [
            DecomposedHop(question=first_hop, entity=self._extract_entity(first_hop),
                         relation=self._extract_relation(first_hop), hop_order=0),
            DecomposedHop(question=second_hop.replace("[HOP]", first_hop),
                         answer_type=answer_type, hop_order=1)
        ]

        return DecompositionResult(
            original_question=question, first_hop=first_hop,
            second_hop_template=second_hop, hops=hops,
            is_multi_hop=(first_hop != question)
        )

    def _extract_wh_word(self, question: str) -> Tuple[Optional[str], str]:
        q = question.lower().strip()
        for wh, atype in self.WH_PATTERNS.items():
            if q.startswith(wh + " ") or q.startswith(wh + "'"):
                return wh, atype
        return None, 'unknown'

    def _fix_property_and_verb_prep(self, question: str, first_hop: str, second_hop: str) -> Tuple[str, str]:
        """
        Remove property expressions and verb+preposition phrases from first_hop
        and add them to second_hop.

        Example: "What is the cause of death of X's father?"
        first_hop: "the cause of death of X's father" -> "X's father"
        second_hop: "What is [HOP]?" -> "What is the cause of death of [HOP]?"
        """
        # Property expression handling.
        for prop in self.PROPERTY_PATTERNS:
            prop_pattern = rf'\bthe\s+{re.escape(prop)}\s+of\s+'
            prop_match = re.search(prop_pattern, first_hop, re.IGNORECASE)
            if prop_match:
                # Remove the property expression from first_hop.
                first_hop = first_hop.replace(prop_match.group(0), '', 1)
                # Add the property expression to second_hop, preserving the
                # original WH word's capitalization.
                orig_wh_word, _ = self._extract_wh_word(question)
                if orig_wh_word:
                    # "What is [HOP]?" -> "What is the cause of death of [HOP]?"
                    second_hop = re.sub(rf'\b{re.escape(orig_wh_word)}\s+(?:is|are|was|were)?\s*\[HOP\]',
                                        f'{orig_wh_word.capitalize()} is the {prop} of [HOP]', second_hop, flags=re.IGNORECASE)
                break

        # Verb + preposition phrases ("is from", "work at", "got", etc.).
        # If removed from first_hop but already present in second_hop, fix word order.
        for verb_prep in self.VERB_PREP_PATTERNS:
            # Look for a "[HOP] verb_prep" pattern in second_hop (word order fix).
            if verb_prep in ('is from', 'are from', 'was from', 'were from'):
                # "Which country [HOP] is from?" -> "Which country is [HOP] from?"
                pattern = rf'(\S+)\s+\[HOP\]\s+{re.escape(verb_prep)}'
                if re.search(pattern, second_hop, re.IGNORECASE):
                    second_hop = re.sub(pattern, r'\1 ' + verb_prep.replace(' from', '') + ' [HOP] from', second_hop, flags=re.IGNORECASE)
            # Check if first_hop ends with a verb+preposition phrase (to remove).
            vp_pattern = rf'\s+{re.escape(verb_prep)}\s*\??$'
            vp_match = re.search(vp_pattern, first_hop, re.IGNORECASE)
            if vp_match:
                # Remove the verb+preposition from first_hop.
                first_hop = first_hop[:vp_match.start()].strip()
                # Add it to second_hop if not already there.
                if not re.search(rf'\b{re.escape(verb_prep)}\s*\??$', second_hop, re.IGNORECASE):
                    second_hop = second_hop.rstrip('?').rstrip() + ' ' + verb_prep + '?'
                break

        # Handle verbs not in VERB_PREP_PATTERNS: "got", "won", "received", "earned".
        for verb in ['got', 'won', 'received', 'earned']:
            verb_pattern = rf'\s+{verb}\s*\??$'
            verb_match = re.search(verb_pattern, first_hop, re.IGNORECASE)
            if verb_match:
                first_hop = first_hop[:verb_match.start()].strip()
                # Add the verb to second_hop if not already there.
                if not re.search(rf'\b{re.escape(verb)}\s*\??$', second_hop, re.IGNORECASE):
                    second_hop = second_hop.rstrip('?').rstrip() + ' ' + verb + '?'
                break

        return first_hop.strip(), self._clean_template(second_hop)

    def _try_possessive(self, doc, question: str) -> Optional[Tuple[str, str]]:
        """
        Possessive pattern: extract the full "'s" relation expression as first_hop.

        Example: "When did John V's father die?"
        -> first_hop: "John V's father"
        -> second_hop: "When did [HOP] die?"

        Strategy:
        1. Find the "'s" token.
        2. first_hop: owner (left of 's) + 's + relation noun (right of 's).
        3. second_hop: replace the whole first_hop with [HOP].
        """
        for token in doc:
            if token.text == "'s":
                # 1. Find the relation: the first noun to the right of 's.
                # Skip property words like "birthday", "death", "birth".
                # e.g. "X's birthday" -> relation=None, keep "birthday" in second_hop.
                relation = None
                skip_nouns = {'birthday', 'birth', 'death', 'anniversary', 'name', 'age', 'height', 'weight'}
                for i in range(token.i + 1, len(doc)):
                    t = doc[i]
                    if t.pos_ in ["NOUN", "PROPN"]:
                        if t.text.lower() in skip_nouns:
                            # Property words are not extracted as the relation.
                            break
                        relation = t
                        break
                    # Stop at a verb or WH word (no relation).
                    if t.pos_ in ["VERB", "AUX"] or t.text.lower() in self.WH_PATTERNS:
                        break

                if not relation:
                    continue

                # 2. Extract the owner: all tokens left of 's (up to a WH word or aux verb).
                start_idx = 0
                for i in range(token.i - 1, -1, -1):
                    t = doc[i]
                    # Start after a WH word.
                    if t.text.lower() in self.WH_PATTERNS:
                        start_idx = i + 1
                        break
                    # Start after an aux verb (did, does, is, etc.).
                    if t.pos_ == "AUX":
                        start_idx = i + 1
                        break
                    # Start after a ROOT verb.
                    if t.pos_ in ["VERB"] and t.dep_ == "ROOT":
                        start_idx = i + 1
                        break
                else:
                    start_idx = 0

                # Extract the owner (with punctuation), from the original question.
                owner_start = doc[start_idx].idx
                owner_end = token.idx
                owner = question[owner_start:owner_end].strip().rstrip("'").strip()

                # "Which country X's Y" pattern: remove "country" from the owner.
                # e.g. "Which country Gilduin Of Le Puiset's father is from?"
                # -> owner: "country Gilduin Of Le Puiset" -> "Gilduin Of Le Puiset"
                wh_word, _ = self._extract_wh_word(question)
                if wh_word:
                    country_match = re.search(rf'^{re.escape(wh_word)}\s+country\s+(.+)$', owner, re.IGNORECASE)
                    if country_match:
                        owner = country_match.group(1).strip()
                    elif owner.lower().startswith('country '):
                        owner = owner[8:].strip()

                # 3. first_hop: owner + 's + relation noun.
                if not owner:
                    continue

                # "the place of X of Y's Z" pattern: remove "the place of X of" from owner.
                place_of_match = re.search(r'\bthe place of (?:death|birth|origin|residence|burial|interment)\s+of\s+(.+)$', owner, re.IGNORECASE)
                if place_of_match:
                    # Limit owner to the Y of "the place of X of Y".
                    owner = place_of_match.group(1).strip()

                first_hop = f"{owner}'s {relation.text}"
                first_hop = self._fix_parentheses(first_hop, question)

                # If first_hop is not in the question, continue.
                if first_hop not in question:
                    continue

                # 4. second_hop: replace the whole first_hop with [HOP].
                # Preserve property expressions ("the cause of death of", etc.).
                second_hop = question.replace(first_hop, "[HOP]", 1)

                # Restore property expressions: "What is the cause of death of [HOP]?".
                for prop in self.PROPERTY_PATTERNS:
                    prop_pattern = rf'\bthe\s+{re.escape(prop)}\s+of\s+'
                    prop_match = re.search(prop_pattern, question, re.IGNORECASE)
                    if prop_match:
                        # If the property expression precedes first_hop, add it to second_hop.
                        if prop_match.start() < question.find(first_hop):
                            prop_phrase = prop_match.group(0).strip()
                            # If second_hop is "What is [HOP]?", change it to "What is the cause of death of [HOP]?".
                            wh_word, _ = self._extract_wh_word(question)
                            if wh_word:
                                # "What is [HOP]?" -> "What is the cause of death of [HOP]?"
                                second_hop = re.sub(rf'\b{re.escape(wh_word)}\s+(?:is|are|was|were)?\s*\[HOP\]',
                                                    f'{wh_word} is {prop_phrase}[HOP]', second_hop, flags=re.IGNORECASE)

                return first_hop, self._clean_template(second_hop)

        return None

    def _try_the_x_of_y(self, doc, question: str) -> Optional[Tuple[str, str]]:
        """
        "the X of Y" pattern: find a "the ... of ..." pattern in the question
        and extract the full relation chain.

        Example: "Who is the mother of the director of film X?"
        -> first_hop: "the mother of the director of film X" (full chain)
        -> second_hop: "Who is [HOP]?"

        Example: "When does the founder of Microsoft die?"
        -> first_hop: "the founder of Microsoft"
        -> second_hop: "When does [HOP] die?"

        Example: "What is the award that the director of film X won?"
        -> first_hop: "the director of film X" (exclude the relative clause)
        -> second_hop: "What is the award that [HOP] won?"
        """
        # "Who played the character of X" pattern.
        # e.g. "Who played the character of the hero in film Avengers?"
        # -> first_hop: "the hero in film Avengers"
        # -> second_hop: "Who played the character of [HOP]?"
        who_played_match = re.search(r'\b(W|w)ho\s+(played|plays)\s+the\s+(character|role)\s+of\s+(.+)\??', question, re.IGNORECASE)
        if who_played_match:
            entity_part = who_played_match.group(4).strip().rstrip('?')
            # Extract if a film/song pattern is present.
            film_match = re.search(r'\b(film|song)\s+([^\?]+)', entity_part, re.IGNORECASE)
            if film_match:
                first_hop = entity_part.rstrip('?').strip()
            else:
                first_hop = entity_part.rstrip('?').strip()

            second_hop = f"Who played the character of [HOP]?"
            if first_hop and first_hop in question:
                return first_hop, second_hop

        # Nested "the X of the Y of the Z of" pattern: use only the innermost as first_hop.
        # e.g. "Who is the grandfather of the son of the inventor of telephone?"
        # -> first_hop: "the inventor of telephone"
        # -> second_hop: "Who is the grandfather of the son of [HOP]?"
        # e.g. "What is the profession of the brother of the actress who won Oscar?"
        # -> first_hop: "the actress who won Oscar"
        # -> second_hop: "What is the profession of the brother of [HOP]?"

        # Find all "the X of" patterns.
        all_of_matches = list(re.finditer(r'\bthe\s+[\w\s]+?\s+of\s+', question, re.IGNORECASE))

        # Three or more matches -> handle as a 3-level nested pattern.
        if len(all_of_matches) >= 3:
            # Use only the innermost two matches.
            second_innermost = all_of_matches[-2]
            innermost_match = all_of_matches[-1]

            # Use the whole "the X of Y" as first_hop (X = innermost_match, Y = the rest).
            # From innermost_match.start() up to the question mark.
            first_hop_start = innermost_match.start()
            first_hop_end = question.find('?')
            if first_hop_end < 0:
                first_hop_end = len(question)

            first_hop = question[first_hop_start:first_hop_end].strip().rstrip('?')

            # Include year expressions (in 2018).
            year_match = re.search(r'\s+in\s+\d{4}', first_hop)
            if year_match:
                first_hop = first_hop[:year_match.end()].strip()

            # second_hop: outer relations + [HOP].
            # Up to the end of the second-innermost match is the outer relation.
            outer_relations = question[:second_innermost.end()].strip()

            # Remove the WH word from outer_relations (avoid duplication).
            wh_word, _ = self._extract_wh_word(outer_relations)
            if wh_word:
                # "Who is the grandfather of the son of " -> "the grandfather of the son of"
                outer_relations = outer_relations[len(wh_word):].strip()
                # Also remove "is".
                outer_relations = re.sub(r'^is\s+', '', outer_relations, flags=re.IGNORECASE)

            wh_word_orig, _ = self._extract_wh_word(question)
            if wh_word_orig:
                second_hop = f"{wh_word_orig.capitalize()} is {outer_relations} [HOP]?"
            else:
                second_hop = f"What is {outer_relations} [HOP]?"

            second_hop = self._clean_template(second_hop)

            if first_hop and first_hop in question:
                return first_hop, second_hop

        # "the X of the Y of" pattern (2-level nesting): treat the outer "the X of" as a property.
        # e.g. "What is the population of the capital of France?"
        # -> first_hop: "the capital of France"
        # -> second_hop: "What is the population of [HOP]?"
        nested_of_match = re.search(r'\b(the\s+(?:population|capital|CEO|coach|profession|grandfather|grandmother|father|mother|brother|sister|author|book)\s+of)\s+(the\s+.+)$', question, re.IGNORECASE)
        if nested_of_match:
            outer_prop = nested_of_match.group(1).strip()  # "the population of"
            inner_part = nested_of_match.group(2).strip()  # "the capital of France"

            # Extract from inner_part up to a verb/year boundary.
            inner_end = len(inner_part)
            q_idx = inner_part.find('?')
            if q_idx >= 0:
                inner_end = q_idx

            # Include year expressions (in 2018) in inner_part.
            first_hop = inner_part[:inner_end].strip().rstrip('?')

            # Build second_hop.
            wh_word, _ = self._extract_wh_word(question)
            if wh_word:
                second_hop = f"{wh_word.capitalize()} is {outer_prop} [HOP]?"
            else:
                second_hop = f"What is {outer_prop} [HOP]?"

            if first_hop and first_hop in question:
                return first_hop, self._clean_template(second_hop)

        # Check the "that the X of" pattern first (relative clause handling).
        # e.g. "What is the award that the director of film X won?"
        that_match = re.search(r'\bthat\s+(the\s+[\w\s]+?\s+of\s+)', question, re.IGNORECASE)
        if that_match:
            # Use the "the ... of ..." pattern after "that".
            after_that_start = that_match.start(1)
            matches = list(re.finditer(r'\bthe\s+[\w\s]+?\s+of\s+', question[after_that_start:], re.IGNORECASE))
            if matches:
                first_match = matches[0]
                # Adjust to the actual position by adding after_that_start.
                actual_start = after_that_start + first_match.start()
                actual_end = after_that_start + first_match.end()

                # Extract the entity after "of".
                after_of_idx = actual_end
                rest = question[after_of_idx:]

                # Find the verb/question-mark boundary.
                end_idx = len(rest)
                q_idx = rest.find('?')
                if q_idx >= 0:
                    end_idx = q_idx

                verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did',
                         'will', 'would', 'could', 'won', 'find', 'found', 'work', 'earned', 'born', 'die']

                # For relative clauses like "that inspired X", "that won X", do not use a verb boundary.
                if not re.search(r'\bthat\s+(inspired|won|lost|created|made|directed|wrote)\b', rest, re.IGNORECASE):
                    for verb in verbs:
                        pattern = r'\b' + verb + r'\b'
                        m = re.search(pattern, rest, re.IGNORECASE)
                        if m and m.start() < end_idx:
                            end_idx = m.start()

                entity = rest[:end_idx].strip()
                if entity:
                    last_word = entity.split()[-1].lower().rstrip('.,!?')
                    if last_word in verbs:
                        entity = ' '.join(entity.split()[:-1]).strip()

                # Preserve proper nouns (Harry Potter -> whole "Harry Potter").
                # Use spaCy to determine the entity boundary.
                if entity:
                    entity_doc = self.nlp(entity)
                    for ent in entity_doc.ents:
                        if ent.label_ in ["PERSON", "WORK_OF_ART", "GPE", "ORG"]:
                            # Preserve the whole multi-word entity.
                            ent_text = ent.text
                            if ' ' in ent_text and ent_text in entity:
                                # Include any word following the entity.
                                ent_end_idx = entity.find(ent_text) + len(ent_text)
                                remaining = entity[ent_end_idx:].strip()
                                if remaining and not re.match(r'\b(in|at|on|of|the|a|an)\b', remaining, re.IGNORECASE):
                                    # If not a year or preposition, do not include.
                                    pass

                first_hop = first_match.group().strip() + ' ' + entity
                first_hop = ' '.join(first_hop.split())
                first_hop = self._fix_parentheses(first_hop, question)

                if first_hop and first_hop in question:
                    second_hop = question.replace(first_hop, "[HOP]", 1)
                    return first_hop, self._clean_template(second_hop)

        # "the X that + verb + Y" pattern (relative clause + verb).
        # e.g. "When was the company that published American Scientist founded?"
        # -> first_hop: "the company that published American Scientist"
        # -> second_hop: "When was [HOP] founded?"
        that_verb_match = re.search(r'\b(the\s+\w+)\s+that\s+(\w+)\s+([\w\'\s]+?)\s+\w+\?', question, re.IGNORECASE)
        if that_verb_match:
            noun_phrase = that_verb_match.group(1)  # the company
            verb = that_verb_match.group(2)  # published
            entity = that_verb_match.group(3).strip()  # American Scientist

            first_hop = f"{noun_phrase} that {verb} {entity}"
            first_hop = self._fix_parentheses(first_hop, question)

            if first_hop and first_hop in question:
                second_hop = question.replace(first_hop, "[HOP]", 1)
                return first_hop, self._clean_template(second_hop)

        # General "the + (noun phrase) + of" pattern (minimal match).
        matches = list(re.finditer(r'\bthe\s+[\w\s]+?\s+of\s+', question, re.IGNORECASE))

        if not matches:
            return None

        # Check for a WH word.
        wh_word, _ = self._extract_wh_word(question)

        # "the place/cause of X of Y" + "Where/What/Who" pattern.
        # e.g. "Where was the place of death of the director of film X?"
        # -> first_hop: "the director of film X" (inner relation only)
        # -> second_hop: "Where was the place of death of [HOP]?"
        # e.g. "What is the place of birth of the performer of song X?"
        # -> first_hop: "the performer of song X"
        # -> second_hop: "What is the place of birth of [HOP]?"
        # e.g. "What is the cause of death of performer of song X?"
        # -> first_hop: "the performer of song X"
        # -> second_hop: "What is the cause of death of [HOP]?"
        place_pattern = re.search(r'\bthe (?:place|cause) of (?:death|birth|origin|residence|burial|interment)\s+of\s+(.+?)\s*\?', question, re.IGNORECASE)
        if place_pattern and wh_word in ('where', 'what', 'who'):
            inner_content = place_pattern.group(1).rstrip('?').strip()
            # Look for a "the X of Y" pattern within inner_content.
            inner_match = re.search(r'\bthe\s+[\w\s]+?\s+of\s+([\w\'\s]+?)$', inner_content, re.IGNORECASE)
            if inner_match:
                first_hop = inner_match.group(0).strip()
            else:
                # If no "the X of Y" pattern, use the whole inner_content as first_hop.
                first_hop = inner_content.strip()

            if first_hop:
                # Remove a trailing verb from first_hop.
                verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did',
                         'will', 'would', 'could', 'won', 'find', 'found', 'work', 'earned', 'born', 'die']
                last_word = first_hop.split()[-1].lower().rstrip('.,!?')
                if last_word in verbs:
                    first_hop = ' '.join(first_hop.split()[:-1]).strip()

                first_hop = self._fix_parentheses(first_hop, question)
                if first_hop and first_hop in question:
                    second_hop = question.replace(first_hop, "[HOP]", 1)
                    return first_hop, self._clean_template(second_hop)

        # If not a place_of pattern (deduplicated), use the outermost "the X of".
        # film/song is only used to ignore verb boundaries when extracting the entity.
        first_match = matches[0]

        # Extract the entity after "of" (up to a verb or question mark).
        after_of_idx = first_match.end()
        rest = question[after_of_idx:]

        # Predefine the verb list.
        verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did',
                 'will', 'would', 'could', 'won', 'find', 'found', 'work', 'earned', 'born', 'die']

        # "the X of the Y of film/song" pattern: nested relations use only the innermost as first_hop.
        # e.g. "Who is the mother of the director of film X?"
        # -> first_hop: "the director of film X"
        # -> second_hop: "Who is the mother of [HOP]?"
        nested_match = re.search(r'\bthe\s+[\w\s]+?\s+of\s+(?:film|song)\s+([^\?]+)\??', rest, re.IGNORECASE)
        if nested_match:
            # Use only the inner "the Y of film Z" as first_hop.
            # Extract the whole "film/song [title]" as the entity, excluding verb+preposition phrases.
            film_song_word = re.search(r'\b(film|song)\b', nested_match.group(0), re.IGNORECASE)
            if film_song_word:
                entity = f"{film_song_word.group(1)} {nested_match.group(1)}".rstrip('?').strip()
            else:
                entity = nested_match.group(1).rstrip('?').strip()

            # Remove verb+preposition phrases from the entity ("is from", "work at", etc.).
            for verb_prep in self.VERB_PREP_PATTERNS:
                vp_pattern = rf'\s+{re.escape(verb_prep)}\s*$'
                vp_match = re.search(vp_pattern, entity, re.IGNORECASE)
                if vp_match:
                    entity = entity[:vp_match.start()].strip()
                    break

            # Include year expressions in the entity (in 2018, in 2020, etc.).
            year_match = re.search(r'\s+in\s+\d{4}\s*$', entity, re.IGNORECASE)
            if year_match:
                # Remove the year expression from the entity (leave it at the end of the question).
                entity = entity[:year_match.start()].strip()

            # Change first_match to the start of nested_match.
            nested_start_in_rest = nested_match.start()
            actual_nested_start = after_of_idx + nested_start_in_rest
            for m in matches:
                if m.start() == actual_nested_start:
                    first_match = m
                    break
            after_of_idx = first_match.end()
            rest = question[after_of_idx:]
        else:
            # Extract only the film/song title (ignore verb boundaries).
            film_match = re.search(r'\b(film|song)\s+([^\?]+)\??', rest, re.IGNORECASE)
            if film_match:
                entity = film_match.group(0).strip().rstrip('?')
                # Remove verb+preposition phrases from the entity.
                for verb_prep in self.VERB_PREP_PATTERNS:
                    vp_pattern = rf'\s+{re.escape(verb_prep)}\s*$'
                    vp_match = re.search(vp_pattern, entity, re.IGNORECASE)
                    if vp_match:
                        entity = entity[:vp_match.start()].strip()
                        break
            else:
                # If no film/song pattern, use the general logic (verb boundary).
                end_idx = len(rest)
                q_idx = rest.find('?')
                if q_idx >= 0:
                    end_idx = q_idx

                for verb in verbs:
                    pattern = r'\b' + verb + r'\b'
                    m = re.search(pattern, rest, re.IGNORECASE)
                    if m and m.start() < end_idx:
                        end_idx = m.start()

                entity = rest[:end_idx].strip()

        # Remove the last word if it is a verb (double check).
        if entity:
            last_word = entity.split()[-1].lower().rstrip('.,!?')
            if last_word in verbs:
                entity = ' '.join(entity.split()[:-1]).strip()

        # Remove the "'s + property noun" pattern from the entity (e.g. "Members's birthday").
        # e.g. "List Of The Dillinger Escape Plan Band Members's birthday"
        #      -> "List Of The Dillinger Escape Plan Band Members"
        if entity:
            possessive_attr_match = re.search(r"^(.+?)'s\s+(birthday|birth|death|anniversary|name|age|height|weight)\s*$", entity, re.IGNORECASE)
            if possessive_attr_match:
                entity = possessive_attr_match.group(1).strip()

        # first_hop: the whole "the X of Y" (assembled with whitespace).
        first_hop = first_match.group().strip() + ' ' + entity
        first_hop = ' '.join(first_hop.split())  # Normalize whitespace.
        first_hop = self._fix_parentheses(first_hop, question)

        if not first_hop or first_hop not in question:
            return None

        # second_hop: replace first_hop with [HOP].
        second_hop = question.replace(first_hop, "[HOP]", 1)

        # Preserve property expressions: "the cause of death of", "the award that", etc.
        for prop in self.PROPERTY_PATTERNS:
            prop_pattern = rf'\bthe\s+{re.escape(prop)}\s+of\s+'
            prop_match = re.search(prop_pattern, question, re.IGNORECASE)
            if prop_match and prop_match.start() < first_match.start():
                # If the property expression precedes first_hop, add it to second_hop.
                wh_word, _ = self._extract_wh_word(question)
                if wh_word:
                    # "What is [HOP]?" -> "What is the cause of death of [HOP]?"
                    second_hop = re.sub(rf'\b{re.escape(wh_word)}\s+(?:is|are|was|were)?\s*\[HOP\]',
                                        f'{wh_word} is the {prop} of [HOP]', second_hop, flags=re.IGNORECASE)

        return first_hop, self._clean_template(second_hop)

    def _get_full_noun_phrase(self, head_token, doc) -> str:
        """
        Extract the full noun phrase from a noun token (including modifiers,
        apposition, and parentheses).

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
            if left.dep_ in ["amod", "compound", "nmod"]:
                tokens.insert(0, left.text)
                for left2 in left.lefts:
                    if left2.dep_ in ["amod", "compound"]:
                        tokens.insert(0, left2.text)

        # Right modifiers.
        for right in head_token.rights:
            if right.dep_ in ["amod", "compound", "appos"]:
                tokens.append(right.text)

        return " ".join(tokens)

    def _fix_parentheses(self, text: str, question: str) -> str:
        """
        Match parentheses.

        e.g. "the director of film Thomas Jefferson (Film" ->
             "the director of film Thomas Jefferson (Film)"
        """
        # Check whether text is part of the question.
        if text not in question:
            return text

        # If text ends with an open parenthesis, find the closing one in the question.
        if text.endswith('(') or (text.count('(') > text.count(')')):
            # Find the part of the question after text.
            idx = question.find(text)
            if idx >= 0:
                rest = question[idx + len(text):]
                # Find the closing parenthesis in rest.
                close_idx = rest.find(')')
                if close_idx >= 0:
                    # Include up to the closing parenthesis.
                    text = text + rest[:close_idx + 1]

        return text

    def _clean_template(self, template: str) -> str:
        template = re.sub(r'\s+\?', '?', template)
        template = re.sub(r'\s+', ' ', template)
        template = re.sub(r'\[\s*HOP\s*\]', '[HOP]', template)
        return template.strip()

    def _extract_entity(self, text: str) -> Optional[str]:
        doc = self.nlp(text)
        ents = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "ORG", "WORK_OF_ART"]]
        return ents[-1] if ents else None

    def _extract_relation(self, text: str) -> Optional[str]:
        match = re.search(r'\bthe\s+(\w+)\s+of\b', text, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"'s\s+(\w+)", text)
        if match:
            return match.group(1)
        return None


def decompose(question: str, model: str = "en_core_web_sm") -> DecompositionResult:
    return MultiHopDecomposer(model=model).decompose(question)
