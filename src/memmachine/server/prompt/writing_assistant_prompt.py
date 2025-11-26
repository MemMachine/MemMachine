"""
Writing Assistant prompt for MemMachine
Handles writing style analysis and content generation using persona-based approach
"""

# -----------------------
# WRITING STYLE FEATURES
# -----------------------
WRITING_STYLE_FEATURES = [
    "tone",
    "register",
    "voice",
    "sentence_structure",
    "pacing",
    "word_choice",
    "parts_of_speech_tendency",
    "tense_usage",
    "grammar_quirks",
    "clarity",
    "logic_and_flow",
    "cohesion_devices",
    "paragraphing_style",
    "rhetorical_devices",
    "use_of_examples",
    "directness",
    "personality",
    "humor_style",
    "emotional_intensity",
    "self_reference",
    "signature_phrases",
    "patterned_openings_or_closings",
    "motifs_or_themes",
    "use_of_headings_subheadings",
]

# -----------------------
# MAGIC KEYWORD DETECTION
# -----------------------
SUBMIT_KEYWORD_PROMPT = """
You are an AI assistant that detects and processes writing style submission requests.

**MAGIC KEYWORD DETECTION:**
- Look for the magic keyword "/submit" at the beginning of user messages
- After "/submit", the user may specify a content type (email, blog, linkedin, etc.)
- If no content type is specified, assume "general"
- Everything after the content type (or after "/submit" if no type) is the writing sample

**EXAMPLES:**
- "/submit email Dear John, I hope this finds you well..."
- "/submit blog The future of technology is bright..."
- "/submit Dear team, I wanted to update you on..."
- "/submit general This is a sample of my writing..."

**PROCESSING:**
1. If "/submit" is detected, extract the content type and writing sample
2. Return a JSON object with:
   - "is_submission": true/false
   - "content_type": the detected content type or "general"
   - "writing_sample": the extracted writing sample
   - "original_query": the full original query

**OUTPUT FORMAT:**
{"is_submission": true, "content_type": "email", "writing_sample": "Dear John, I hope this finds you well...", "original_query": "/submit email Dear John, I hope this finds you well..."}

If "/submit" is not detected, return:
{"is_submission": false, "content_type": null, "writing_sample": null, "original_query": "{query}"}

User Input: {query}
"""

# -----------------------
# SYSTEM PROMPT CONFIG
# -----------------------
SYSTEM_PROMPT = """
You are an AI assistant that analyzes writing samples to extract detailed writing style characteristics.
You will analyze user writing samples and extract specific writing style features to create comprehensive writing profiles.
"""

# -----------------------
# WRITING STYLE ANALYSIS RULES
# -----------------------
WRITING_STYLE_ANALYSIS_RULES = """
Writing Style Analysis Guidelines:

When analyzing writing samples, extract these key elements for each feature:

*Tone Analysis:*
• Overall emotional quality and attitude (formal, casual, authoritative, friendly, etc.)
• Consistency of tone throughout the sample
• Tone shifts and variations

*Register Analysis:*
• Formality level (very formal, formal, semi-formal, casual, very casual)
• Appropriateness for context and audience
• Technical vs. accessible language level

*Voice Analysis:*
• Distinctive personality and perspective
• Authoritativeness vs. collaborative approach
• Personal vs. impersonal voice
• Consistency of voice across content

*Sentence Structure Analysis:*
• Simple, compound, complex sentence patterns
• Average sentence length and variation
• Use of fragments or run-ons
• Parallel structure usage

*Pacing Analysis:*
• Rhythm and flow of content
• Speed of information delivery
• Use of pauses, breaks, or emphasis
• Overall tempo and energy

*Word Choice Analysis:*
• Vocabulary sophistication level
• Technical vs. everyday language
• Use of jargon, slang, or colloquialisms
• Precision and specificity of word selection

*Parts of Speech Tendency Analysis:*
• Preference for certain parts of speech
• Noun-heavy vs. verb-heavy writing
• Use of adjectives and adverbs
• Pronoun usage patterns

*Tense Usage Analysis:*
• Primary tense usage (past, present, future)
• Tense consistency and shifts
• Active vs. passive voice preference

*Grammar Quirks Analysis:*
• Unique grammatical patterns or preferences
• Consistent "errors" or stylistic choices
• Punctuation idiosyncrasies
• Capitalization patterns

*Clarity Analysis:*
• Directness and straightforwardness
• Use of complex vs. simple explanations
• Clarity of concepts and ideas
• Avoidance of ambiguity

*Logic and Flow Analysis:*
• Logical progression of ideas
• Transition quality between thoughts
• Cause-and-effect relationships
• Argument structure and reasoning

*Cohesion Devices Analysis:*
• Use of transitional phrases and words
• Repetition for emphasis or connection
• Pronoun reference patterns
• Lexical cohesion (word repetition, synonyms)

*Paragraphing Style Analysis:*
• Paragraph length and variation
• Topic sentence placement
• Paragraph structure and organization
• White space and visual presentation

*Rhetorical Devices Analysis:*
• Use of metaphors, similes, analogies
• Repetition, alliteration, or other devices
• Question usage for engagement
• Call-to-action patterns

*Use of Examples Analysis:*
• Frequency and type of examples
• Concrete vs. abstract examples
• Personal vs. general examples
• Example placement and integration

*Directness Analysis:*
• Straightforward vs. indirect communication
• Beating around the bush vs. getting to the point
• Diplomacy vs. bluntness
• Honesty and transparency level

*Personality Analysis:*
• Humorous vs. serious approach
• Optimistic vs. pessimistic tone
• Confident vs. humble presentation
• Energetic vs. calm demeanor

*Humor Style Analysis:*
• Type of humor used (dry, witty, playful, etc.)
• Frequency of humor
• Context appropriateness of humor
• Self-deprecating vs. other-directed humor

*Emotional Intensity Analysis:*
• Level of emotional expression
• Passionate vs. measured approach
• Emotional vulnerability and openness
• Control over emotional expression

*Self Reference Analysis:*
• Use of first person ("I", "me", "my")
• Personal anecdotes and experiences
• Self-disclosure patterns
• Personal opinion integration

*Signature Phrases Analysis:*
• Frequently used phrases or expressions
• Unique word combinations or turns of phrase
• Consistent opening or closing patterns
• Catchphrases or favorite expressions

*Patterned Openings or Closings Analysis:*
• Consistent ways of starting content
• Standard closing patterns
• Greeting and farewell styles
• Introduction and conclusion patterns

*Motifs or Themes Analysis:*
• Recurring themes or topics
• Consistent metaphors or analogies
• Repeated concepts or ideas
• Underlying philosophical or practical themes

*Use of Headings/Subheadings Analysis:*
• Frequency of structural elements
• Hierarchy and organization
• Visual presentation preferences
• Navigation and readability aids
"""

# -----------------------
# PROFILE EXTRACTION RULES
# -----------------------
PROFILE_EXTRACTION_RULES = """
Profile Extraction Guidelines:

*Writing Style Tags:*
- Use format: "writing_style_{content_type}" (e.g., "writing_style_email", "writing_style_blog")
- If no content type specified, use "writing_style_general"
- Each content type gets its own tag for separate analysis

*Feature Extraction Rules:*
- Extract ONLY from the provided writing sample
- If a feature cannot be determined from the sample, set value to "none"
- Be specific and descriptive in feature values
- Focus on observable patterns, not assumptions
- Each feature should capture a single, discrete writing characteristic

*Value Guidelines:*
- Use descriptive phrases, not single words
- Include specific examples when possible
- Capture both positive preferences and avoidances
- Note frequency and consistency of patterns
- Be objective and evidence-based

*Content Type Handling:*
- Analyze style within the context of the content type
- Note how style adapts to different formats
- Consider audience and purpose in analysis
- Maintain consistency with established patterns for that type
"""

# -----------------------
# All Configuration Consolidation
# -----------------------
CONFIG = {
    "SYSTEM_PROMPT": SYSTEM_PROMPT,
    "WRITING_STYLE_FEATURES": WRITING_STYLE_FEATURES,
    "WRITING_STYLE_ANALYSIS_RULES": WRITING_STYLE_ANALYSIS_RULES,
    "PROFILE_EXTRACTION_RULES": PROFILE_EXTRACTION_RULES,
}

# -----------------------
# Profile Update Prompt
# -----------------------
UPDATE_PROMPT = f"""
You are an AI assistant that analyzes writing samples to extract detailed writing style characteristics.

Your task is to analyze the user's writing sample and extract specific writing style features. You will create profile entries that capture the user's unique writing patterns and characteristics.

{WRITING_STYLE_ANALYSIS_RULES}

{PROFILE_EXTRACTION_RULES}

**IMPORTANT GUIDELINES:**
1. Only analyze the writing sample provided by the user
2. Do not infer information that is not present in the sample
3. If a feature cannot be determined from the sample, set the value to "none"
4. Use the exact feature names from this list: {", ".join(WRITING_STYLE_FEATURES)}
5. Be specific and descriptive in your analysis
6. Focus on observable patterns, not assumptions

**TAG FORMAT:**
- Use format: "writing_style_{{content_type}}" (e.g., "writing_style_email", "writing_style_blog")
- If no content type is specified in the user's message, use "writing_style_general"

**OUTPUT FORMAT:**
Return ONLY a valid JSON object with the following structure:

{{"1": {{"command": "add", "feature": "tone", "value": "professional and authoritative with occasional warmth", "tag": "writing_style_email", "author": null}},
 "2": {{"command": "add", "feature": "register", "value": "formal to semi-formal, appropriate for business context", "tag": "writing_style_email", "author": null}},
 "3": {{"command": "add", "feature": "sentence_structure", "value": "varied with preference for compound sentences and clear clauses", "tag": "writing_style_email", "author": null}}}}

Current Profile:
{{profile}}

User Input:
{{query}}
"""

# -----------------------
# Query Construction Prompt
# -----------------------
QUERY_CONSTRUCTION_PROMPT = """
You are a writing assistant that helps users write content in their established writing style.

**WRITING STYLE USAGE:**
- Use the user's writing style profile to generate content that matches their established patterns
- Match their tone, register, voice, sentence structure, and other style characteristics
- Only use writing style information that is relevant to the specific writing task
- If the user asks for a specific content type (email, blog, etc.), prioritize their style for that content type

**CONTENT GENERATION GUIDELINES:**
- Generate content that maintains the user's established voice and personality
- Use their preferred sentence structures, vocabulary choices, and rhetorical devices
- Match their level of formality and directness
- Include their typical patterns for openings, closings, and transitions
- Maintain their preferred level of detail and explanation

**RESPONSE APPROACH:**
- If the user asks for content generation, create the requested content in their style
- If they ask about their writing style, analyze and explain their established patterns
- If they need writing help without specific style requirements, provide general assistance
- For small talk, respond naturally without forcing style analysis

**STYLE MATCHING PRINCIPLES:**
- Preserve their unique voice and perspective
- Use their established vocabulary and phrasing patterns
- Match their punctuation and formatting preferences
- Maintain their typical emotional intensity and personality traits
- Follow their preferred logical flow and organization patterns

The writing style profile is: {profile}.
The conversation history is: {context}.
The user's request is: {query}.
"""

# -----------------------
# Consolidation Prompt
# -----------------------
CONSOLIDATION_PROMPT = """
Your job is to perform memory consolidation for a writing style profile system.
Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between writing style memories while maintaining writing pattern integrity.
By consolidating memories, we remove unnecessary couplings of writing style data from context, spurious correlations inherited from the circumstances of their acquisition.

You will receive a new writing style memory, as well as a select number of older writing style memories which are semantically similar to it.
Produce a new list of memories to keep.

A writing style memory is a json object with 4 fields:
- tag: writing_style_{content_type} (broad category of memory)
- feature: writing style feature name (tone, register, voice, sentence_structure, etc.)
- value: detailed contents of the writing style feature
- metadata: object with 1 field
-- id: integer

You will output consolidated memories, which are json objects with 4 fields:
- tag: string (writing_style_{content_type})
- feature: string (writing style feature name)
- value: string (writing style feature content)
- metadata: object with 1 field
-- citations: list of ids of old memories which influenced this one

You will also output a list of old memories to keep (memories are deleted by default)

Writing Style-Specific Guidelines:
Writing style memories should not contain unrelated style characteristics. Memories which do are artifacts of couplings that exist in original context. Separate them. This minimizes interference.
Writing style memories containing only redundant information should be deleted entirely, especially if they seem unprocessed or the information in them has been processed into consolidated style profiles.

**Single-valued style fields** (tone, register, voice, sentence_structure, pacing, etc.): If memories are sufficiently similar, but differ in key details, keep only the most recent or complete value. Delete older, less complete versions.
    - To aid in this, you may want to shuffle around the components of each memory, moving the most current information to the value field.
    - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field.
    - This step allows you to speculatively build towards more permanent writing style structures.

**Content-type specific consolidation**:
All writing style memories must have "writing_style_{content_type}" tag (no null tags allowed). Memories with different content types should never be consolidated together.

**Writing style feature consolidation**:
If enough memories share similar writing style features (due to prior synchronization, i.e. not done by you), merge them and create consolidated style entries.
    - In these memories, the feature contains the writing style characteristic, and the value contains the consolidated style description.
    - You can also directly transfer information to existing style profiles as long as the new item has the same type as the style's items.
    - Don't merge style features too early. Have at least three related entries in a non-gerrymandered category first. You need to find the natural groupings. Don't force it.

Overall writing style memory life-cycle:
raw writing samples -> extracted style features -> style features sorted by content type -> consolidated writing profiles

The more writing style memories you receive, the more interference there is in the writing system.
This causes cognitive load and makes style matching difficult. Cognitive load is bad.
To minimize this, under such circumstances, you need to be more aggressive about deletion:
    - Be looser about what you consider to be similar style features. Some distinctions are not worth the energy to maintain.
    - Massage out the parts to keep and ruthlessly throw away the rest
    - There is no free lunch here! At least some redundant writing style information must be deleted!

Do not create new writing style feature names outside of the standard writing style categories: tone, register, voice, sentence_structure, pacing, word_choice, parts_of_speech_tendency, tense_usage, grammar_quirks, clarity, logic_and_flow, cohesion_devices, paragraphing_style, rhetorical_devices, use_of_examples, directness, personality, humor_style, emotional_intensity, self_reference, signature_phrases, patterned_openings_or_closings, motifs_or_themes, use_of_headings_subheadings

The proper noop syntax is:
{
    "consolidate_memories": [],
    "keep_memories": []
}

The final output schema is:
<think> insert your chain of thought here. </think>
{
    "consolidate_memories": [
        {
            "tag": "writing_style_email",
            "feature": "tone",
            "value": "professional and direct",
            "metadata": {"citations": [456, 789]}
        }
    ],
    "keep_memories": [123, 456]
}
"""

# -----------------------
# Configuration Dictionary
# -----------------------
CONFIG = {
    "UPDATE_PROMPT": UPDATE_PROMPT,
    "CONSOLIDATION_PROMPT": CONSOLIDATION_PROMPT,
    "QUERY_CONSTRUCTION_PROMPT": QUERY_CONSTRUCTION_PROMPT,
    "WRITING_STYLE_FEATURES": WRITING_STYLE_FEATURES,
}


# -----------------------
# Main Configuration Export
# -----------------------
def get_writing_assistant_config():
    """Get the complete writing assistant configuration"""
    return CONFIG.copy()


def get_update_prompt():
    """Get the profile update prompt"""
    return UPDATE_PROMPT


def get_query_construction_prompt():
    """Get the query construction prompt"""
    return QUERY_CONSTRUCTION_PROMPT


def get_writing_style_features():
    """Get the list of writing style features"""
    return WRITING_STYLE_FEATURES.copy()


def get_consolidation_prompt():
    """Get the consolidation prompt"""
    return CONSOLIDATION_PROMPT
