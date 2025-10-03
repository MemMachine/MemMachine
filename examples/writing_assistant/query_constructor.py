import logging
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_query_constructor import BaseQueryConstructor

logger = logging.getLogger(__name__)


class WritingAssistantQueryConstructor(BaseQueryConstructor):
    def __init__(self):
        self.prompt_template = """
You are a writing assistant that helps users write content in their writing style.

<CURRENT_DATE>
{current_date}
</CURRENT_DATE>

**WRITING ASSISTANT CAPABILITIES:**
- Analyze and store your writing style from samples you provide using "/submit" command
- Generate new content in your established writing style
- Help improve existing content while maintaining your voice
- Provide writing guidance and suggestions

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

**HOW TO SUBMIT WRITING SAMPLES:**
Use the "/submit" command followed by the content type and your writing sample:
- "/submit email Dear John, I hope this finds you well..."
- "/submit blog The future of technology is bright..."
- "/submit general This is a sample of my writing..."

<PROFILE>
{profile}
</PROFILE>

<CONTEXT>
{context_block}
</CONTEXT>

<USER_QUERY>
{query}
</USER_QUERY>
"""

    def detect_submit_command(self, query: str) -> dict:
        """Detect if the query contains a /submit command and extract components"""
        # Check if query starts with /submit
        if not query.strip().lower().startswith("/submit"):
            return {
                "is_submission": False,
                "content_type": None,
                "writing_sample": None,
                "original_query": query,
            }

        # Remove /submit and split by space
        remaining = query[7:].strip()  # Remove "/submit"

        # Split into potential content type and writing sample
        parts = remaining.split(" ", 1)

        if len(parts) == 1:
            # Only writing sample, no content type
            return {
                "is_submission": True,
                "content_type": "general",
                "writing_sample": parts[0],
                "original_query": query,
            }
        else:
            # Has content type and writing sample
            content_type = parts[0].lower()
            writing_sample = parts[1]

            # Common content types
            valid_types = [
                "email",
                "blog",
                "linkedin",
                "twitter",
                "facebook",
                "instagram",
                "report",
                "proposal",
                "memo",
                "letter",
                "essay",
                "article",
                "creative",
                "technical",
                "formal",
                "casual",
                "general",
            ]

            if content_type not in valid_types:
                # If first word is not a recognized content type, treat it as general
                return {
                    "is_submission": True,
                    "content_type": "general",
                    "writing_sample": remaining,
                    "original_query": query,
                }

            return {
                "is_submission": True,
                "content_type": content_type,
                "writing_sample": writing_sample,
                "original_query": query,
            }

    def create_query(
        self, profile: Optional[str], context: Optional[str], query: str
    ) -> str:
        """Create a writing assistant query using the prompt template"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Check if this is a /submit command
        submit_info = self.detect_submit_command(query)

        if submit_info["is_submission"]:
            # For submissions, create a specialized prompt
            return self.create_submission_query(profile, context, submit_info)
        else:
            # For regular queries, use the standard template
            profile_str = profile or ""
            context_block = f"{context}\n\n" if context else ""
            current_date = datetime.now().strftime("%Y-%m-%d")

            try:
                return self.prompt_template.format(
                    current_date=current_date,
                    profile=profile_str,
                    context_block=context_block,
                    query=query,
                )
            except Exception as e:
                logger.error(f"Error creating writing assistant query: {e}")
                # Fallback to simple format
                return f"{profile_str}\n\n{context_block}{query}"

    def create_submission_query(
        self, profile: Optional[str], context: Optional[str], submit_info: dict
    ) -> str:
        """Create a specialized query for writing sample submissions"""
        content_type = submit_info["content_type"]
        writing_sample = submit_info["writing_sample"]

        submission_prompt = f"""
You are analyzing a writing sample to extract the user's writing style characteristics.

**SUBMISSION DETAILS:**
- Content Type: {content_type}
- Writing Sample: {writing_sample}

**ANALYSIS TASK:**
Analyze this writing sample and extract detailed writing style features. Focus on observable patterns and characteristics that can be used to generate similar content in the future.

**WRITING STYLE FEATURES TO ANALYZE:**
tone, register, voice, sentence_structure, pacing, word_choice, parts_of_speech_tendency, tense_usage, grammar_quirks, clarity, logic_and_flow, cohesion_devices, paragraphing_style, rhetorical_devices, use_of_examples, directness, personality, humor_style, emotional_intensity, self_reference, signature_phrases, patterned_openings_or_closings, motifs_or_themes, use_of_headings_subheadings

**ANALYSIS GUIDELINES:**
- Extract ONLY from the provided writing sample
- If a feature cannot be determined from the sample, set value to "none"
- Be specific and descriptive in feature values
- Focus on observable patterns, not assumptions
- Use the tag format: "writing_style_{content_type}"

**OUTPUT FORMAT:**
Return a JSON object with the extracted features:
{{"1": {{"command": "add", "feature": "tone", "value": "description", "tag": "writing_style_{content_type}", "author": null}}, ...}}

Current Profile:
{profile or "No existing profile"}

Context:
{context or "No relevant context"}

Please analyze this writing sample and provide the style characteristics.
"""

        return submission_prompt
