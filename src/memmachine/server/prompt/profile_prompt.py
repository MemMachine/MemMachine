"""Prompt templates for profile-focused semantic categories."""

from memmachine.semantic_memory.semantic_model import (
    SemanticCategory,
    StructuredSemanticPrompt,
)

meta_tags: dict[str, str] = {
    "Assistant Response Preferences": "How the user prefers the assistant to communicate (style, tone, structure, data format).",
    "Notable Past Conversation Topic Highlights": "Recurring or significant discussion themes.",
    "Helpful User Insights": "Key insights that help personalize assistant behavior.",
    "User Interaction Metadata": "Behavioral/technical metadata about platform use.",
    "Political Views, Likes and Dislikes": "Explicit opinions or stated preferences.",
    "Psychological Profile": "Personality characteristics or traits.",
    "Communication Style": "Describes the user's communication tone and pattern.",
    "Learning Preferences": "Preferred modes of receiving information.",
    "Cognitive Style": "How the user processes information or makes decisions.",
    "Emotional Drivers": "Motivators like fear of error or desire for clarity.",
    "Personal Values": "User's core values or principles.",
    "Career & Work Preferences": "Interests, titles, domains related to work.",
    "Productivity Style": "User's work rhythm, focus preference, or task habits.",
    "Demographic Information": "Education level, fields of study, or similar data.",
    "Geographic & Cultural Context": "Physical location or cultural background.",
    "Financial Profile": "Any relevant information about financial behavior or context.",
    "Health & Wellness": "Physical/mental health indicators.",
    "Education & Knowledge Level": "Degrees, subjects, or demonstrated expertise.",
    "Platform Behavior": "Patterns in how the user interacts with the platform.",
    "Tech Proficiency": "Languages, tools, frameworks the user knows.",
    "Hobbies & Interests": "Non-work-related interests.",
    "Social Identity": "Group affiliations or demographics.",
    "Media Consumption Habits": "Types of media consumed (e.g., blogs, podcasts).",
    "Life Goals & Milestones": "Short- or long-term aspirations.",
    "Relationship & Family Context": "Any information about personal life.",
    "Risk Tolerance": "Comfort with uncertainty, experimentation, or failure.",
    "Assistant Trust Level": "Whether and when the user trusts assistant responses.",
    "Time Usage Patterns": "Frequency and habits of use.",
    "Preferred Content Format": "Formats preferred for answers (e.g., tables, bullet points).",
    "Assistant Usage Patterns": "Habits or styles in how the user engages with the assistant.",
    "Language Preferences": "Preferred tone and structure of assistant's language.",
    "Motivation Triggers": "Traits that drive engagement or satisfaction.",
    "Behavior Under Stress": "How the user reacts to failures or inaccurate responses.",
}

description = """
    IMPORTANT: Extract ALL personal information, even basic facts like names, ages, locations, etc. Do not consider any personal information as "irrelevant" - names, basic demographics, and simple facts are valuable profile data.
"""

UserProfileSemanticCategory = SemanticCategory(
    name="profile",
    prompt=StructuredSemanticPrompt(
        tags=meta_tags,
        description=description,
    ),
)

SEMANTIC_TYPE = UserProfileSemanticCategory
