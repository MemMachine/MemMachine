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
    "Decision-Making Style": "How the user tends to think, plan, and decide.",
    "Hard Skills": "Specialized technical or professional capabilities in a work domain.",
    "Soft Skills": "Professional competencies such as communication, teamwork, or leadership.",
    "Communication Style": "Describes the user's communication tone and pattern.",
    "Learning Preferences": "Preferred modes of receiving information.",
    "Cognitive Style": "How the user processes information or makes decisions.",
    "Emotional Drivers": "Motivators like fear of error or desire for clarity.",
    "Personal Values": "User's core values or principles.",
    "Career & Work Preferences": "Interests, titles, domains related to work.",
    "Productivity Style": "User's work rhythm, focus preference, or task habits.",
    "Working Habit Preferences": "Recurring, personally driven ways the user organizes, performs, communicates about, or manages work.",
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

    Category-specific rules:
    - Working Habit Preferences: A recurring, personally driven tendency in how the user organizes, performs, communicates about, or manages their work.
      * Explicit linguistic signals: "I prefer", "I like to", "I tend to", "I usually", "I'm used to", "I'd rather", "It's easier for me to", "I avoid", "I don't like", "I try to avoid", "Works best for me when".
      * Boundary test: If the external rule disappeared, would the user still choose to do it that way? If yes, it's a preference.
    - Psychological Profile (Personality traits): Only extract traits when the user clearly and directly indicates them. Avoid speculative inference.
    - Decision-Making Style: Only add when there is clear evidence for one of these dimensions: SystematicThinking, QualityFirstPrinciple, DataDrivenDecisionMaking, ForwardLookingPlanning, ClearResponsibilityBoundaries, ContinuousImprovementMindset.
    - Hard Skills: Extract ONLY when the user explicitly demonstrates or describes hands-on use of tools/technologies, or explains technical principles in depth.
      * Use a level descriptor in the value: Expert / Proficient / Familiar, based on evidence in the user's message.
      * Example value format: "Backend Development - Proficient; basis: built REST APIs in Python and FastAPI".
    - Soft Skills: Rate only when evidence exists in the user's message. Use the six dimensions: Communication, Teamwork, Emotional Intelligence, Time Management, Problem-Solving, Leadership.
      * Example value format: "Communication - Strong; basis: concise structured requests and explicit constraints".

    General guidance:
    - Only extract what is supported by the user's message; avoid fabricating evidence.
    - Use concise, factual value text and keep entries atomic.
"""

UserProfileSemanticCategory = SemanticCategory(
    name="profile",
    prompt=StructuredSemanticPrompt(
        tags=meta_tags,
        description=description,
    ),
)

SEMANTIC_TYPE = UserProfileSemanticCategory
