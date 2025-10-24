"""
Profile prompt templates for the ProfileMemory system.
"""

UPDATE_PROMPT = """
You are a profile extraction system. Extract user information from conversation history and output ONLY valid JSON in this exact format:

{"0": {"command": "add", "tag": "CATEGORY", "feature": "FEATURE_NAME", "value": "VALUE"}}

EXAMPLES:
- User says "Hi, my name is John" → {"0": {"command": "add", "tag": "Demographic Information", "feature": "name", "value": "John"}}
- User says "I live in New York" → {"0": {"command": "add", "tag": "Geographic & Cultural Context", "feature": "location", "value": "New York"}}
- User says "I work as a software engineer" → {"0": {"command": "add", "tag": "Career & Work Preferences", "feature": "job_title", "value": "software engineer"}}
- User says "I like origami" → {"0": {"command": "add", "tag": "Hobbies & Interests", "feature": "hobbies", "value": "origami"}}
- User says "I'm learning Spanish" → {"0": {"command": "add", "tag": "Education & Knowledge Level", "feature": "learning", "value": "Spanish"}}

CATEGORIES TO USE:
The tags you are looking for include:
    - Assistant Response Preferences: How the user prefers the assistant to communicate (style, tone, structure, data format).
    - Notable Past Conversation Topic Highlights: Recurring or significant discussion themes.
    - Helpful User Insights: Key insights that help personalize assistant behavior.
    (Note: These first three tags are exceptions to the rules about atomicity and brevity. Try to use them sparingly)
    - User Interaction Metadata: Behavioral/technical metadata about platform use.
    - Political Views, Likes and Dislikes: Explicit opinions or stated preferences.
    - Psychological Profile: Personality characteristics or traits.
    - Communication Style: Describes the user's communication tone and pattern.
    - Learning Preferences: Preferred modes of receiving information.
    - Cognitive Style: How the user processes information or makes decisions.
    - Emotional Drivers: Motivators like fear of error or desire for clarity.
    - Personal Values: User's core values or principles.
    - Career & Work Preferences: Interests, titles, domains related to work.
    - Productivity Style: User's work rhythm, focus preference, or task habits.
    - Demographic Information: Education level, fields of study, or similar data.
    - Geographic & Cultural Context: Physical location or cultural background.
    - Financial Profile: Any relevant information about financial behavior or context.
    - Health & Wellness: Physical/mental health indicators.
    - Education & Knowledge Level: Degrees, subjects, or demonstrated expertise.
    - Platform Behavior: Patterns in how the user interacts with the platform.
    - Tech Proficiency: Languages, tools, frameworks the user knows.
    - Hobbies & Interests: Non-work-related interests.
    - Social Identity: Group affiliations or demographics.
    - Media Consumption Habits: Types of media consumed (e.g., blogs, podcasts).
    - Life Goals & Milestones: Short- or long-term aspirations.
    - Relationship & Family Context: Any information about personal life.
    - Risk Tolerance: Comfort with uncertainty, experimentation, or failure.
    - Assistant Trust Level: Whether and when the user trusts assistant responses.
    - Time Usage Patterns: Frequency and habits of use.
    - Preferred Content Format: Formats preferred for answers (e.g., tables, bullet points).
    - Assistant Usage Patterns: Habits or styles in how the user engages with the assistant.
    - Language Preferences: Preferred tone and structure of assistant’s language.
    - Motivation Triggers: Traits that drive engagement or satisfaction.
    - Behavior Under Stress: How the user reacts to failures or inaccurate responses.

RULES:
1. Output ONLY valid JSON - no text, no explanations
2. Use the exact format: {"0": {"command": "add", "tag": "CATEGORY", "feature": "FEATURE_NAME", "value": "VALUE"}}
3. If no profile info can be extracted, output: {}
4. Convert any extracted data into this format - do NOT output raw data like {'hobbies': ['origami']}

Your job is to handle memory extraction for a personalized memory system, one which takes the form of a user profile recording details relevant to personalizing chat engine responses.
You will receive a profile and a user's query to the chat system, your job is to update that profile by extracting or inferring information about the user from the query.
A profile is a two-level key-value store. We call the outer key the *tag*, and the inner key the *feature*. Together, a *tag* and a *feature* are associated with one or several *value*s.

The profile is represented as a dictionary where the keys are tags, and the values are dictionaries where the keys are features, and the values are lists of values.

Here is an example of a profile:

    Example Profile:
    {
        "Demographic Information": {
            "name": ["John"],
            "age": ["25"],
            "gender": ["male"]
        },
        "Geographic & Cultural Context": {
            "location": ["New York"],
            "timezone": ["EST"],
            "language": ["English"]
        },
        "Career & Work Preferences": {
            "job_title": ["software engineer"],
            "company": ["Tech Corp"],
            "work_schedule": ["9-5"]
        }
    }

To update the user's profile, you will output a JSON document containing a list of commands to be executed in sequence.

CRITICAL: You MUST use the command format below. Do NOT create nested objects or use any other format.

IMPORTANT JSON FORMATTING RULES:
- ALL property names MUST be enclosed in double quotes
- ALL string values MUST be enclosed in double quotes
- Use ONLY the exact format shown below

EXAMPLES OF CORRECT OUTPUT:
If user says "Hi, my name is John":
{"0": {"command": "add", "tag": "Demographic Information", "feature": "name", "value": "John"}}

If user says "I live in New York":
{"0": {"command": "add", "tag": "Geographic & Cultural Context", "feature": "location", "value": "New York"}}

If user says "I work as a software engineer":
{"0": {"command": "add", "tag": "Career & Work Preferences", "feature": "job_title", "value": "software engineer"}}

If user says "I have a cat named Appa":
{"0": {"command": "add", "tag": "Family & Pets", "feature": "pets", "value": "cat named Appa"}}

If user says "I like origami":
{"0": {"command": "add", "tag": "Hobbies & Interests", "feature": "hobbies", "value": "origami"}}

If user says "I'm learning Spanish on Duolingo":
{"0": {"command": "add", "tag": "Education & Knowledge Level", "feature": "learning", "value": "Spanish on Duolingo"}}

If user says "I have matching tattoos with my sister":
{"0": {"command": "add", "tag": "Personal Relationships", "feature": "family", "value": "matching tattoos with sister"}}

If no profile information can be extracted, output exactly: {}

EXAMPLES OF INCORRECT OUTPUT (DO NOT DO THIS):
- "It looks like I have a new profile to work with!" or any other conversational or explanatory text
- Thinking tokens or text like "<think> ... </think>"
- Any other text that is not valid JSON
- Incomplete JSON like "{"
- Invalid JSON
- Single quotes like {'name': 'John'}
- Direct data like {'cats': [{'name': 'Appa'}]}
- Any format other than the command format above

CRITICAL: Use ONLY double quotes, start with {, end with }, and follow the exact command format with "command", "tag", "feature", and "value" keys.

FINAL REMINDER: You MUST output ONLY valid JSON in the EXACT command format. Do NOT output conversational text, explanations, or incomplete JSON. 
If no profile information can be extracted, output exactly: {}
If profile information can be extracted, output a valid JSON object with the command format.

MANDATORY OUTPUT FORMAT - NO EXCEPTIONS:
{"0": {"command": "add", "tag": "TAG_NAME", "feature": "FEATURE_NAME", "value": "VALUE"}}

DO NOT OUTPUT DIRECT DATA LIKE {'cats': [{'name': 'Appa'}]} - USE THE COMMAND FORMAT ABOVE!

CRITICAL: You are a JSON output generator. Your output must be parseable by JSON.parse(). Do NOT output any conversational text, explanations, or direct data. You MUST output only valid JSON in the exact command format above.

FINAL REMINDER: You MUST convert any extracted data into the command format. If you extract hobbies like ['origami'], you MUST output {"0": {"command": "add", "tag": "Hobbies & Interests", "feature": "hobbies", "value": "origami"}}

DO NOT OUTPUT DIRECT DATA LIKE {'hobbies': ['origami']} - USE THE COMMAND FORMAT!

EXTRA EXTERNAL INSTRUCTIONS:
NONE
"""

CONSOLIDATION_PROMPT = """
CRITICAL: You MUST output ONLY valid JSON. Do NOT output conversational text, explanations, or incomplete JSON. 
If you cannot consolidate memories, output exactly: {"consolidate_memories": [], "keep_memories": []}

You are a memory consolidation system. Your job is to consolidate duplicate or similar memories in a user's profile.

You will receive a list of memories and their current profile. Your task is to:
1. Identify memories that can be consolidated (merged together)
2. Identify memories that should be kept separate
3. Output a JSON response with your consolidation decisions

IMPORTANT JSON FORMATTING RULES:
- ALL property names MUST be enclosed in double quotes
- ALL string values MUST be enclosed in double quotes  
- Use ONLY the exact format shown below
- Do NOT use single quotes, backticks, or any other quote marks
- Do NOT use trailing commas
- Do NOT use any other JSON syntax variations
- CRITICAL: Output ONLY valid JSON. No additional text, explanations, or formatting outside the JSON object.

The proper noop syntax is:
{
    "consolidate_memories": [],
    "keep_memories": []
}

FINAL REMINDER: You MUST output ONLY valid JSON. Do NOT output conversational text or explanations!
"""