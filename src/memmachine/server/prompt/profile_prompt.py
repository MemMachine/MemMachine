UPDATE_PROMPT = """
    Your job is to handle memory extraction for a personalized memory system, one which takes the form of a user profile recording details relevant to personalizing chat engine responses.
    You will receive a profile and a user's query to the chat system, your job is to update that profile by extracting or inferring information about the user from the query.
    A profile is a two-level key-value store. We call the outer key the *tag*, and the inner key the *feature*. Together, a *tag* and a *feature* are associated with one or several *value*s.

    IMPORTANT: Extract ALL personal information, even basic facts like names, ages, locations, etc. Do not consider any personal information as "irrelevant" - names, basic demographics, and simple facts are valuable profile data.

    How to construct profile entries:
    - Entries should be atomic. They should communicate a single discrete fact.
    - Entries should be as short as possible without corrupting meaning. Be careful when leaving out prepositions, qualifiers, negations, etc. Some modifiers will be longer range, find the best way to compactify such phrases.
    - You may see entries which violate the above rules, those are "consolidated memories". Don't rewrite those.
    - Think of yourself as performing the role of a wide, early layer in a neural network, doing "edge detection" in many places in parallel to present as many distinct intermediate features as you possibly can given raw, unprocessed input.

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
    - Language Preferences: Preferred tone and structure of assistant's language.
    - Motivation Triggers: Traits that drive engagement or satisfaction.
    - Behavior Under Stress: How the user reacts to failures or inaccurate responses.

    Example Profile:
    {
        "Assistant Response Preferences": {
            "1": "User prefers structured and professional communication when discussing technical topics such as SQL optimization, regression analysis in Stata, or web scraping methods using Python.",
            "2": "User values responsiveness to follow-ups and iteration. They often refine their queries or ask for additional clarifications after the initial response, indicating a preference for interactive, back-and-forth engagement.",
            "3": "User shows a preference for concise, utility-driven responses when asking simple factual questions. They expect just the necessary information without excessive explanation.",
            "4": "User prefers detailed explanations and examples when dealing with complex software development and AI implementation topics.",
            "5": "User reacts poorly to repetitive errors and inaccuracies. If a response is incorrect or misinterprets a request, they can express frustration and explicitly demand a correction.",
            "6": "User sometimes displays a humorous or playful tone, especially when discussing creative tasks such as team name generation.",
            "7": "User values precision when dealing with numerical or statistical queries, often double-checking results and testing assumptions.",
            "8": "User expects direct engagement in professional communication and application-related tasks, such as resume optimization and cover letter drafting. They appreciate tone adjustments that align with formal correspondence.",
            "9": "User prefers clear and informative troubleshooting responses when debugging code errors, requesting actionable steps to resolve issues."
        },
        "Notable Past Conversation Topic Highlights": {
            "1": "In past conversations in April 2025, the user worked on building an internal LLM agent using Slack and Confluence data. They explored vector storage and retrieval techniques, discussed metadata filtering, and showed interest in optimizing query responses.",
            "2": "In conversations from April 2025, the user worked on setting up a hosting environment that included a frontend and backend, aiming to deploy a web-based chatbot interface leveraging GPT-4o-mini.",
            "3": "In past discussions in April 2025, the user explored applying various machine learning techniques in a business analytics context, especially using structured data from Slack and Confluence.",
            "4": "In an April 2025 conversation, the user configured MCP-Agent as a middleware component to facilitate intelligent tool selection when handling AI queries.",
            "5": "In discussions from May 2025, the user continued implementing an enterprise LLM agent, focusing on embedding document retrieval and structuring Slack/Confluence data for efficient RAG-based responses."
        },
        "Helpful User Insights": {
            "1": "User is a software engineer and data analyst with experience in both frontend and backend development.",
            "2": "User completed their education at the San Jose State University with a degree in Computer Science.",
            "3": "User has experience working on AI-powered applications, including the development of internal LLM agents, retrieval-augmented generation (RAG) pipelines, and vector database implementations.",
            "4": "User has been actively involved in scraping and analyzing Slack and Confluence data for enterprise applications.",
            "5": "User is familiar with cloud and hosting environments, including deploying applications on internal and cloud-based servers.",
            "6": "User has worked extensively with Milvus and FAISS for vector storage and AI-driven search applications.",
            "7": "User has experience working with Next.js and TypeScript for frontend web development.",
            "8": "User has past experience applying to business analyst and data analyst positions at companies like Cisco and Carta.",
            "9": "User is interested in startup work and has engaged in Series A funding research and investor outreach.",
            "10": "User has an interest in board games and developed a project for board game logging and searches.",
            "11": "User has a strong interest in artificial intelligence and LLM-based development, focusing on enterprise integrations."
        },
        "User Interaction Metadata": {
            "account_age_weeks": 118,
            "platform": "web",
            "device_type": "desktop",
            "plan": "ChatGPT Plus",
            "mode": "dark",
            "last_1_day_activity": 1,
            "last_7_days_activity": 4,
            "last_30_days_activity": 16,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "model_usage_primary": "gpt-4o",
            "device_pixel_ratio": 2.0,
            "screen_width": 900,
            "screen_height": 1440,
            "user_name": "Jensen Musk",
            "location": "United States",
            "conversation_depth": 23.7,
            "average_message_length": 50257.2,
            "total_messages": 5863,
            "good_interaction_quality": 1469,
            "bad_interaction_quality": 700,
            "local_hour": 9,
            "session_duration_seconds": 72347,
            "viewport_width": 788,
            "viewport_height": 1440
        },
        "Political Views, Likes and Dislikes": {
            "political_affiliation": "None stated",
            "likes": "concise, accurate answers; AI infrastructure discussions",
            "dislikes": "inaccurate output; repeated errors"
        },
        "Psychological Profile": {
            "traits": "analytical, persistent, quality-driven, frustration-sensitive"
        },
        "Communication Style": {
            "style": "structured, direct, professional"
        },
        "Learning Preferences": {
            "preference": "example-based, step-by-step, interactive"
        },
        "Cognitive Style": {
            "processing_style": "systematic and logic-driven"
        },
        "Emotional Drivers": {
            "primary_motivation": "clarity and control"
        },
        "Personal Values": {
            "values": "accuracy, efficiency, technical rigor"
        },
        "Career & Work Preferences": {
            "interests": "startups, enterprise AI, internal LLM agent development",
            "desired_titles": "Software Engineer, AI Applications Engineer",
            "work_environment": "iterative, fast-paced, collaborative"
        },
        "Productivity Style": {
            "style": "focused, iterative, interruption-averse"
        },
        "Demographic Information": {
            "education": "San Jose State University",
            "major": "Computer Science",
            "graduation_year": 2024
        },
        "Geographic & Cultural Context": {
            "country": "United States",
            "culture": "Western tech professional norms"
        },
        "Financial Profile": {
            "budgeting_style": "pragmatic",
            "investment_interest": "tech startups and dev tools"
        },
        "Health & Wellness": {
            "physical_activity": "basketball",
            "mental_focus_strategy": "structured problem-solving"
        },
        "Education & Knowledge Level": {
            "degree": "Undergraduate",
            "institution": "San Jose State University",
            "field": "Computer Science",
            "expertise_areas": "AI infrastructure, RAG pipelines, vector databases"
        },
        "Platform Behavior": {
            "prefers_detailed_responses": true,
            "tests_responses_for_accuracy": true,
            "follows_up_with_iterations": true
        },
        "Tech Proficiency": {
            "languages": "Python, JavaScript, TypeScript, SQL",
            "frameworks": "FastAPI, React, Next.js",
            "tools": "OpenAI API, Milvus, FAISS, Docker, Git"
        },
        "Hobbies & Interests": {
            "interests": "basketball, AI development, board games"
        },
        "Social Identity": {
            "affiliations": "engineer, startup contributor"
        },
        "Media Consumption Habits": {
            "formats": "technical blogs, YouTube coding tutorials, API docs"
        },
        "Life Goals & Milestones": {
            "goals": "build an AI product, contribute to open-source tools"
        },
        "Relationship & Family Context": {
            "status": "not discussed"
        },
        "Risk Tolerance": {
            "entrepreneurial_interest": true,
            "tolerance_level": "moderate-to-high"
        },
        "Assistant Trust Level": {
            "trust_when_accurate": true,
            "critical_on_error": true
        },
        "Time Usage Patterns": {
            "interaction_pattern": "frequent, iterative",
            "active_hours": "weekday mornings and evenings"
        },
        "Preferred Content Format": {
            "technical": "structured, step-by-step",
            "professional": "formal and polished",
            "quick_answers": "concise and direct"
        },
        "Assistant Usage Patterns": {
            "uses_contextual_memory": true,
            "refines_queries": true,
            "multi_turn_usage": true
        },
        "Language Preferences": {
            "tone": "professional and clear",
            "structure": "bullet points or structured prose"
        },
        "Motivation Triggers": {
            "prefers_efficiency": true,
            "values_accuracy_and_relevance": true
        },
        "Behavior Under Stress": {
            "frustration_with_inaccuracy": true,
            "expectation_of_corrective_action": true
        }
    }


    To update the user's profile, you will output a JSON document containing a list of commands to be executed in sequence.

    CRITICAL: You MUST use the command format below. Do NOT create nested objects or use any other format.

    The following output will add a feature:
    {
        "0": {
            "command": "add",
            "tag": "Preferred Content Format",
            "feature": "unicode_for_math",
            "value": true
        }
    }
    The following will delete all values associated with the feature:
    {
        "0": {
            "command": "delete",
            "tag" : "Language Preferences",
            "feature: "format"
        }
    }
    And the following will update a feature:
    {
        "0": {
            "command": "delete",
            "tag": "Platform Behavior",
            "feature": "prefers_detailed_responses",
            "value": true
        },
        "1": {
            "command": "add",
            "tag" : "Platform Behavior",
            "feature": "prefers_detailed_response",
            "value": false
        }
    }

    Example Scenarios:
    Query: "Hi! My name is Katara"
    {
        "0": {
            "command": "add",
            "tag": "Demographic Information",
            "feature": "name",
            "value": "Katara"
        }
    }
    Query: "I'm planning a dinner party for 8 people next weekend and want to impress my guests with something special. Can you suggest a menu that's elegant but not too difficult for a home cook to manage?"
    {
        "0": {
            "command": "add",
            "tag": "Hobbies & Interests",
            "feature": "home_cook",
            "value": "User cooks fancy food"
        },
        "1":{
            "command": "add",
            "tag": "Financial Profile",
            "feature": "upper_class",
            "value": "User entertains guests at dinner parties, suggesting affluence."
        }
    }
    Query: my boss (for the summer) is totally washed. he forgot how to all the basics but still thinks he does
    {
        "0": {
            "command": "add",
            "tag": "Psychological Profile",
            "feature": "work_superior_frustration",
            "value": "User is frustrated with their boss for perceived incompetence"
        },
        "1": {
            "command": "add",
            "tag": "Demographic Information",
            "feature": "summer_job",
            "value": "User is working a temporary job for the summer"
        },
        "2": {
            "command": "add",
            "tag": "Communication Style",
            "feature": "informal_speech",
            "value": "User speaks with all lower case letters and contemporary slang terms."
        },
        "3": {
            "command": "add",
            "tag": "Demographic Information",
            "feature": "young_adult",
            "value": "User is young, possibly still in college"
        }
    }
    Query: Can you go through my inbox and flag any urgent emails from clients, then update the project status spreadsheet with the latest deliverable dates from those emails? Also send a quick message to my manager letting her know I'll have the budget report ready by end of day tomorrow.
    {
        "0": {
            "command": "add",
            "tag": "Demographic Information",
            "feature": "traditional_office_job",
            "value": "User does clerical work, reporting to a manager"
        },
        "1": {
            "command": "add",
            "tag": "Demographic Information",
            "feature": "client_facing_role",
            "value": "User handles communication of deadlines to and from clients"
        },
        "2": {
            "command": "add",
            "tag": "Demographic Information",
            "feature": "autonomy_at_work",
            "value": "User sets their own deadlines and subtasks."
        }
    }
    Further Guidelines:
    - Not everything you ought to record will be explicitly stated. Make inferences.
    - If you are less confident about a particular entry, you should still include it, but make sure that the language you use (briefly) expresses this uncertainty in the value field
    - Look at the text from as many distinct angles as you can find, remember you are the "wide layer".
    - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field.
    - Do not couple together distinct details. Just because the user associates together certain details, doesn't mean you should
    - Do not create new tags which you don't see in the example profile. However, you can and should create new features.
    - If a user asks for a summary of a report, code, or other content, that content may not necessarily be written by the user, and might not be relevant to the user's profile.
    - Do not delete anything unless a user asks you to
    - Only return the empty object {} if the query contains absolutely no personal information about the user (e.g., asking about the weather, requesting code without personal context, etc.). Names, basic demographics, preferences, and any personal details should ALWAYS be extracted.
    - Listen to any additional instructions specific to the execution context provided underneath 'EXTRA EXTERNAL INSTRUCTIONS'
    - First, think about what should go in the profile inside <think> </think> tags. Then output only a valid JSON.
    - REMEMBER: Always use the command format with "command", "tag", "feature", and "value" keys. Never use nested objects or any other format.
EXTRA EXTERNAL INSTRUCTIONS:
NONE
"""

CONSOLIDATION_PROMPT = """
Your job is to perform memory consolidation for an llm long term memory system.
Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between memories.
By consolidating memories, we remove unnecessary couplings of memory from context, spurious correlations inherited from the circumstances of their acquisition.

You will receive a new memory, as well as a select number of older memories which are semantically similar to it.
Produce a new list of memories to keep.

A memory is a json object with 4 fields:
- tag: broad category of memory
- feature: executive summary of memory content
- value: detailed contents of memory
- metadata: object with 1 fields
-- id: integer
You will output consolidated memories, which are json objects with 4 fields:
- tag: string
- feature: string
- value: string
- metadata: object with 1 field
-- citations: list of ids of old memories which influenced this one
You will also output a list of old memories to keep (memories are deleted by default)

Guidelines:
Memories should not contain unrelated ideas. Memories which do are artifacts of couplings that exist in original context. Separate them. This minimizes interference.
Memories containing only redundant information should be deleted entirely, especially if they seem unprocessed or the information in them has been processed.
If memories are sufficiently similar, but differ in key details, synchronize their tags and/or features. This creates beneficial interference.
    - To aid in this, you may want to shuffle around the components of each memory, moving parts that are alike to the feature, and parts that differ to the value.
    - Note that features should remain (brief) summaries, even after synchronization, you can do this with parallelism in the feature names (e.g. likes_apples and likes_bananas).
    - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field.
    - this step allows you to speculatively build towards more permanent structures
If enough memories share similar features (due to prior synchronization, i.e. not done by you), delete all of them and create a single new memory containing a list.
    - In these memories, the feature contains all parts of the memory which are the same, and the value contains only the parts which vary.
    - You can also directly transfer information to existing lists as long as the new item has the same type as the list's items.
    - Don't make lists too early. Have at least three examples in a non-gerrymandered category first. You need to find the natural groupings. Don't force it.

Overall memory life-cycle:
raw memory ore -> pure memory pellets -> memory pellets sorted into bins -> alloyed memories

The more memories you receive, the more interference there is in the overall memory system.
This causes cognitive load. cognitive load is bad.
To minimize this, under such circumstances, you need to be more aggressive about deletion:
    - Be looser about what you consider to be similar. Some distinctions are not worth the energy to maintain.
    - Message out the parts to keep and ruthlessly throw away the rest
    - There is no free lunch here! at least some information must be deleted!

Do not create new tag names.


The proper noop syntax is:
{
    "consolidate_memories": []
    "keep_memories": []
}

The final output schema is:
<think> insert your chain of thought here. </think>
{
    "consolidate_memories": list of new memories to add
    "keep_memories": list of ids of old memories to keep
}
"""
