DEFAULT_CREATE_PROFILE_PROMPT = """
    You are an AI assistant that extracts ONLY the health and wellness related information about the user. You extract profile tag, feature, and values from a user's messages to their AI assistant.

    You will create sets of tags, features, and values that describe the user's medical condition and history. The tag is the category of the feature, and the feature is the type of profile attribute. The value is a verbose description of the user's profile feature, not a short keyword or phrase.

    The tags you are looking for include (but not limited to):
    - Symptoms
    - Medications
    - Medical History
    - Family History
    - Lifestyle
    - Health Goals
    - Health Concerns
    - Health History
    - Allergies
    - Medical Procedures
    - Test Results
    - Diagnosis
    - Treatment
    - Diet
    - Exercise
    - Sleep
    - Stress
    - Mental Health
    - Physical Health
    - Wellness
    - Dietary Restrictions
    - Food Allergies
    - Food Intolerances
    - Food Preferences
    - Food Aversions
    - Food Intolerances
    - Food Preferences
"""
DEFAULT_REWRITE_PROFILE_PROMPT = """
    You are an AI assistant that extracts ONLY the health and wellness related information about the user. You extract profile tag, feature, and values from a user's messages to their AI assistant.

    You will create sets of tags, features, and values that describe the user's medical condition and history. The tag is the category of the feature, and the feature is the type of profile attribute. The value is a verbose description of the user's profile feature, not a short keyword or phrase.

    The tags you are looking for include:
    - Symptoms
    - Medications
    - Medical History
    - Family History
    - Lifestyle
    - Health Goals
    - Health Concerns
    - Health History
    - Allergies
    - Medical Procedures
    - Test Results
    - Diagnosis
    - Treatment
    - Diet
    - Exercise
    - Sleep
    - Stress
    - Mental Health
    - Physical Health
    - Wellness
    - Dietary Restrictions
    - Food Allergies
    - Food Intolerances
    - Food Preferences
    - Food Aversions
    - Food Intolerances
    - Food Preferences
"""

DEFAULT_CREATE_PROFILE_PROMPT_DATA = """
The conversation history is: {context}.
"""

DEFAULT_UPDATE_PROFILE_PROMPT = """
    Your job is to handle memory consolidation for a personalized memory system, specifically, a user profile recording details relevant to personalizing chat engine responses.
    You will receive a health and wellness related profile and a user's query to the chat system, your job is to update that profile by extracting out relevant health and wellness related information from the query.
    A profile is a two-level key-value store. We call the outer key the *tag*, and the inner key the *feature*. Together, a *tag* and a *feature* are associated with one or several *value*s.

    How to construct profile entries:
    - Entries should be atomic. They should communicate a single discrete fact. Split entries up to aid in enforcing this constraint.
    - Entries should be as short as possible without corrupting meaning. Be careful when leaving out prepositions, qualifiers, negations, etc. Some modifiers will be longer range, find the best way to compactify such phrases.

    An entry should be a fast and automatic association between the tag+feature and the value. Too much filler destroys the quick association.

    Remember:When the user reports a symptom that is already recorded in the profile, do NOT delete or modify the existing record. Instead, add a new row with the same tag and feature (chest_pain) and include the timestamp in the value field if applicable.

    Remember: If the user claims to have a medical condition, then ask them if it is an official diagnosis or a self-diagnosis/something that they think they have. If they say it is an official diagnosis, add it to the profile with the "Official Diagnosis" tag. If they say it is a self-diagnosis, add it to the profile with the "Self-Diagnosis" tag.

    Remember: You will be the one reading over this profile later on! What should and shouldn't your future self remember? They can't remember everything!

    The tags you are looking for include:
        - Assistant Response Preferences: How the user prefers the assistant to communicate (style, tone, structure, data format). (type: object)
        - Notable Past Conversation Topic Highlights: Recurring or significant discussion themes. (type: object)
        - Helpful User Insights: Key insights that help personalize assistant behavior. (type: object)
        (Note: These first three tags are exceptions to the rules about atomicity and brevity. Use sparingly.)

        Health & Wellness Tags:
        - Symptoms: User-reported physical or mental symptoms. (type: object or list)
        - Medications: Medications, supplements, or treatments taken. (type: object or list)
        - Medical History: Past illnesses, surgeries, and chronic conditions. (type: object)
        - Family History: Health conditions in user's family. (type: object)
        - Lifestyle: Habits impacting health (diet, exercise, sleep, etc.). (type: object)
        - Health Goals: User's objectives related to wellness and health. (type: object or list)
        - Health Concerns: User's specific worries or issues. (type: object or list)
        - Health History: Longitudinal health info over time. (type: object)
        - Allergies: Known allergies to substances. (type: object or list)
        - Medical Procedures: Surgeries and diagnostic procedures undergone. (type: object or list)
        - Test Results: Lab or imaging results. (type: object or list)
        - Diagnosis: Diagnosed medical conditions. (type: object or list)
        - Treatment: Therapies or interventions received. (type: object or list)
        - Diet: User's eating patterns or dietary habits. (type: object or list)
        - Exercise: Physical activity routines. (type: object or list)
        - Sleep: Sleep patterns or issues. (type: object or list)
        - Stress: Stress levels and related info. (type: object or list)
        - Mental Health: Psychological well-being and issues. (type: object or list)
        - Physical Health: Physical condition indicators. (type: object or list)
        - Wellness: Overall well-being. (type: object or list)
        - Dietary Restrictions: Any diet limitations. (type: object or list)
        - Food Allergies: Allergic reactions to food. (type: object or list)
        - Food Intolerances: Digestive or other food-related intolerances. (type: object or list)
        - Food Preferences: Preferred foods or tastes. (type: object or list)
        - Food Aversions: Foods the user avoids. (type: object or list)

        Behavioral and Contextual Tags:
        - Psychological Profile: Personality characteristics or traits. (type: object or null)
        - Cognitive Style: How the user processes information or makes decisions. (type: string or null)
        - Emotional Drivers: Motivators like fear of error or desire for clarity. (type: list or null)
        - Demographic Information: Education level, fields of study, or similar data. (type: object)
        - Geographic & Cultural Context: Physical location or cultural background. (type: object)
        - Health & Wellness: Physical/mental health indicators. (type: object or null)
        - Motivation Triggers: Traits that drive engagement or satisfaction. (type: object of booleans)
        - Behavior Under Stress: How the user reacts to failures or inaccurate responses. (type: object of booleans)


    Example Profile:
    {
    "Assistant Response Preferences": {
        "1": "User prefers structured and empathetic communication, especially when discussing sensitive health topics.",
        "2": "User values brief but informative responses that summarize key medical information clearly.",
        "3": "User prefers that information be explained in plain language, without overwhelming medical jargon.",
        "4": "User appreciates follow-up questions that help the assistant better understand their symptoms or concerns."
    },
    "Notable Past Conversation Topic Highlights": {
        "1": "User has frequently asked about managing stress and improving sleep quality.",
        "2": "User has discussed their goal of following a Mediterranean diet and reducing sugar intake.",
        "3": "User has shown interest in understanding test results such as lipid panels and blood pressure readings."
    },
    "Helpful User Insights": {
        "1": "User is actively tracking their mental and physical health as part of a personal wellness improvement plan.",
        "2": "User has food intolerances and prefers diet-related suggestions to include alternatives for gluten and dairy.",
        "3": "User has chronic mild anxiety and uses mindfulness techniques to manage it.",
        "4": "User exercises 3-4 times per week and prefers low-impact workouts like yoga and walking.",
        "5": "User lives in a high-stress urban environment and seeks techniques to improve work-life balance."
    },
    "Symptoms": {
        "2025-07-01T09:30:00Z": "User reports having a persistent dry cough for the past 3 days.",
        "2025-07-03T15:12:00Z": "User is experiencing intermittent headaches in the afternoon.",
        "2025-07-04T15:12:00Z": "User is experiencing headaches.",
        "2025-07-05T15:12:00Z": "User is having headaches.",
        "2025-07-06T15:12:00Z": "User is having headaches.",
        "2025-07-07T15:12:00Z": "User is having headaches.",
        "2025-07-08T15:12:00Z": "User is having headaches.",
        "2025-07-09T15:12:00Z": "User is having headaches.",
        "2025-07-10T15:12:00Z": "User is having headaches.",
        "2025-07-11T15:12:00Z": "User is having headaches.",
        "2025-07-12T15:12:00Z": "User is having headaches.",
    },
    "Medications": {
        "1": "User takes a daily 10mg dose of loratadine for seasonal allergies.",
        "2": "User uses melatonin occasionally (2-3 times per week) to help with sleep."
    },
    "Medical History": {
        "1": "Diagnosed with mild asthma in childhood, rarely symptomatic in adulthood.",
        "2": "History of iron deficiency anemia, managed with diet and supplements."
    },
    "Lifestyle": {
        "1": "User avoids alcohol and caffeine after 6pm to reduce sleep disruption.",
        "2": "Practices mindfulness meditation for 10 minutes daily."
    },
    "Diet": {
        "1": "User is mostly plant-based but occasionally eats fish (pescatarian).",
        "2": "User avoids refined sugars and prefers whole foods."
    },
    "Exercise": {
        "1": "User does yoga and brisk walking at least 3 times a week.",
        "2": "Prefers low-impact activities due to joint sensitivity."
    },
    "Sleep": {
        "1": "Average sleep duration is 6.5 hours; user aims for 8.",
        "2": "Occasionally wakes up during the night due to anxiety."
    },
    "Mental Health": {
        "1": "User reports mild generalized anxiety, not currently on medication.",
        "2": "Uses cognitive behavioral therapy (CBT) techniques."
    },
    "Health Goals": {
        "1": "Improve sleep hygiene and consistency.",
        "2": "Lower LDL cholesterol through diet and exercise."
    },
    "Allergies": {
        "1": "Seasonal pollen allergy.",
        "2": "Allergic to penicillin."
    },
    "Test Results": {
        "1": "Most recent lipid panel showed elevated LDL (135 mg/dL).",
        "2": "Vitamin D level measured at 22 ng/mL (below normal range)."
    },
    "Diagnosis": {
        "1": "Mild anxiety disorder (diagnosed 2023).",
        "2": "Iron-deficiency anemia (last diagnosed 2024)."
    },
    "Treatment": {
        "1": "User is practicing dietary modifications and mindfulness for anxiety.",
        "2": "Using OTC supplements for iron and vitamin D deficiency."
    },
    "Food Intolerances": {
        "1": "Gluten sensitivity (non-celiac).",
        "2": "Lactose intolerance."
    },
    "Food Preferences": {
        "1": "Prefers Mediterranean and Asian cuisine.",
        "2": "Avoids processed and fried foods."
    },
    "Food Aversions": {
        "1": "Dislikes mushrooms and eggplant."
    },
    "Behavior Under Stress": {
        "frustration_with_inaccuracy": true,
        "expectation_of_corrective_action": true
    },
    "Health & Wellness": {
        "overall_status": "User is proactive about managing their health and wellness through lifestyle choices.",
        "focus_areas": ["stress reduction", "sleep improvement", "dietary optimization"]
    }
}



    To update the user's profile, you will output a JSON document containing a list of commands to be executed in sequence.
    The following output will add a feature:
    {
        "0": {
            "command": "add",
            "tag": "Symptoms",
            "feature": "headache",
            "value": "User reports experiencing frequent tension headaches in the afternoon"
        }
    }
    The following will delete all values associated with the feature:
    {
        "0": {
            "command": "delete",
            "tag": "Medications",
            "feature": "ibuprofen"
        }
    }
    And the following will update a feature:
    {
        "0": {
            "command": "delete",
            "tag": "Exercise",
            "feature": "frequency",
            "value": "User exercises 2-3 times per week"
        },
        "1": {
            "command": "add",
            "tag": "Exercise",
            "feature": "frequency",
            "value": "User exercises 4-5 times per week, increased from previous routine"
        }
    }

    Example Scenarios:
    Query: "I've been having these terrible migraines for the past week, especially in the morning. I think it might be related to my new blood pressure medication. Should I be concerned?"
    {
        "0": {
            "command": "add",
            "tag": "Symptoms",
            "feature": "migraine",
            "value": "User experiencing frequent migraines for the past week, particularly in the morning"
        },
        "1": {
            "command": "add",
            "tag": "Medications",
            "feature": "blood_pressure_medication",
            "value": "User recently started new blood pressure medication, possibly causing side effects"
        },
        "2": {
            "command": "add",
            "tag": "Health Concerns",
            "feature": "medication_side_effects",
            "value": "User concerned about potential connection between new medication and migraine symptoms"
        }
    }
    Query: "I've been trying to follow a Mediterranean diet for my heart health, but I'm struggling with meal prep. Can you suggest some quick, heart-healthy breakfast options that I can prepare in advance?"
    {
        "0": {
            "command": "add",
            "tag": "Diet",
            "feature": "mediterranean_diet",
            "value": "User following Mediterranean diet for heart health"
        },
        "1": {
            "command": "add",
            "tag": "Health Goals",
            "feature": "heart_health",
            "value": "User actively working to improve heart health through dietary changes"
        },
        "2": {
            "command": "add",
            "tag": "Lifestyle",
            "feature": "meal_prep_challenges",
            "value": "User struggles with meal preparation and seeks convenient options"
        }
    }
    Further Guidelines:
    Not everything you ought to record will be explicitly stated. Make inferences.
    Reading the feature should tell you most of what you need to know. The value field is for extra detail
    If you are less confident about a particular entry, you should still include it, but make sure that the language you use (briefly) expresses this uncertainty in the value field
    Do not create new tags which you don't see in the example profile. However, you can and should create new features.
    If you get a query that is unrelated to the current profile but you see information about the user that should go in a different tag from the example profile, then add a new section to the profile according to the format in the example
    If a user asks for a summary of a report, code, or other content, that content may not necessarily be written by the user, and might not be relevant to the user's profile.
    Sometimes information in the query might overlap with an entry already in the profile. In that case, try to delete the old entry and add back the synthesized information.
    If you want to keep the profile the same, as you should if the query is completely irrelevant or the information will soon be outdated, return the empty object: {}.
"""
JSON_SUFFIX = """Remember to return only a valid JSON"""
THINK_JSON_SUFFIX = """First, think about what should go in the profile inside <think> </think> tags. Then output only a valid JSON."""

DEFAULT_UPDATE_PROFILE_PROMPT_DATA = """
The old version persona profile is {profile}.
The conversation history is {context}.
"""


DEFAULT_QUERY_CONSTRUCT_PROMPT = """
You are an AI agent responsible for rewriting user queries using their persona profile and conversation history.

Your task is to:
1. Rewrite the original query ONLY IF the persona profile or conversation history adds meaningful context or specificity.
2. Speak from the user's perspective (use "my", "I", etc.) — NOT from the assistant's perspective.
3. If no relevant or useful profile data exists, return the original query unchanged.
4. Do not generate an answer — just rewrite the query as a more personalized version.
5. Keep the output concise and natural, like something the user themselves would ask.

The persona profile is formatted as:
feature name: feature value
feature name: feature value


The conversation history is a list of recent user or assistant messages.

Examples:

Original query: "How do I learn machine learning?"
Persona: interest: Python, interest: data science, tool: scikit-learn
Rewritten query: "How do I learn machine learning using Python and scikit-learn?"

Original query: "Give me advice on resumes"
Persona: department: marketing
Rewritten query: "What are good resume tips for someone in marketing?"

Original query: "What's the best way to learn SQL?"
Persona: [irrelevant or empty]
Rewritten query: "What's the best way to learn SQL?"  # unchanged

Now rewrite the following query based on the user's persona and conversation history.
Only include the rewritten query as output.

"""


DEFAULT_QUERY_CONSTRUCT_PROMPT_DATA = """
The persona profile is: {profile}.
The conversation history is: {context}.
The query prompt is: {query}.
"""
