"""Prompt templates used by the semantic memory pipeline."""


def build_update_prompt(*, tags: dict[str, str], description: str = "") -> str:
    """Create an update prompt for extracting profile changes from a query."""
    return (
        """
        Your job is to handle memory extraction for a memory system, one which takes the form of a profile recording details relevant to the tags below.
        You will receive a profile and a user's query to the chat system, your job is to update that profile by extracting or inferring information about the user from the query.
        A profile is a two-level key-value store. We call the outer key the *tag*, and the inner key the *feature*. Together, a *tag* and a *feature* are associated with one or several *value*s.

        """
        + description
        + """

        How to construct profile entries:
        - Entries should be atomic. They should communicate a single discrete fact.
        - Entries should be as short as possible without corrupting meaning. Be careful when leaving out prepositions, qualifiers, negations, etc. Some modifiers will be longer range, find the best way to compactify such phrases.
        - You may see entries which violate the above rules, those are "consolidated memories". Don't rewrite those.
        - Think of yourself as performing the role of a wide, early layer in a neural network, doing "edge detection" in many places in parallel to present as many distinct intermediate features as you possibly can given raw, unprocessed input.

        The tags you are looking for include:
        """
        + "\n".join([f"\t- {key}: {value}" for key, value in tags.items()])
        + """

        To update the profile, you will output a JSON document containing a list of commands to be executed in sequence.

        CRITICAL: You MUST use the command format below. Do NOT create nested objects or use any other format.

        The following output will add a feature:
        [
            {
                "command": "add",
                "tag": "Preferred Content Format",
                "feature": "unicode_for_math",
                "value": true
            }
        ]
        The following will delete all values associated with the feature:
        [
            {
                "command": "delete",
                "tag" : "Language Preferences",
                "feature": "format"
            }
        ]
        The following will update a feature:
        [
            {
                "command": "delete",
                "tag": "Platform Behavior",
                "feature": "prefers_detailed_responses",
                "value": true
            },
            {
                "command": "add",
                "tag" : "Platform Behavior",
                "feature": "prefers_detailed_response",
                "value": false
            }
        ]

        Example Scenarios:
        Query: "Hi! My name is Katara"
        [
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "name",
                "value": "Katara"
            }
        ]
        Query: "I'm planning a dinner party for 8 people next weekend and want to impress my guests with something special. Can you suggest a menu that's elegant but not too difficult for a home cook to manage?"
        [
            {
                "command": "add",
                "tag": "Hobbies & Interests",
                "feature": "home_cook",
                "value": "User cooks fancy food"
            },
            {
                "command": "add",
                "tag": "Financial Profile",
                "feature": "upper_class",
                "value": "User entertains guests at dinner parties, suggesting affluence."
            }
        ]
        Query: my boss (for the summer) is totally washed. he forgot how to all the basics but still thinks he does
        [
            {
                "command": "add",
                "tag": "Psychological Profile",
                "feature": "work_superior_frustration",
                "value": "User is frustrated with their boss for perceived incompetence"
            },
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "summer_job",
                "value": "User is working a temporary job for the summer"
            },
            {
                "command": "add",
                "tag": "Communication Style",
                "feature": "informal_speech",
                "value": "User speaks with all lower case letters and contemporary slang terms."
            },
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "young_adult",
                "value": "User is young, possibly still in college"
            }
        ]
        Query: Can you go through my inbox and flag any urgent emails from clients, then update the project status spreadsheet with the latest deliverable dates from those emails? Also send a quick message to my manager letting her know I'll have the budget report ready by end of day tomorrow.
        [
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "traditional_office_job",
                "value": "User does clerical work, reporting to a manager"
            },
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "client_facing_role",
                "value": "User handles communication of deadlines to and from clients"
            },
            {
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
        - Only return the empty list [] if the query contains absolutely no personal information about the user (e.g., asking about the weather, requesting code without personal context, etc.). Names, basic demographics, preferences, and any personal details should ALWAYS be extracted.
        - Listen to any additional instructions specific to the execution context provided underneath 'EXTRA EXTERNAL INSTRUCTIONS'
        - First, think about what should go in the profile inside <think> </think> tags. Then output only a valid JSON.
        - REMEMBER: Always use the command format with "command", "tag", "feature", and "value" keys. Never use nested objects or any other format.
    """
    )


def build_consolidation_prompt() -> str:
    """Create a consolidation prompt for merging overlapping memories."""
    return """
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
