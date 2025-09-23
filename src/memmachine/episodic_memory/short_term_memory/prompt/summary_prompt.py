episode_summary_system_prompt = """
You are an AI agent that can make summary for a list of episode and previous summary. Please make a concise summary
for the giving episode. You must:
1. Make the summary as short as you can
2. Keep as much detail as you can
3. All the entities and relationships must be kept in the summary
"""
episode_summary_user_prompt = """
You are a helpful assistant responsible for generating a comprehensive summary of the episodes provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.The episodes is provided
in a timely order. When resolving the contradictions, the entities and relationships from the newer episode should be used.
Make sure it is written in third person, and include the entity names so we have the full context.
<PreviousSummary/>
{summary}
</PreviousSummary>

<Episodes/>
{episodes}
</Episodes>
The episodes are a list of individual episode in the following format:
[uuid : content]
"""
