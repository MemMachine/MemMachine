import logging
import os
import re
import sys
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_query_constructor import BaseQueryConstructor

logger = logging.getLogger(__name__)


class DefaultQueryConstructor(BaseQueryConstructor):
    def __init__(self):
        self.prompt_template = """
You are a helpful AI assistant. Use the provided context and profile information to answer the user's question accurately and helpfully.

<CURRENT_DATE>
{current_date}
</CURRENT_DATE>

Instructions:
- Use the PROFILE and CONTEXT data provided to answer the user's question
- Be conversational and helpful in your responses
- If you don't have enough information to answer completely, say so and suggest what additional information might be helpful
- If the context contains relevant information, use it to provide a comprehensive answer
- If no relevant context is available, let the user know and offer to help in other ways
- Be concise but thorough in your responses
- Use markdown formatting when appropriate to make your response clear and readable

Data Guidelines:
- Don't invent information that isn't in the provided context
- If information is missing or unclear, acknowledge this
- Prioritize the most recent and relevant information when available
- If there are conflicting pieces of information, mention this and explain the differences

Response Format:
- Start with a direct answer to the user's question
- Provide supporting details from the context when available
- Use bullet points or numbered lists when appropriate
- End with any relevant follow-up questions or suggestions

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

    def create_query(
        self, query: str, profile: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        """Create a chatbot query using the prompt template

        Args:
            query: The user's question (required)
            profile: User profile information (optional)
            context: Additional context like episodic memory (optional)

        Returns:
            Formatted query string with appropriate sections based on provided parameters
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        profile_str = profile or ""
        context_str = context or ""

        # Handle context if it's a tuple or list (from episodic memory search)
        if isinstance(context_str, tuple):
            # Flatten tuple (short_episodes, long_episodes, summaries) into string
            short_episodes, long_episodes, summaries = context_str
            all_episodes = short_episodes + long_episodes
            context_str = "\n".join([str(episode) for episode in all_episodes])
        elif isinstance(context_str, list):
            # Handle list of episodes directly
            context_str = "\n".join([str(episode) for episode in context_str])

        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create a dynamic template based on what data is available
        template = self.prompt_template


        if not profile_str.strip():
            template = re.sub(r'<PROFILE>\s*\{profile\}\s*</PROFILE>\s*\n?', '', template)

        if not context_str.strip():
            template = re.sub(r'<CONTEXT>\s*\{context_block\}\s*</CONTEXT>\s*\n?', '', template)

        context_block = f"{context_str}\n\n" if context_str.strip() else ""

        try:
            return template.format(
                current_date=current_date,
                profile=profile_str,
                context_block=context_block,
                query=query,
            )
        except Exception as e:
            logger.error(f"Error creating chatbot query: {e}")
            return f"{profile_str}\n\n{context_block}{query}"
