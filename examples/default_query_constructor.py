import logging
import os
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
        self, profile: Optional[str], context: Optional[str], query: str
    ) -> str:
        """Create a general chatbot query using the prompt template"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
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
            logger.error(f"Error creating chatbot query: {e}")
            # Fallback to simple format
            return f"{profile_str}\n\n{context_block}{query}"
