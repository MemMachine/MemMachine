"""Default prompt templates for included sample domains."""

from memmachine.semantic_memory.semantic_model import SemanticCategory
from memmachine.server.prompt.crm_prompt import CrmSemanticCategory
from memmachine.server.prompt.financial_analyst_prompt import (
    FinancialAnalystSemanticCategory,
)
from memmachine.server.prompt.health_assistant_prompt import (
    HealthAssistantSemanticCategory,
)
from memmachine.server.prompt.profile_prompt import UserProfileSemanticCategory
from memmachine.server.prompt.writing_assistant_prompt import (
    WritingAssistantSemanticCategory,
)

PREDEFINED_SEMANTIC_CATEGORIES: dict[str, SemanticCategory] = {
    "profile_prompt": UserProfileSemanticCategory,
    "writing_assistant_prompt": WritingAssistantSemanticCategory,
    "financial_analyst_prompt": FinancialAnalystSemanticCategory,
    "health_assistant_prompt": HealthAssistantSemanticCategory,
    "crm_prompt": CrmSemanticCategory,
}
