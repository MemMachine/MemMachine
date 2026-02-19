"""Default prompt templates for included sample domains."""

from memmachine_server.semantic_memory.semantic_model import SemanticCategory
from memmachine_server.server.prompt.coding_style_prompt import (
    CodingStyleSemanticCategory,
)
from memmachine_server.server.prompt.crm_prompt import CrmSemanticCategory
from memmachine_server.server.prompt.financial_analyst_prompt import (
    FinancialAnalystSemanticCategory,
)
from memmachine_server.server.prompt.health_assistant_prompt import (
    HealthAssistantSemanticCategory,
)
from memmachine_server.server.prompt.profile_prompt import UserProfileSemanticCategory
from memmachine_server.server.prompt.writing_assistant_prompt import (
    WritingAssistantSemanticCategory,
)

PREDEFINED_SEMANTIC_CATEGORIES: dict[str, SemanticCategory] = {
    "profile_prompt": UserProfileSemanticCategory,
    "coding_prompt": CodingStyleSemanticCategory,
    "writing_assistant_prompt": WritingAssistantSemanticCategory,
    "financial_analyst_prompt": FinancialAnalystSemanticCategory,
    "health_assistant_prompt": HealthAssistantSemanticCategory,
    "crm_prompt": CrmSemanticCategory,
}
