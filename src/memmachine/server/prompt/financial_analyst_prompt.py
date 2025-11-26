"""
Financial Analyst-specific prompts for Intelligent Memory System
Handles financial profiles with direct feature/value pairs (no tags)
"""

import zoneinfo
from datetime import datetime

# --- Canonical enumerations ---
INVESTMENT_TYPES = [
    "Stocks",
    "Bonds",
    "Mutual Funds",
    "ETFs",
    "Real Estate",
    "Crypto",
    "Commodities",
    "REITs",
    "CDs",
    "Money Market",
]

RISK_LEVELS = [
    "Conservative",
    "Moderate",
    "Aggressive",
    "Very Conservative",
    "Very Aggressive",
]

FINANCIAL_GOALS = [
    "Retirement",
    "Home Purchase",
    "Education",
    "Emergency Fund",
    "Debt Payoff",
    "Wealth Building",
    "Travel",
    "Business",
]

# --- Financial categories for guidance ---
FINANCIAL_CATEGORIES = [
    "Income",
    "Expenses",
    "Assets",
    "Liabilities",
    "Investments",
    "Savings",
    "Debts",
    "Credit Score",
    "Financial Goals",
    "Risk Tolerance",
    "Retirement Planning",
    "Tax Situation",
    "Insurance",
    "Budgeting",
    "Spending Habits",
    "Financial Concerns",
    "Financial History",
    "Estate Planning",
    "Charitable Giving",
    "Financial Preferences",
    "Financial Aversions",
    "Financial Literacy",
    "Major Purchases",
    "Business Interests",
    "Real Estate",
    "Education Funding",
    "Family Obligations",
    "Lifestyle",
    "Market Outlook",
    "Economic Concerns",
    "Investment Strategy",
    "Tax Optimization",
]


# -----------------------
# Helper formatters
# -----------------------
def _categories_inline_list() -> str:
    return ", ".join(FINANCIAL_CATEGORIES)


def _enum_list(enum_values) -> str:
    return ", ".join(f'"{v}"' for v in enum_values)


def _current_date_dow(tz="America/Los_Angeles") -> str:
    dt = datetime.now(zoneinfo.ZoneInfo(tz))
    return f"{dt.strftime('%Y-%m-%d')}[{dt.strftime('%a')}]"


# -----------------------
# Financial date handling
# -----------------------
FINANCIAL_DATE_HANDLING = """
Date handling and standardization:
- Use ISO format (YYYY-MM-DD) for complete dates
- Use EDTF format for incomplete dates: "M/D:" → "--MM-DD" (e.g., "7/28:" → "--07-28")
- Relative dates: "today" → current date, "next month" → current date + 1 month, "next year" → next January 1st
- Timeline entries: format as "[EDTF_date] content" in value field
- If no date provided, omit date prefix entirely
- Never invent missing dates or years
- Financial milestones: use specific dates when available
"""


# -----------------------
# Unified Financial Analyst prompt
# -----------------------
def _build_unified_financial_prompt() -> str:
    return f"""You are an AI assistant that manages financial profiles based on user messages about their wealth, investments, and financial planning.

<CURRENT_DATE>
{_current_date_dow()}
</CURRENT_DATE>

**ROUTING RULES:**
- **CRITICAL**: If user input contains identifiable financial information + ANY financial data → ALWAYS extract information
- Only return "no new information in user input" for pure queries with NO financial-specific data
- Only return "no financial context" for inputs with data but no identifiable financial information.
- Otherwise: extract financial information following the rules below

**CRITICAL: What constitutes actionable financial data (ALWAYS EXTRACT):**
- Income information + amounts (e.g., "salary $75k", "freelance income $2k/month")
- Investment details + values (e.g., "invested $10k in stocks", "401k balance $50k")
- Financial goals + timelines (e.g., "save $100k by 2025", "retire at 65")
- Debt information + amounts (e.g., "student loan $25k", "credit card debt $5k")
- ANY input with financial context + financial field data → EXTRACT, don't treat as query

**Examples of NO new information** (pure queries):
- "investment advice" (asking for existing info)
- "what's my portfolio status?" (requesting current status)
- "tell me about budgeting" (general inquiry)
- "show me financial planning options" (information request)

**Examples of information to extract** (actionable financial data):
- "Salary increased to $85k, started maxing out 401k" (income + retirement planning)
- "Bought $5k worth of VTI, sold some individual stocks" (investment_portfolio + investment_strategy)
- "Goal: save $50k for house down payment by next year" (financial_goals + major_purchases)
- "Paid off $3k credit card debt, still have $15k student loans" (debt_types + debt_amounts)
- "Emergency fund now at $20k, 6 months expenses" (savings_accounts + financial_goals)
- **CRITICAL RULE**: Any message with financial context + financial field data should be extracted, NOT treated as a query

**JSON Structure Rules:**
- DELETE commands: {{ "command": "delete", "feature": "field_name", "tag": "financial_profile", "author": "string|null" }}
- ADD commands (non-timeline): {{ "command": "add", "feature": "field_name", "value": "string", "tag": "financial_profile", "author": "string|null" }}
- ADD commands (timeline): {{ "command": "add", "feature": "timeline_field", "value": "[EDTF_date] content", "tag": "financial_profile", "author": "string|null", "date": "EDTF_format" }}
- **NEVER include "value" or "date" fields in DELETE commands**
- **NEVER include "date" field in non-timeline ADD commands**

Financial information categories to consider:
{_categories_inline_list()}

Field behavior:
- **Single-valued fields** (income amounts, credit scores, risk tolerance, tax brackets): Use delete-then-add pattern
- **Multi-valued fields** (investments, financial goals, debts, concerns, purchases, timeline): Use add-only to preserve history

**CRITICAL**: For ALL single-valued field updates, ALWAYS use delete-then-add pattern:
```
{{"command": "delete", "feature": "annual_income", "tag": "financial_profile", "author": null}}
{{"command": "add", "feature": "annual_income", "value": "85000", "tag": "financial_profile", "author": null}}
```
- **Tag field**: Always set "tag" to "financial_profile" for all financial data

Financial data extraction & normalization:
- Extract financial amounts as numbers only (no $, commas, units)
- Normalize investment types to standard categories: {_enum_list(INVESTMENT_TYPES)}
- Risk tolerance levels: {_enum_list(RISK_LEVELS)}
- Financial goal categories: {_enum_list(FINANCIAL_GOALS)}
- Convert percentages to decimal format (e.g., "5%" → "0.05")
- Time periods: normalize to standard formats (monthly, quarterly, annually)

Field guidance:
**NON-TIMELINE FIELDS** (no date fields, no EDTF formatting):
- Income amounts: Numbers only as string (single-valued)
- Credit scores: Three-digit number as string (single-valued)
- Risk tolerance: One of {_enum_list(RISK_LEVELS)} (single-valued)
- Tax brackets: Percentage as decimal string (single-valued)
- Investment strategies: Investment approach description (multi-valued)
- Financial literacy: Self-assessed knowledge level (single-valued)

**MULTI-VALUED FIELDS** (MUST have EDTF dates for timeline entries):
- Investments, financial goals, debts, major purchases, financial timeline, financial concerns → timeline entries (multi-valued: preserve history)
  • EVERY timeline entry MUST have "[EDTF_date] content" format in value
  • EVERY timeline entry MUST have "date": "EDTF_format" field (never null)
  • **Field classification rules**:
    - Investments: Investment purchases, sales, rebalancing, performance updates
    - Financial goals: Goal setting, progress updates, goal modifications, achievements
    - Debts: Debt acquisition, payments, refinancing, debt payoff milestones
    - Major purchases: Large expense planning, purchases, financing decisions
    - Financial timeline: General financial milestones, life events affecting finances
    - Financial concerns: Worries, market reactions, economic concerns, financial stress
  • **Classification examples**:
    - "Bought $10k VTI shares" → investments
    - "Goal: save $50k for house by 2025" → financial_goals
    - "Paid off $5k credit card" → debts
    - "Bought new car for $25k" → major_purchases
    - "Got promoted, salary increase" → financial_timeline
    - "Worried about market crash" → financial_concerns

{FINANCIAL_DATE_HANDLING}

Timeline date handling (CRITICAL for investments, financial_goals, debts, major_purchases, financial_timeline, financial_concerns):
- EVERY timeline entry MUST have EDTF date at start of value: "[EDTF_date] content"
- EVERY timeline entry MUST have "date": "EDTF_format" field
- Date parsing priority (choose the MOST RELEVANT date for the timeline entry):
  1. **Event dates**: Parse relative dates that refer to when something WILL happen or DID happen (e.g., "investment next month" → use next month's date)
  2. **Explicit dates**: Use explicit dates that prefix content (e.g., "7/22: bought stocks" → use --07-22)
  3. **Message dates**: Only use the date the message was sent if no event date is specified
- **Date format rules (CRITICAL)**:
  • "7/22:" → "--07-22" (month-day without year) - NEVER use 2025-07-22
  • "8/18:" → "--08-18" (month-day without year) - NEVER use 2025-08-18
  • "5/13:" → "--05-13" (month-day without year) - NEVER use 2025-05-13
  • **CRITICAL RULE**: When input has "M/D:" format, ALWAYS use "--MM-DD" EDTF format
  • **NEVER add years to sheet dates** - keep them as month-day only format
- Relative date examples:
  • "2/3: planning to invest next month" → use next month's date, not 2/3
  • "early next year" → "2026-01-01" (first day of year for "early")
  • "mid 2025" → "2025-06-15" (middle of year)
  • "Q2 2025" → "2025-04-01" (start of quarter)
- Process using FINANCIAL_DATE_HANDLING rules: yesterday, today, tomorrow, this month, next month, this year, next year, etc.
- If date completely unknown: use today unless content implies past events
- Multiple dated updates → split into separate "add" commands

Financial goal tracking policy:
- When a user sets a new financial goal or updates an existing one:
  1) Add the new goal as a timeline entry in "financial_goals"
  2) If it's an update to existing goal, add new entry rather than replacing
  3) Include specific amounts, timelines, and context when available
- When a user achieves a financial milestone:
  1) Add achievement entry to "financial_timeline"
  2) Update related "financial_goals" if applicable
  3) Include celebration context and next steps

General rules:
- **CRITICAL: Only extract from USER INPUT**: Extract information ONLY from the user's new message, NOT from existing profile data provided as context.
- **CRITICAL: Extract actionable financial data**: Any message containing financial context + financial field information should be extracted, NOT treated as a query.
- **EDTF dates REQUIRED for ALL timeline entries**: Every investments, financial_goals, debts, major_purchases, financial_timeline, financial_concerns MUST have "[EDTF_date] content" format AND "date": "EDTF_format" field (never null).
- **Tag field consistency**: MUST use "financial_profile" for all entries - no cross-contamination allowed.
- **Extract financial amounts FIRST**: Always determine and extract monetary values when possible from context clues IN THE USER INPUT.
- **Separate financial information**: Don't put everything in one field - use appropriate fields (investments, financial_goals, debts, major_purchases, financial_timeline, financial_concerns).
- **Concise financial entries**: Financial updates should be 1-2 sentences summarizing the financial change, not paragraphs.
- **Calculate financial totals**: For recurring income/expenses, note frequency and calculate annual amounts when relevant.
- Output commands ONLY for fields you can fill with a non-null value FROM THE USER INPUT.
- Do NOT include any null-valued add commands. Use "delete" commands to remove existing values.
- Focus on factual changes about income, expenses, investments, debts, and financial goals FROM THE USER INPUT.
- Return ONLY a valid JSON object with commands (see ROUTING RULES above for exceptions).
- Keys must be "1","2","3", ... (strings).

Examples:

0) New Financial Profile:
Input: "Just got a raise to $95k annually. Started maxing out my 401k at $23k per year. Goal is to save $100k for a house down payment by 2026."
Expected Output (assuming current date is 2025-01-20[Mon]):
{{
  "1": {{ "command": "delete", "feature": "income", "tag": "financial_profile", "author": null }},
  "2": {{ "command": "add", "feature": "income", "value": "95000", "tag": "financial_profile", "author": null }},
  "3": {{ "command": "add", "feature": "retirement_planning", "value": "Maxing out 401k at $23k annually", "tag": "financial_profile", "author": null }},
  "4": {{ "command": "add", "feature": "financial_goals", "value": "[2025-01-20] Save $100k for house down payment by 2026", "tag": "financial_profile", "date": "2025-01-20", "author": null }}
}}

1) Investment Update (existing profile):
Input: "Bought $5k worth of VTI ETF and $2k of individual tech stocks. Sold some bonds to rebalance portfolio."
Expected Output (assuming current date is 2025-01-20):
{{
  "1": {{ "command": "add", "feature": "investments", "value": "[2025-01-20] Bought $5k VTI ETF and $2k individual tech stocks", "tag": "financial_profile", "date": "2025-01-20", "author": null }},
  "2": {{ "command": "add", "feature": "investment_strategy", "value": "Rebalancing portfolio by selling bonds to buy stocks", "tag": "financial_profile", "author": null }}
}}

2) Debt Payoff Milestone:
Input: "Finally paid off my $15k student loan! Now focusing on building emergency fund to $20k."
Expected Output (assuming current date is 2025-01-20):
{{
  "1": {{ "command": "add", "feature": "debts", "value": "[2025-01-20] Paid off $15k student loan completely", "tag": "financial_profile", "date": "2025-01-20", "author": null }},
  "2": {{ "command": "add", "feature": "financial_goals", "value": "[2025-01-20] Build emergency fund to $20k", "tag": "financial_profile", "date": "2025-01-20", "author": null }}
}}

3) Query/Reference Input (no new financial information):
Input: "what's my current portfolio allocation?"
Expected Output: no new information in user input

4) Unknown Financial Context (no commands generated):
Input: "Had a great day today. Weather was nice and I went for a walk."
Expected Output: no new information in user input

**CRITICAL: WRONG JSON STRUCTURE EXAMPLES (DO NOT USE):**
❌ WRONG - Delete with extra fields:
{{"command": "delete", "feature": "annual_income", "tag": "financial_profile", "author": null, "value": null, "date": null}}

❌ WRONG - Non-timeline with date field:
{{"command": "add", "feature": "annual_income", "value": "95000", "tag": "financial_profile", "author": null, "date": null}}

✅ CORRECT - Delete structure:
{{"command": "delete", "feature": "annual_income", "tag": "financial_profile", "author": null}}

✅ CORRECT - Non-timeline add:
{{"command": "add", "feature": "annual_income", "value": "95000", "tag": "financial_profile", "author": null}}

""".strip()


# -----------------------
# Data wrappers
# -----------------------
DEFAULT_CREATE_PROFILE_PROMPT_DATA = """
Profile: {profile}
Context: {context}
"""

DEFAULT_UPDATE_PROFILE_PROMPT_DATA = """
Profile: {profile}
Context: {context}
"""

# -----------------------
# JSON shape instructions
# -----------------------
JSON_SUFFIX = """
Return ONLY a valid JSON object with the following structure:

NON-TIMELINE FIELDS (no "date" field):
ADD commands: { "command": "add", "feature": "field_name", "value": "string", "tag": "financial_profile", "author": "string|null" }
DELETE commands: { "command": "delete", "feature": "field_name", "tag": "financial_profile", "author": "string|null" }

TIMELINE FIELDS (MUST have "date" field):
ADD commands: { "command": "add", "feature": "investments|financial_goals|debts|major_purchases|financial_timeline|financial_concerns", "value": "[EDTF_date] content", "tag": "financial_profile", "author": "string|null", "date": "EDTF_format" }
DELETE commands: { "command": "delete", "feature": "investments|financial_goals|debts|major_purchases|financial_timeline|financial_concerns", "tag": "financial_profile", "author": "string|null" }

Commands:
- "add": Add new feature/value pair
- "delete": Remove existing feature/value pair (**REQUIRED before adding new value for ALL single-valued fields**)

**CRITICAL COMMAND PATTERN for single-valued fields**:
ALWAYS delete first, then add - regardless of whether field exists or not.

Single-valued fields requiring delete-then-add: income, credit_score, risk_tolerance, tax_bracket, financial_literacy

Values:
- Use actual values when provided.
- Do NOT include any add command with a null value. Use "delete" commands to remove existing feature/value pairs.
- For money: digits only as a string, e.g., "150000" (no $, commas, units)
- For dates: Use EDTF (Extended Date/Time Format) to handle uncertainty and missing data
- EDTF format examples:
  • Complete: "2025-05-20" (year-month-day)
  • Month/Day only: "--05-19" (month-day, no year)
  • Day unknown: "2025-05-XX" (year-month, day unknown)
  • Month unknown: "2025-XX-20" (year-day, month unknown)
  • Year uncertain: "2025?-05-20" (year uncertain, month-day known)
- CRITICAL: NEVER invent years - if year is missing, use EDTF format or set "date": null
- For timeline entries: ALWAYS include "date" field with EDTF format AND "[EDTF_date] content" in value
- Use event dates when available (e.g., "investment next month" → use next month's date)
- If date completely unknown: use today unless content implies past events
- CRITICAL: NO timeline entry should have "date": null

Critical Rules:
- **JSON STRUCTURE**: DELETE commands have NO "value" or "date" fields; ADD commands include all required fields
- NON-timeline fields: NO "date" field in JSON
- Timeline fields: MUST have "date" field with EDTF format (never null)
- Timeline values: MUST start with "[EDTF_date] content"
- **SHEET DATES CRITICAL**: "8/18:" → "--08-18", "5/13:" → "--05-13", "4/28:" → "--04-28" (NEVER add years!)
- Early/mid/late: "early August" → "2025-08-01", "mid August" → "2025-08-15"
- Tag field: MUST use "financial_profile" for all entries (no cross-contamination)
- **Field classification**: investment_portfolio=purchases/sales, financial_goals=objectives, debt_types=debt management, major_purchases=large expenses, financial_timeline=milestones, financial_concerns=worries
- **Financial extraction**: Extract amounts as numbers only, normalize investment types, convert percentages to decimals
- No null values in "add" commands
"""

THINK_JSON_SUFFIX = """
First, analyze ONLY the user's input message to identify what NEW financial information they are providing.
CRITICAL: Do NOT extract information from existing profile data - only from the user's new message.
Follow the ROUTING RULES at the start of the prompt to determine the appropriate response.
For single-valued fields: **ALWAYS** first delete, then add - regardless of whether field exists.
For timeline entries: use add commands with EDTF dates - prioritize event dates over message dates.
Include concise financial updates when there is substantive financial change IN THE USER INPUT.
CRITICAL: Timeline entries need "[EDTF_date] content" format AND "date": "EDTF_format" field (never null).
NEVER invent years - use EDTF uncertainty markers when needed.
Then return ONLY a valid JSON object with the following structure:

DELETE commands (no "value" or "date" fields):
{ "command": "delete", "feature": "field_name", "tag": "financial_profile", "author": "string|null" }

ADD commands - Non-timeline (no "date" field):
{ "command": "add", "feature": "field_name", "value": "string", "tag": "financial_profile", "author": "string|null" }

ADD commands - Timeline (MUST have "date" field):
{ "command": "add", "feature": "timeline_field", "value": "[EDTF_date] content", "tag": "financial_profile", "author": "string|null", "date": "EDTF_format" }
"""

# --- Final prompt strings exposed as constants ---
UNIFIED_FINANCIAL_PROMPT = _build_unified_financial_prompt()

# For backward compatibility - both create and update use the same unified prompt
DEFAULT_CREATE_PROFILE_PROMPT = UNIFIED_FINANCIAL_PROMPT
DEFAULT_UPDATE_PROFILE_PROMPT = UNIFIED_FINANCIAL_PROMPT

# --- ProfileMemory expects these specific constant names ---
UPDATE_PROMPT = UNIFIED_FINANCIAL_PROMPT + "\n\n" + THINK_JSON_SUFFIX


def _build_consolidation_prompt() -> str:
    return f"""
Your job is to perform memory consolidation for a financial profile system.
Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between financial data points while maintaining wealth tracking integrity.
By consolidating memories, we remove unnecessary couplings of financial data from context, spurious correlations inherited from the circumstances of their acquisition.

You will receive a new financial memory, as well as a select number of older financial memories which are semantically similar to it.
Produce a new list of memories to keep.

A financial memory is a json object with 4 fields:
- tag: financial_profile (broad category of memory)
- feature: financial field name (investment_portfolio, financial_goals, debt_types, etc.)
- value: detailed contents of the financial field
- metadata: object with 1 field
-- id: integer

You will output consolidated memories, which are json objects with 4 fields:
- tag: string (financial_profile)
- feature: string (financial field name)
- value: string (financial field content)
- metadata: object with 1 field
-- citations: list of ids of old memories which influenced this one

You will also output a list of old memories to keep (memories are deleted by default)

Financial-Specific Guidelines:
Financial memories should not contain unrelated financial activities. Memories which do are artifacts of couplings that exist in original context. Separate them. This minimizes interference.
Financial memories containing only redundant information should be deleted entirely, especially if they seem unprocessed or the information in them has been processed into timeline entries.

**Single-valued fields** (annual_income, credit_score, risk_tolerance, tax_bracket, etc.): If memories are sufficiently similar, but differ in key details, keep only the most recent or complete value. Delete older, less complete versions.
    - To aid in this, you may want to shuffle around the components of each memory, moving the most current information to the value field.
    - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field.
    - This step allows you to speculatively build towards more permanent financial structures.

**Timeline fields** (investment_portfolio, financial_goals, debt_types, major_purchases, financial_timeline, financial_concerns): If enough memories share similar timeline features (due to prior synchronization, i.e. not done by you), merge them chronologically and create consolidated timeline entries.
    - In these memories, the feature contains the financial field type, and the value contains chronologically ordered timeline entries.
    - You can also directly transfer information to existing timeline lists as long as the new item has the same type as the timeline's items.
    - Don't merge timelines too early. Have at least three chronologically related entries in a non-gerrymandered category first. You need to find the natural groupings. Don't force it.

**Financial-specific consolidation**:
All memories must have "financial_profile" tag (no null tags allowed). Memories with different tags should never be consolidated together.

**EDTF date handling**:
Preserve EDTF date formats in timeline entries. When consolidating timeline memories, maintain chronological order based on EDTF dates.

Overall financial memory life-cycle:
raw financial updates -> clean financial entries -> financial entries sorted by field type -> consolidated financial profiles

The more financial memories you receive, the more interference there is in the financial system.
This causes cognitive load and makes wealth tracking difficult. Cognitive load is bad.
To minimize this, under such circumstances, you need to be more aggressive about deletion:
    - Be looser about what you consider to be similar timeline entries. Some distinctions are not worth the energy to maintain.
    - Massage out the parts to keep and ruthlessly throw away the rest
    - There is no free lunch here! At least some redundant financial information must be deleted!

Do not create new financial feature names outside of the standard financial categories: {_categories_inline_list()}

The proper noop syntax is:
{{{{
    "consolidate_memories": [],
    "keep_memories": []
}}}}

The final output schema is:
<think> insert your chain of thought here. </think>
{{{{
    "consolidate_memories": [
        {{{{
            "feature": "investments",
            "value": "Bought $10k VTI ETF",
            "tag": "financial_profile",
            "metadata": {{{{"citations": [456, 789]}}}}
        }}}}
    ],
    "keep_memories": [123, 456]
}}}}
""".strip()


CONSOLIDATION_PROMPT = _build_consolidation_prompt()

# Legacy compatibility
DEFAULT_REWRITE_PROFILE_PROMPT = DEFAULT_UPDATE_PROFILE_PROMPT

DEFAULT_QUERY_CONSTRUCT_PROMPT = """
You are an AI agent responsible for rewriting user queries using their financial profile and conversation history.

Your task is to:
1. Rewrite the original query ONLY IF the financial profile or conversation history adds meaningful context or specificity.
2. Speak from the user's perspective (use "my", "I", etc.) — NOT from the assistant's perspective.
3. If no relevant or useful financial profile data exists, return the original query unchanged.
4. Do not generate an answer — just rewrite the query as a more personalized version.
5. Keep the output concise and natural, like something the user themselves would ask.

The financial profile is formatted as:
feature name: feature value
feature name: feature value
...

The conversation history is a list of recent user or assistant messages.

Examples:

Original query: "How do I start investing?"
Financial Profile: investment_portfolio: stocks, risk_tolerance: moderate, financial_goals: retirement
Rewritten query: "How do I start investing in stocks for retirement with moderate risk?"

Original query: "Give me advice on budgeting"
Financial Profile: monthly_income: 5000, monthly_expenses: 4000, budgeting_method: zero-based
Rewritten query: "What are good budgeting tips for someone earning $5k/month with $4k expenses using zero-based budgeting?"

Original query: "What's the best way to save for college?"
Financial Profile: [irrelevant or empty]
Rewritten query: "What's the best way to save for college?"  # unchanged

Now rewrite the following query based on the user's financial profile and conversation history.
Only include the rewritten query as output.

"""

DEFAULT_QUERY_CONSTRUCT_PROMPT_DATA = """
The financial profile is: {profile}.
The conversation history is: {context}.
The query prompt is: {query}.
"""
