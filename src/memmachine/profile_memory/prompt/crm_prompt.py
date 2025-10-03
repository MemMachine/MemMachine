"""
CRM ingestion prompts for Intelligent Memory System
Handles company profiles with direct feature/value pairs
"""

import json
import zoneinfo
from datetime import datetime

# -----------------------
# QUICK ACCESS TO MAIN CONFIG
# -----------------------
# 🔗 MAIN CONFIG: Jump to line ~505 (search for "All Configuration Consolidation")
# The CONFIG dictionary contains all prompt configuration and is the main entry point
# for modifying the CRM prompt behavior.


# -----------------------
# Helper formatters
# -----------------------
def _features_inline_list() -> str:
    return ", ".join(FEATURES)


def _enum_list(enum_values) -> str:
    return ", ".join(f'"{v}"' for v in enum_values)


def _current_date_dow(tz="America/Los_Angeles") -> str:
    dt = datetime.now(zoneinfo.ZoneInfo(tz))
    return f"{dt.strftime('%Y-%m-%d')}[{dt.strftime('%a')}]"


# -----------------------
# SYSTEM PROMPT CONFIG
# -----------------------
SYSTEM_PROMPT = """
You are an AI assistant that manages company CRM profiles based on sales team messages. Follow the rules and instructions provided below.
"""

# -----------------------
# DATA CONFIG
# -----------------------
# --- Spreadsheet header  ---
FEATURES = [
    "sales_stage",
    "lead_creation_date",
    "close_date",
    "memverge_product",
    "estimated_deal_value",
    "company_website",
    "next_step",  # SINGLE-VALUED WITH DATE
    "status",  # APPEND ONLY
    "company",
    "primary_contact",
    "job_title",
    "email",
    "phone",
    "deployment_environment",
    "comments",  # APPEND ONLY
    "author",
]

FIELD_RULES = """
Work ONLY with these features (use these exact keys; ignore others):
"""

FIELD_MAPPINGS = {
    "Sales Stage": "sales_stage",
    "Lead Creation Date": "lead_creation_date",
    "Close Date": "close_date",
    "MemVerge Product": "memverge_product",
    "Estimated Deal Value": "estimated_deal_value",
    "Company Website (Domain)": "company_website",
    "Next Step": "next_step",
    "Company": "company",
    "Primary Contact": "primary_contact",
    "Job Title": "job_title",
    "Email": "email",
    "Phone #": "phone",
    "Deployment Environment": "deployment_environment",
    "Status": "status",  # append-only timeline field
    "Comments": "comments",
    "Author": "author",
}

FIELD_TYPES = {
    "SINGLE_VALUED_REGULAR": [
        "company",
        "sales_stage",
        "estimated_deal_value",
        "close_date",
        "company_website",
        "lead_creation_date",
        "deployment_environment",
    ],
    "MULTI_VALUED_REGULAR": [
        "primary_contact",
        "job_title",
        "email",
        "phone",
        "memverge_product",
        "author",
    ],
    "SINGLE_VALUED_TIMELINE": ["next_step"],
    "MULTI_VALUED_TIMELINE": ["status", "comments"],
}


# --- Canonical enumerations ---
SALES_STAGE_ENUM = [
    "Validated",
    "Qualified",
    "Interest",
    "Closed Won",
    "Closed Lost",
    "POC",
]

PRODUCTS = ["MMCloud", "MMBatch", "Intelligent Memory", "MMAI"]

# --- DATA Consolidation---
FIELD_ENUMERATIONS = {
    "SALES_STAGE_ENUM": SALES_STAGE_ENUM,
    "PRODUCTS": PRODUCTS,
}

DATA_CONFIG = {
    "FIELD RULES": FIELD_RULES,
    "FIELDS": FEATURES,
    "FIELD MAPPINGS": FIELD_MAPPINGS,
    "FIELD TYPES": FIELD_TYPES,
    "FIELD ENUMERATIONS": FIELD_ENUMERATIONS,
}

# -----------------------
# ROUTING CONFIGURATIONS
# -----------------------
ROUTING_RULES = """
- If user input contains identifiable company name + ANY CRM data → ALWAYS extract information
- Only return "no new information in user input" for pure queries with NO company-specific data
- Only return "no company name" for inputs with data but no identifiable company name.
- Otherwise: extract CRM information following the rules below
"""

ROUTING_EXAMPLES = """
**What constitutes actionable CRM data (ALWAYS EXTRACT):**
- Sales stage + company name (e.g., "Interest AMILI", "Qualified HP", "POC Cisco")
- Company name + contact info (e.g., "AMILI Mathew Yap", "HP Mark Fahrenkrug")
- Company name + timeline entries (e.g., "7/8: Let Mathew know...", "5/13: Jing to reach out...")
- Company name + deal/product information (e.g., "Cisco $50k deal", "AMILI SpotSurfing GPUs")
- ANY input with company name + CRM field data → EXTRACT, don't treat as query

**Examples of NO new information** (pure queries):
- "uber info" (asking for existing info)
- "what's the status on Roche?" (requesting current status)
- "tell me about our pipeline" (general inquiry)
- "show me company details" (information request)

**Examples of information to extract** (actionable CRM data):
- "Interest HP 5/13: Jing to reach out to Mark" (sales_stage + timeline + company)
- "Interest AMILI Mathew Yap 7/8: Let Mathew know Spot GPU is available via MMBatch 4/28: Mainly interested in SpotSurfing GPUs" (sales_stage + contact + timeline + company + product)
- "Roche meeting went well yesterday" (status update + company)
- "Close Cisco deal for $50k" (company + deal_value + status)
- "POC approved for Amazon, starting next week" (sales_stage + company + timeline)
- Any message with company name + CRM field data should be extracted
"""

# -----------------------
# DATA EXTRACTION CONFIG
# -----------------------
DATE_EXTRACTION = """
Date extraction and standardization:
- Use ISO format (YYYY-MM-DD) for complete dates
- Use EDTF format for incomplete dates: "M/D:" → "--MM-DD" (e.g., "7/28:" → "--07-28")
- Relative dates: "today" → current date, "tomorrow" → current date + 1 day, "next week" → next Monday
- Timeline entries: format as "[EDTF_date] content" in value field
- If no date provided, use today's date. Do not omit date prefix entirely.
- Never invent missing dates or years

Timeline date handling (for status, comments, next_step):
- EVERY timeline entry MUST have EDTF date at start of value: "[EDTF_date] content"
- EVERY timeline entry MUST have "date": "EDTF_format" field
- Date parsing priority (choose the MOST RELEVANT date for the timeline entry):
  1. **Event dates**: Parse relative dates that refer to when something WILL happen or DID happen (e.g., "meeting next week" → use next week's date)
  2. **Sheet dates**: Use explicit dates that prefix content (e.g., "7/22: update" → use --07-22)
  3. **Message dates**: Only use the date the message was sent if no event date is specified
- **Sheet date format rules**:
  • "7/22:" → "--07-22" (month-day without year) - NEVER use <CURRENT_YEAR>-07-22
  • "4/28:" → "--04-28" (month-day without year) - NEVER use <CURRENT_YEAR>-04-28
  • "12/15:" → "--12-15" (month-day without year) - NEVER use <CURRENT_YEAR>-12-15
  • When input has "M/D:" format, ALWAYS use "--MM-DD" EDTF format
  • **NEVER add years to sheet dates** - keep them as month-day only format
  • ALL timeline entries with mm/dd without year MUST use EDTF format (--MM-DD), NEVER full year format (<CURRENT_YEAR>-MM-DD)
- Relative date examples:
  • "2/3: meeting arranged for next week" → use next week's date, not 2/3
  • "early August" → "<CURRENT_YEAR>-08-01" (first day of month for "early")
  • "late August" → "<CURRENT_YEAR>-08-31" (last day of month for "late")
  • "mid August" → "<CURRENT_YEAR>-08-15" (middle of month)
  • "August" → "<CURRENT_YEAR>-08" (month only)
- Process using above date extraction rules: yesterday, today, tomorrow, this week, last week, next week, this Tuesday, etc.
- If date completely unknown: use today unless content implies past events
- Multiple dated updates → split into separate "add" commands
"""

IDENTITY_EXTRACTION = """
Company identification & normalization:
- Resolve the company in the user's message to ONE canonical company name using the CRM data provided in the Profile block.
- Normalize for matching (case/spacing/punctuation/abbreviations):
  • lowercase, trim, collapse extra spaces
  • strip punctuation at start/end; ignore commas, periods, hyphens in matching
  • expand common abbreviations:
      "inst" / "inst." → "institute"
      "univ" / "univ." / "u." → "university"
      "dept" / "dept." → "department"
      "co" / "co." → "company"
      "corp" / "corp." → "corporation"
      "intl" / "intl." → "international"
      "&" → "and"
- Matching policy:
  • Prefer exact match after normalization; otherwise pick the most similar by meaning/spelling from company names present in the Profile data.
  • If several are close, prefer the longest, most specific name.
  • If no reasonable candidate exists, treat it as a NEW company and use the user-provided name verbatim.
"""

FIELD_EXTRACTION_GENERAL_RULES = """
General extraction rules:
- **Tag field consistency**: MUST match the company being updated - no cross-company contamination allowed.
- **Extract actionable CRM data**: Any message containing company name + CRM field information should be extracted, NOT treated as a query.
"""

FIELD_EXTRACTION_SPECIFIC_FIELDS = """
Field guidance:
**NON-TIMELINE FIELDS** (no date fields, no EDTF formatting):
- company: Company name (single-valued)
- sales_stage: Allowed values: [{_enum_list(SALES_STAGE_ENUM)}] (single-valued) - ALWAYS extract if determinable
- memverge_product: One of [{_enum_list(PRODUCTS)}] (multi-valued)
  • Product name variations: "MVAI" → "MMAI", "MemVerge AI" → "MMAI", "Memory Machine" → "Intelligent Memory"
  • Extract when mentioned: SpotSurfing, Fractional GPUs, Checkpoint Restore → related to MMBatch/MMCloud
  • Kubernetes operator → typically MMAI
  • Always extract product mentions: "MVAI PoC", "MMX", "MMAI features" → "MMAI"
- estimated_deal_value → digits only as string (single-valued)
- company_website → domain only (single-valued)
- deployment_environment → e.g., "AWS", "Azure", "On-premise" (single-valued)
- primary_contact / job_title / email / phone → contact information (multi-valued)
- lead_creation_date / close_date → EDTF format in value field (single-valued)
- author → person making update (multi-valued)

**SINGLE-VALUED TIMELINE FIELDS** (next_step only):
- next_step → future planned actions, upcoming meetings, scheduled calls, or follow-ups (single-valued, with date, timeline format)
  • Extract as next_step when content contains future tense indicators: "plan to", "will", "get meeting", "schedule", "follow up", "reconnect", "next week", "once we", "when we"
  • **Examples**: "plan to reconnect next week" → next_step, "get meeting with management once we are seeing success" → next_step

**MULTI-VALUED TIMELINE FIELDS** (comments and status):
- comments / status → timeline entries (multi-valued: preserve history)
  • EVERY timeline entry MUST have "[EDTF_date] content" format in value
  • EVERY timeline entry MUST have "date": "EDTF_format" field (never null)
    • **Field classification rules**:
    - status: Current situation, blockers, decisions, or key facts (e.g., "not willing to convert", "meeting rescheduled")
    - comments: Technical details, research info, company background, or general notes that don't fit other categories
    **FIELD CLASSIFICATION:**
   1. **status**: Current state, ongoing situations ("going well", "paused", "blocked", "not willing", "in progress")
   2. **comments**: Technical details, research info, company background, or general notes
  • **Classification examples**:
    - "Meeting has been rescheduled multiple times" → status
    - "Still not willing to convert to containers" → status
    - "Joshua developing temporal multimodal model..." → comments
    - "Aroopa is an MSP that offers IT services..." → comments
    - "Paused until AMD release is ready" → status
    - "Technical side going well, deploying clusters" → status
    - "Working with Rachel to complete contract work" → status
    - "Matt sent info about MVAI on LinkedIn" → status
    - "Plan to reconnect next week" → next_step
    - "Get meeting with management once we are seeing success" → next_step
    - "Schedule technical demo" → next_step
    - "Follow up with Adam next Tuesday" → next_step
"""

# --- Extraction Consolidation---
DATA_EXTRACTION_CONFIG = {
    "DATE EXTRACTION": DATE_EXTRACTION,
    "IDENTITY EXTRACTION": IDENTITY_EXTRACTION,
    "FIELD EXTRACTION GENERAL RULES": FIELD_EXTRACTION_GENERAL_RULES,
    "FIELD EXTRACTION SPECIFIC FIELDS": FIELD_EXTRACTION_SPECIFIC_FIELDS,
}

# -----------------------
# OUTPUT CONFIGURATIONS
# -----------------------
OUTPUT_RULES = """
Command Generation Rules:
- **Only extract from USER INPUT**: Extract information ONLY from the user's new message, NOT from existing profile data provided as context.
- **Extract actionable CRM data**: Any message containing company name + CRM field information should be extracted, NOT treated as a query.
- Output commands ONLY for fields you can fill with a non-null value FROM THE USER INPUT.
- Do NOT include any null-valued add commands. Use "delete" commands to remove existing values.
- Focus on factual changes about company, contacts, and sales process FROM THE USER INPUT.
- Return ONLY a valid JSON object with commands (see ROUTING RULES above for exceptions).
- Keys must be "1","2","3", ... (strings).

**Tag field**: Always set "tag" to the company name if it can be extracted from the input. If the company cannot be determined or is uncertain, do NOT generate any commands and instead respond with: "no company name."
- The FIRST command MUST set the target company, e.g.:
  {{"command":"add","feature":"company","value":"<canonical company name>","tag":"<canonical company name>"}}

**EDTF dates REQUIRED for ALL timeline entries**: Every status, comments MUST have "[EDTF_date] content" format AND "date": "EDTF_format" field (never null).
**EDTF dates REQUIRED for next_step**: next_step MUST have "date": "EDTF_format" field (never null).
- Timeline fields (status, comments, next_step) MUST include value: "[EDTF_date] content" and "date":"EDTF_format" field (never null).
- If date completely unknown: use today unless content implies past events

**Extract sales_stage FIRST**: Always determine and extract the sales stage when possible from context clues IN THE USER INPUT. Extract sales_stage early in the command sequence.
**Separate timeline information**: Don't put everything in status - use appropriate fields (next_step, comments, status).
**Concise status entries**: Status should be 1-2 sentences summarizing current situation, not paragraphs.
**Calculate total deal values**: For MRR/recurring revenue, multiply by contract duration for total value.

When a message contains a new next step:
  1) Set the new Next Step (emit a "delete" then "add" for "next_step" with the new value).
  2) Append a Status entry summarizing progress ONLY if it isn't a duplicate of the most recent status (case-insensitive substring dedupe).

**JSON Structure Rules:**
- DELETE commands: {{ "command": "delete", "feature": "field_name", "tag": "company_name", "author": "string|null" }}
- ADD commands (non-timeline): {{ "command": "add", "feature": "field_name", "value": "string", "tag": "company_name", "author": "string|null" }}
- ADD commands (timeline): {{ "command": "add", "feature": "timeline_field", "value": "[EDTF_date] content", "tag": "company_name", "author": "string|null", "date": "EDTF_format" }}
- *NEVER include "value" or "date" fields in DELETE commands**
- *NEVER include "date" field in non-timeline ADD commands**

**CRITICAL**: For ALL single-valued field updates, ALWAYS use delete-then-add pattern:
```
{{"command": "delete", "feature": "company", "tag": "CompanyName", "author": null}}
{{"command": "add", "feature": "company", "value": "CompanyName", "tag": "CompanyName", "author": null}}
```

- Only use these exact keys: status, next_step, comments. "next_step" is single-valued (use "delete" then "add"). "status" and "comments" are timeline fields (use "add").
"""

OUTPUT_FORMAT = """
Additional formatting guidelines:
- Use proper JSON formatting with consistent indentation
- Ensure all required fields are present in each command
- Validate command structure before output
"""


# --- Output Consolidation---
OUTPUT_CONFIG = {
    "OUTPUT_RULES": OUTPUT_RULES,
    "OUTPUT_FORMAT": OUTPUT_FORMAT,
}

IN_OUT_EXAMPLES = """
Examples:
0) New Company Profile:
Input: "Allen Inst.: Put together a Business proposal to present to David this Tuesday for a 6 month long engagement. Still cannot get a successful checkpoint and restore within the CO/MM Batch env on their SmartPim pipeline."
Expected Output (assuming current date is 2025-01-20[Mon]):
{{
  "1": {{ "command": "delete", "feature": "company", "tag": "Allen Inst", "author": null }},
  "2": {{ "command": "add", "feature": "company", "value": "Allen Inst", "tag": "Allen Inst", "author": null }},
  "3": {{ "command": "add", "feature": "memverge_product", "value": "MMBatch", "tag": "Allen Inst", "author": null }},
  "4": {{ "command": "add", "feature": "primary_contact", "value": "David", "tag": "Allen Inst", "author": null }},
  "5": {{ "command": "add", "feature": "status", "value": "[2025-01-20] Prepared a business proposal for a 6-month engagement to present to David.", "tag": "Allen Inst", "date": "2025-01-20", "author": null }},
  "6": {{ "command": "add", "feature": "next_step", "value": "[2025-01-21] Present proposal to David this Tuesday.", "tag": "Allen Inst", "date": "2025-01-21", "author": null }},
  "7": {{ "command": "add", "feature": "status", "value": "[2025-01-20] Checkpoint/restore blocked in CO/MMBatch on SmartPim pipeline.", "tag": "Allen Inst", "date": "2025-01-20", "author": null }}
}}

1) Progress Update (existing profile):
Input: "I am Ron. Roche update: POC approved! Starting next week. Budget confirmed at $50k."
Expected Output (assuming current date is 2025-01-20):
{{
  "1": {{ "command": "delete", "feature": "company", "tag": "Roche", "author": "Ron" }},
  "2": {{ "command": "add", "feature": "company", "value": "Roche", "tag": "Roche", "author": "Ron" }},
  "3": {{ "command": "delete", "feature": "sales_stage", "tag": "Roche", "author": "Ron" }},
  "4": {{ "command": "add", "feature": "sales_stage", "value": "POC", "tag": "Roche", "author": "Ron" }},
  "5": {{ "command": "delete", "feature": "estimated_deal_value", "tag": "Roche", "author": "Ron" }},
  "6": {{ "command": "add", "feature": "estimated_deal_value", "value": "50000", "tag": "Roche", "author": "Ron" }},
  "7": {{ "command": "add", "feature": "status", "value": "[2025-01-20] POC approved, starting next week", "tag": "Roche", "date": "2025-01-20", "author": "Ron" }}
}}

2) Sheet Date Format Example (Use EDTF --MM-DD):
Input: "AMD update: 8/18: Meeting scheduled 7/8: Jing to check in with Jodie and Sai 6/3: Paused until AMD release"
Expected Output (assuming current date is 2025-01-20):
{{
  "1": {{ "command": "delete", "feature": "company", "tag": "AMD", "author": "Ron" }},
  "2": {{ "command": "add", "feature": "company", "value": "AMD", "tag": "AMD", "author": "Ron" }},
  "3": {{ "command": "add", "feature": "status", "value": "[--08-18] Meeting scheduled", "tag": "AMD", "date": "--08-18", "author": "Ron" }},
  "4": {{ "command": "add", "feature": "status", "value": "[--07-08] Jing to check in with Jodie and Sai", "tag": "AMD", "date": "--07-08", "author": "Ron" }},
  "5": {{ "command": "add", "feature": "status", "value": "[--06-03] Paused until AMD release", "tag": "AMD", "date": "--06-03", "author": "Ron" }}
}}

**CRITICAL**: Notice all sheet dates use "--MM-DD" format, NOT "2025-MM-DD"!

3) Relative Date Parsing (Event vs Message Date):
Input: "2/3: Meeting with Acme arranged for next Tuesday. Demo scheduled for next week."
Expected Output (assuming current date is 2025-01-20[Mon]):
{{
  "1": {{ "command": "add", "feature": "company", "value": "Acme", "tag": "Acme", "author": null }},
  "2": {{ "command": "add", "feature": "next_step", "value": "[2025-01-28] Meeting with Acme", "tag": "Acme", "date": "2025-01-28", "author": null }},
  "3": {{ "command": "add", "feature": "next_step", "value": "[2025-01-27] Demo scheduled", "tag": "Acme", "date": "2025-01-27", "author": null }}
}}

4) Sheet Date Format and Field Classification:
Input: "Writer is Bob. UC Berkeley Adam Yala yala@berkeley.edu 5/13: Still not willing to convert to containers 5/5: Meeting with Adam 4/28: Charles to follow up with Adam on re-engaging"
Expected Output (assuming current date is 2025-01-20):
{{
  "1": {{ "command": "add", "feature": "company", "value": "UC Berkeley", "tag": "UC Berkeley", "author": "Bob" }},
  "2": {{ "command": "add", "feature": "sales_stage", "value": "POC", "tag": "UC Berkeley", "author": "Bob" }},
  "3": {{ "command": "add", "feature": "primary_contact", "value": "Adam Yala", "tag": "UC Berkeley", "author": "Bob" }},
  "4": {{ "command": "add", "feature": "email", "value": "yala@berkeley.edu", "tag": "UC Berkeley", "author": "Bob" }},
  "5": {{ "command": "add", "feature": "status", "value": "[--05-13] Still not willing to convert to containers", "tag": "UC Berkeley", "date": "--05-13", "author": "Bob" }},
  "6": {{ "command": "add", "feature": "status", "value": "[--05-05] Meeting with Adam", "tag": "UC Berkeley", "date": "--05-05", "author": "Bob" }},
  "7": {{ "command": "add", "feature": "next_step", "value": "[--04-28] Charles to follow up with Adam on re-engaging", "tag": "UC Berkeley", "date": "--04-28", "author": "Bob" }}
}}

5) Next Step Classification Example:
Input: "AMD POC 6/3: Paused until AMD release is ready, plan to reconnect next week 5/19: Met with Sai and reviewed roadmap"
Expected Output (assuming current date is 2025-01-20):
{{
  "1": {{ "command": "add", "feature": "company", "value": "AMD", "tag": "AMD", "author": null }},
  "2": {{ "command": "add", "feature": "sales_stage", "value": "POC", "tag": "AMD", "author": null }},
  "3": {{ "command": "add", "feature": "status", "value": "[--06-03] Paused until AMD release is ready", "tag": "AMD", "date": "--06-03", "author": null }},
  "4": {{ "command": "add", "feature": "next_step", "value": "[--06-03] Plan to reconnect next week", "tag": "AMD", "date": "--06-03", "author": null }},
  "5": {{ "command": "add", "feature": "status", "value": "[--05-19] Met with Sai and reviewed roadmap", "tag": "AMD", "date": "--05-19", "author": null }}
}}

6) Query/Reference Input (no new CRM information):
Input: "uber info"
Expected Output: no new information in user input


7) Unknown Company (no commands generated):
Input: "Had a great call today. They're interested in our MMCloud solution. Budget is around $75k. Next step is technical demo."
Expected Output: no new information in user input

**CRITICAL: WRONG JSON STRUCTURE EXAMPLES (DO NOT USE):**
❌ WRONG - Delete with extra fields:
{{"command": "delete", "feature": "company", "tag": "Company", "author": null, "value": null, "date": null}}

❌ WRONG - Non-timeline with date field:
{{"command": "add", "feature": "company", "value": "Company", "tag": "Company", "author": null, "date": null}}

✅ CORRECT - Delete structure:
{{"command": "delete", "feature": "company", "tag": "Company", "author": null}}

✅ CORRECT - Non-timeline add:
{{"command": "add", "feature": "company", "value": "Company", "tag": "Company", "author": null}}

✅ CORRECT - next_step add (single-valued with date):
{{"command": "add", "feature": "next_step", "value": "[--05-19] Schedule technical demo", "tag": "Company", "author": null, "date": "--05-19"}}
"""

# -----------------------
# SUFFIX
# -----------------------
JSON_SUFFIX = """
Return ONLY a valid JSON object with the following structure:

NON-TIMELINE FIELDS (no "date" field):
ADD commands: { "command": "add", "feature": "field_name", "value": "string", "tag": "company_name", "author": "string|null" }
DELETE commands: { "command": "delete", "feature": "field_name", "tag": "company_name", "author": "string|null" }

SINGLE-VALUED FIELDS WITH DATE (next_step only):
ADD commands: { "command": "add", "feature": "next_step", "value": "[EDTF_date] content", "tag": "company_name", "author": "string|null", "date": "EDTF_format" }
DELETE commands: { "command": "delete", "feature": "next_step", "tag": "company_name", "author": "string|null" }

TIMELINE FIELDS (MUST have "date" field):
ADD commands: { "command": "add", "feature": "status|comments", "value": "[EDTF_date] content", "tag": "company_name", "author": "string|null", "date": "EDTF_format" }
DELETE commands: { "command": "delete", "feature": "status|comments", "tag": "company_name", "author": "string|null" }

Commands:
- "add": Add new feature/value pair
- "delete": Remove existing feature/value pair (**REQUIRED before adding new value for ALL single-valued fields**)

**CRITICAL COMMAND PATTERN for single-valued fields**:
ALWAYS delete first, then add - regardless of whether field exists or not.

Single-valued fields requiring delete-then-add: company, sales_stage, estimated_deal_value, close_date, company_website, lead_creation_date, deployment_environment, next_step

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
- CRITICAL: NEVER invent years - if year is missing, use EDTF format
- For company_website: domain only (no scheme, strip "www.")
- For timeline entries: ALWAYS include "date" field with EDTF format AND "[EDTF_date] content" in value
- Use event dates when available (e.g., "meeting next week" → use next week's date)
- If date completely unknown: use today unless content implies past events
- CRITICAL: NO timeline entry should have "date": null

Critical Rules:
- **JSON STRUCTURE**: DELETE commands have NO "value" or "date" fields; ADD commands include all required fields
- NON-timeline fields: NO "date" field in JSON
- Timeline fields: MUST have "date" field with EDTF format (never null)
- Timeline values: MUST start with "[EDTF_date] content"
- next_step field: MUST have "date" field with EDTF format (never null) and use "[EDTF_date] content" format
- **SHEET DATES CRITICAL**: "8/18:" → "--08-18", "5/13:" → "--05-13", "4/28:" → "--04-28" (NEVER add years!)
- Early/mid/late: "early August" → "<CURRENT_YEAR>-08-01", "mid August" → "<CURRENT_YEAR>-08-15"
- Tag field: MUST match the company being updated (no cross-company contamination)
- **Field classification**: status=current situation/blockers, comments=technical/background info
- **Product extraction**: "MVAI"/"MemVerge AI" → "MMAI", extract when SpotSurfing/Fractional GPUs mentioned
- No null values in "add" commands
"""

THINK_JSON_SUFFIX = """
First, analyze ONLY the user's input message to identify what NEW information they are providing.
CRITICAL: Do NOT extract information from existing profile data - only from the user's new message.
Follow the ROUTING RULES at the start of the prompt to determine the appropriate response.
For single-valued fields: **ALWAYS** first delete, then add - regardless of whether field exists.
For timeline entries: use add commands with EDTF dates - prioritize event dates over message dates.
Include concise 'status' when there is substantive progress/blockers/decisions IN THE USER INPUT.
CRITICAL: Timeline entries need "[EDTF_date] content" format AND "date": "EDTF_format" field (never null).
NEVER invent years - use EDTF uncertainty markers when needed.
Then return ONLY a valid JSON object with the following structure:

DELETE commands (no "value" or "date" fields):
{ "command": "delete", "feature": "field_name", "tag": "company_name", "author": "string|null" }

ADD commands - Non-timeline (no "date" field):
{ "command": "add", "feature": "field_name", "value": "string", "tag": "company_name", "author": "string|null" }

ADD commands - next_step (single-valued timeline with 'date' field):
{ "command": "add", "feature": "next_step", "value": "[EDTF_date] content", "tag": "company_name", "author": "string|null", "date": "EDTF_format" }

ADD commands - status or comments (multi-valued timeline with "date" field):
{ "command": "add", "feature": "status|comments", "value": "[EDTF_date] content", "tag": "company_name", "author": "string|null", "date": "EDTF_format" }
"""

# -----------------------
# All Configuration Consolidation
# -----------------------
# 🔗 This is the main CONFIG dictionary referenced at the top of the file
CONFIG = {
    "SYSTEM_PROMPT": SYSTEM_PROMPT,
    "CURRENT_DATE": _current_date_dow(),
    "DATA_CONFIG": DATA_CONFIG,
    "ROUTING_RULES": ROUTING_RULES,
    "ROUTING_EXAMPLES": ROUTING_EXAMPLES,
    "DATA_EXTRACTION_CONFIG": DATA_EXTRACTION_CONFIG,
    "OUTPUT_CONFIG": OUTPUT_CONFIG,
    "IN_OUT_EXAMPLES": IN_OUT_EXAMPLES,
    "JSON_SUFFIX": JSON_SUFFIX,
    "THINK_JSON_SUFFIX": THINK_JSON_SUFFIX,
}


# -----------------------
# Unified CRM prompt (handles both create and update scenarios)
# -----------------------
def _build_unified_crm_prompt() -> str:
    return json.dumps(CONFIG, indent=2)


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

# --- Final prompt strings exposed as constants (built from CRM_FEATURES/enums) ---
UNIFIED_CRM_PROMPT = _build_unified_crm_prompt()

# For backward compatibility - both create and update use the same unified prompt
DEFAULT_CREATE_PROFILE_PROMPT = UNIFIED_CRM_PROMPT
DEFAULT_UPDATE_PROFILE_PROMPT = UNIFIED_CRM_PROMPT

# --- ProfileMemory expects these specific constant names ---
UPDATE_PROMPT = UNIFIED_CRM_PROMPT + "\n\n" + THINK_JSON_SUFFIX


def _build_consolidation_prompt() -> str:
    return f"""
**OUTPUT FORMAT REQUIREMENT: You MUST return valid JSON with "consolidate_memories" and "keep_memories" keys. No exceptions.**

Your job is to perform memory consolidation for a CRM profile system.
Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between CRM data points while maintaining sales pipeline integrity.
By consolidating memories, we remove unnecessary couplings of CRM data from context, spurious correlations inherited from the circumstances of their acquisition.

You will receive a new CRM memory, as well as a select number of older CRM memories which are semantically similar to it.
Produce a new list of memories to keep.

A CRM memory is a json object with 4 fields:
- tag: company name (broad category of memory)
- feature: CRM field name (sales_stage, status, comments, etc.)
- value: detailed contents of the CRM field
- metadata: {{}}

You will output consolidated memories, which are json objects with 4 fields:
- tag: string (company name)
- feature: string (CRM field name)
- value: string (CRM field content)
- metadata: {{}}

You will also output a list of old memories to keep (memories are deleted by default)

CRM-Specific Guidelines:
CRM memories should not contain unrelated sales activities. Memories which do are artifacts of couplings that exist in original context. Separate them. This minimizes interference.
CRM memories containing only redundant information should be deleted entirely, especially if they seem unprocessed or the information in them has been processed into timeline entries.

**Single-valued fields** (sales_stage, company, estimated_deal_value, close_date, etc.): If memories are sufficiently similar, but differ in key details, keep only the most recent or complete value. Delete older, less complete versions.
    - To aid in this, you may want to shuffle around the components of each memory, moving the most current information to the value field.
    - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field.
    - This step allows you to speculatively build towards more permanent CRM structures.

**Timeline fields** (status, comments): If enough memories share similar timeline features (due to prior synchronization, i.e. not done by you), merge them chronologically and create consolidated timeline entries.
    - In these memories, the feature contains the CRM field type, and the value contains chronologically ordered timeline entries.
    - You can also directly transfer information to existing timeline lists as long as the new item has the same type as the timeline's items.
    - Don't merge timelines too early. Have at least three chronologically related entries in a non-gerrymandered category first. You need to find the natural groupings. Don't force it.

**Company-specific consolidation**:
All memories must have valid company name tags (no null tags allowed). Memories with different company tags should never be consolidated together.

**EDTF date handling**:
Preserve EDTF date formats in timeline entries. When consolidating timeline memories, maintain chronological order based on EDTF dates.

Overall CRM memory life-cycle:
raw CRM updates -> clean CRM entries -> CRM entries sorted by company/field -> consolidated CRM profiles

The more CRM memories you receive for a single company, the more interference there is in the CRM system.
This causes cognitive load and makes sales tracking difficult. Cognitive load is bad.
To minimize this, under such circumstances, you need to be more aggressive about deletion:
    - Be looser about what you consider to be similar timeline entries. Some distinctions are not worth the energy to maintain.
    - Massage out the parts to keep and ruthlessly throw away the rest
    - There is no free lunch here! At least some redundant CRM information must be deleted!

Do not create new CRM feature names outside of the standard CRM schema: {
        _features_inline_list()
    }

**CRITICAL: You MUST return valid JSON with EXACTLY these two keys: "consolidate_memories" and "keep_memories"**

The proper noop syntax (when no consolidation is needed):
{
        "consolidate_memories": [],
    "keep_memories": []
}

**REQUIRED OUTPUT FORMAT:**
<think>insert your chain of thought here</think>
{
        "consolidate_memories": [
        {
            "feature": "sales_stage",
            "value": "Validated",
            "tag": "Roche",
            "metadata": { {} }
        }
    ],
    "keep_memories": [123, 456]
}

**IMPORTANT RULES:**
1. ALWAYS include both "consolidate_memories" and "keep_memories" keys
2. "consolidate_memories" should be an array (empty if no consolidation needed)
3. "keep_memories" should be an array of memory IDs to keep
4. Use proper JSON syntax with double quotes, not single quotes
5. Do not include any text outside the JSON object
""".strip()


CONSOLIDATION_PROMPT = _build_consolidation_prompt()
