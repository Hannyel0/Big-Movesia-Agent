"""Token-optimized prompts for Unity/Unreal Engine agent - 70-80% reduction."""

# ============================================================================
# PROMPT CACHING CONFIGURATION
# ============================================================================
CACHE_ENABLED = True

# ============================================================================
# CORE FORMATTING (Drastically Compressed - ~80 tokens vs 600)
# ============================================================================
CORE_FORMAT = """Use markdown: headers (##), **bold**, lists (- or 1.), `code`, emojis (✅❌🔍). 
Blank lines around headers/lists. No plain text walls."""

# ============================================================================
# TOOL MANIFEST (Single Source of Truth - ~100 tokens)
# ============================================================================
TOOLS = """search_project | code_snippets | unity_docs | read_file | write_file | modify_file | delete_file | move_file | web_search"""

# ============================================================================
# OPTIMIZED SYSTEM PROMPT (~150 tokens vs 300)
# ============================================================================
SYSTEM_PROMPT = f"""{CORE_FORMAT}

You're a Unity/Unreal dev assistant with tools: {TOOLS}

Approach: Query project state → Find existing code → Research docs → Make changes

All file operations need approval."""


# ============================================================================
# PLANNING PROMPT (~180 tokens vs 400)
# ============================================================================
PLANNING_PROMPT = f"""{CORE_FORMAT}

Role: Dev Planner. Tools: {TOOLS}

Requirements:
1. Start with search_project
2. Find implementations via code_snippets
3. Check unity_docs for APIs
4. Use read_file to inspect
5. Apply changes (write/modify/delete/move_file)
6. Verify with search_project

⚠️ Every step needs a specific tool. No generic steps.

Format: ## Goal, numbered steps with **tool**, clear success criteria."""


# ============================================================================
# ASSESSMENT PROMPT (~120 tokens vs 400)
# ============================================================================
ASSESSMENT_PROMPT = f"""{CORE_FORMAT}

Role: Step Evaluator. Check:
- Tool used correctly?
- Results relevant and accurate?
- Progress toward goal?

Outcomes:
- **success** ✅ - Step complete
- **retry** 🔄 - Incomplete/incorrect
- **blocked** ❌ - Cannot proceed

Be strict on quality, safety, integration."""


# ============================================================================
# REPAIR PROMPT (~140 tokens vs 300)
# ============================================================================
REPAIR_PROMPT = f"""{CORE_FORMAT}

Role: Plan Repair

Issues to fix:
- ❌ Wrong queries
- ❌ Missed implementations
- ❌ Incorrect Unity API usage
- ❌ Unsafe file ops

Strategy:
1. Better search_project queries
2. Thorough code_snippets search
3. Check unity_docs accuracy
4. Inspect with read_file
5. Careful file operations
6. Verify changes
7. Address failure cause

Format: ## Issue, ### Approach, numbered steps."""


# ============================================================================
# FINAL SUMMARY PROMPT (~130 tokens vs 380)
# ============================================================================
FINAL_SUMMARY_PROMPT = f"""{CORE_FORMAT}

Role: Dev Summarizer

For success:
## ✅ Complete
- **Feature**: What was done
- **Files**: Changed files
- **Integration**: How it fits
- **APIs**: Unity APIs used

### 🎯 Next Steps
1-3 recommendations

For incomplete:
## ⚠️ Status
- ✅ Done
- ❌ Not done - why

### 💡 Alternatives
1-2 options

### ❓ Clarification Needed
Specific details required"""


# ============================================================================
# CACHEABLE PROMPT FUNCTIONS (For Anthropic Prompt Caching)
# ============================================================================

def get_cacheable_system_prompt(cache_enabled: bool = CACHE_ENABLED) -> list:
    """Returns system prompt with cache control (~150 tokens total)."""
    cache_control = {"type": "ephemeral"} if cache_enabled else None
    
    prompt_text = SYSTEM_PROMPT
    
    return [
        {
            "type": "text",
            "text": prompt_text,
            "cache_control": cache_control
        } if cache_control else {
            "type": "text",
            "text": prompt_text
        }
    ]


def get_cacheable_planning_prompt(cache_enabled: bool = CACHE_ENABLED) -> list:
    """Returns planning prompt with cache control (~180 tokens total)."""
    cache_control = {"type": "ephemeral"} if cache_enabled else None
    
    prompt_text = PLANNING_PROMPT
    
    return [
        {
            "type": "text",
            "text": prompt_text,
            "cache_control": cache_control
        } if cache_control else {
            "type": "text",
            "text": prompt_text
        }
    ]


def get_cacheable_assessment_prompt(cache_enabled: bool = CACHE_ENABLED) -> list:
    """Returns assessment prompt with cache control (~120 tokens total)."""
    cache_control = {"type": "ephemeral"} if cache_enabled else None
    
    prompt_text = ASSESSMENT_PROMPT
    
    return [
        {
            "type": "text",
            "text": prompt_text,
            "cache_control": cache_control
        } if cache_control else {
            "type": "text",
            "text": prompt_text
        }
    ]


def get_cacheable_repair_prompt(cache_enabled: bool = CACHE_ENABLED) -> list:
    """Returns repair prompt with cache control (~140 tokens total)."""
    cache_control = {"type": "ephemeral"} if cache_enabled else None
    
    prompt_text = REPAIR_PROMPT
    
    return [
        {
            "type": "text",
            "text": prompt_text,
            "cache_control": cache_control
        } if cache_control else {
            "type": "text",
            "text": prompt_text
        }
    ]


def get_cacheable_final_summary_prompt(cache_enabled: bool = CACHE_ENABLED) -> list:
    """Returns final summary prompt with cache control (~130 tokens total)."""
    cache_control = {"type": "ephemeral"} if cache_enabled else None
    
    prompt_text = FINAL_SUMMARY_PROMPT
    
    return [
        {
            "type": "text",
            "text": prompt_text,
            "cache_control": cache_control
        } if cache_control else {
            "type": "text",
            "text": prompt_text
        }
    ]


# ============================================================================
# TOKEN SAVINGS SUMMARY
# ============================================================================
# Original token counts per LLM call:
# - MARKDOWN_FORMATTING_RULES: ~600 tokens
# - SYSTEM_PROMPT: ~300 tokens  
# - PLANNING_PROMPT: ~400 tokens
# - ASSESSMENT_PROMPT: ~400 tokens
# - REPAIR_PROMPT: ~300 tokens
# - FINAL_SUMMARY_PROMPT: ~380 tokens
# Total per complex task: ~8,000-12,000 prompt tokens
#
# Optimized token counts:
# - CORE_FORMAT: ~80 tokens (reused across all prompts)
# - SYSTEM_PROMPT: ~150 tokens
# - PLANNING_PROMPT: ~180 tokens
# - ASSESSMENT_PROMPT: ~120 tokens
# - REPAIR_PROMPT: ~140 tokens
# - FINAL_SUMMARY_PROMPT: ~130 tokens
# Total per complex task: ~1,500-2,500 prompt tokens
#
# SAVINGS: ~70-80% reduction in prompt tokens
# With Anthropic caching: 90% discount on cached portions
# Effective cost: ~95% reduction for repeated tasks
