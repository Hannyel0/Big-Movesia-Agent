"""
Sophisticated 3-Tier Memory Architecture for LangGraph Agent.

Exports:
- MemoryManager: Unified memory coordinator
- WorkingMemory: Short-term task context
- Episode, EpisodeStatus: Episodic memory types
- Pattern, SemanticFact: Semantic memory types
- Integration utilities: get_memory_manager, inject_memory_context, etc.

Memory is integrated directly into graph nodes (classify.py, plan.py, act.py, assess.py, finish.py).
All integration utilities are in working.py.
"""

from react_agent.memory.working_memory import WorkingMemory
from react_agent.memory.episodic import EpisodicMemory, Episode, EpisodeStatus
from react_agent.memory.semantic import SemanticMemory, Pattern, SemanticFact
from react_agent.memory.manager import MemoryManager
from react_agent.memory.entity_extractor import extract_entities_simple, debug_extraction
from react_agent.memory.working import (
    get_memory_manager,
    extract_entities_from_state,
    extract_topics_from_state,
    extract_entities_from_request,
    extract_topics_from_request,
    format_memory_context,
    inject_memory_into_prompt,
    get_memory_insights,
    initialize_memory_system,
    _extract_thread_id_from_config,
)

__all__ = [
    # Core memory types
    "WorkingMemory",
    "EpisodicMemory",
    "Episode",
    "EpisodeStatus",
    "SemanticMemory",
    "Pattern",
    "SemanticFact",
    "MemoryManager",
    # Extraction utilities
    "extract_entities_simple",
    "debug_extraction",
    # Integration utilities (from working.py)
    "get_memory_manager",
    "extract_entities_from_state",
    "extract_topics_from_state",
    "extract_entities_from_request",
    "extract_topics_from_request",
    "format_memory_context",
    "inject_memory_into_prompt",
    "get_memory_insights",
    "initialize_memory_system",
    "_extract_thread_id_from_config",
]
