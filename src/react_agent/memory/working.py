"""
Complete Memory Integration Layer for LangGraph Agent.

✅ FIXED: Now properly configures movesia.db path
✅ FIXED: Tool result summaries work correctly
✅ FIXED: Session-based memory loading
✅ FIXED: Uses LangGraph thread_id for consistent session tracking across runs
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timedelta, UTC
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.utils import get_model, get_message_text

# Avoid circular import
if TYPE_CHECKING:
    from react_agent.state import State
    from react_agent.memory import MemoryManager
else:
    MemoryManager = None

logger = logging.getLogger(__name__)


def _parse_timestamp_safe(timestamp: str) -> datetime:
    """
    Safely parse timestamp string to datetime with UTC.
    
    ✅ FIXED: Handles both datetime objects and ISO strings with timezone consistency.
    """
    try:
        dt = datetime.fromisoformat(timestamp)
        # Ensure it has timezone info
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        # If parsing fails, return very old date so it gets cleaned up
        return datetime(2000, 1, 1, tzinfo=UTC)


# ============================================================================
# GLOBAL MEMORY MANAGER (Singleton Pattern)
# ============================================================================
#
# ✅ CURRENT IMPLEMENTATION: Thread-based session tracking
#    - Uses LangGraph thread_id for consistent session tracking
#    - thread_id is consistent across runs in the same conversation
#    - Automatically detects thread changes and loads appropriate memory
#    - Fallback sources: session_id (Electron) -> thread_id (LangGraph) -> metadata
#
# ALTERNATIVE PATTERNS:
#
# 1. SESSION-BASED MEMORY (For web apps):
#    - Pass session_id in config
#    - Create separate DB per session: f"session_{session_id}.db"
#    - Store memory instance in session state, not global
#
# 2. THREAD-SAFE SINGLETON:
#    - Add threading.Lock around global access
#    - Use thread-local storage for per-thread instances
#
# 3. DEPENDENCY INJECTION:
#    - Remove global variable entirely
#    - Pass MemoryManager instance through state
#    - Create new instance per agent invocation
#
# Example for multi-user (replace get_memory_manager):
# ```python
# def get_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
#     thread_id = config.get("configurable", {}).get("thread_id", "default")
#     db_path = Path(f"./memory/thread_{thread_id}.db")
#     return MemoryManager(db_path=db_path, auto_persist=True)
# ```
#
# ============================================================================

_global_memory: Optional[Any] = None  # MemoryManager instance


def _extract_thread_id_from_config(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Extract thread_id from various sources in config.
    
    Priority order:
    1. configurable.thread_id (LangGraph thread_id - most reliable)
    2. configurable.session_id (Electron-provided session_id)
    3. metadata.thread_id (alternative location)
    4. tags with 'thread:' prefix
    
    Returns:
        thread_id string or None if not found
    """
    if not config:
        return None
    
    # Try configurable dict first
    configurable = config.get("configurable", {})
    
    # Priority 1: LangGraph thread_id (most reliable for conversation continuity)
    thread_id = configurable.get("thread_id")
    if thread_id:
        return thread_id
    
    # Priority 2: Electron-provided session_id
    thread_id = configurable.get("session_id")
    if thread_id:
        return thread_id
    
    # Priority 3: metadata.thread_id
    thread_id = config.get("metadata", {}).get("thread_id")
    if thread_id:
        return thread_id
    
    # Priority 4: Extract from tags (format: "thread:abc123")
    tags = config.get("tags", [])
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("thread:"):
            return tag.split(":", 1)[1]
    
    return None


def _get_movesia_db_path() -> Optional[Path]:
    """
    Get the path to movesia.db from environment or default location.
    
    FIXED: Now properly determines movesia.db location
    """
    # Option 1: Check environment variable (set by Electron app)
    if "MOVESIA_DB_PATH" in os.environ:
        db_path = Path(os.environ["MOVESIA_DB_PATH"])
        if db_path.exists():
            return db_path
    
    # Option 2: Check standard Electron userData path
    if os.name == 'nt':  # Windows
        # C:\Users\<user>\AppData\Roaming\Movesia\movesia.db
        appdata = os.environ.get('APPDATA')
        if appdata:
            db_path = Path(appdata) / "Movesia" / "movesia.db"
            if db_path.exists():
                return db_path
    elif os.name == 'posix':  # macOS/Linux
        # macOS: ~/Library/Application Support/Movesia/movesia.db
        # Linux: ~/.config/Movesia/movesia.db
        home = Path.home()
        
        # Try macOS path first
        db_path = home / "Library" / "Application Support" / "Movesia" / "movesia.db"
        if db_path.exists():
            return db_path
        
        # Try Linux path
        db_path = home / ".config" / "Movesia" / "movesia.db"
        if db_path.exists():
            return db_path
    
    # Option 3: Check if we're running in development with relative path
    dev_path = Path("./agent_memory/movesia.db")
    if dev_path.exists():
        return dev_path
    
    logger.warning(" [Memory] Could not find movesia.db - memory will not persist")
    return None


async def get_memory_manager(config: Optional[Dict[str, Any]] = None):
    """
    Get or create the global memory manager (singleton pattern).
    
    FIXED: Now uses LangGraph thread_id for consistent session tracking
    """
    from react_agent.memory.manager import MemoryManager

    global _global_memory
    
    # ✅ CRITICAL FIX: Extract thread_id from config (consistent across runs)
    thread_id = _extract_thread_id_from_config(config)
    
    # If still no thread_id, generate one (fallback)
    if not thread_id:
        thread_id = f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        print(f"⚠️  [Memory] No thread_id found in config, generated: {thread_id}")
    
    # Debug logging
    print(f"\n{'='*60}")
    print(f"🧠 [MemoryManager] get_memory_manager called")
    print(f"   Thread ID: {thread_id}")
    print(f"   Global memory exists: {_global_memory is not None}")
    
    if _global_memory:
        current_thread = getattr(_global_memory, 'session_id', 'unknown')
        print(f"   Current thread: {current_thread}")
        if hasattr(_global_memory, 'working_memory'):
            tool_count = len(_global_memory.working_memory.recent_tool_results)
            print(f"   Stored tool results: {tool_count}")
            for i, tool in enumerate(_global_memory.working_memory.recent_tool_results[-3:], 1):
                print(f"      {i}. {tool['summary'][:60]}")
    print(f"{'='*60}\n")
    
    # ✅ FIXED: Session tracking with thread_id
    if _global_memory is not None and thread_id:
        current_thread = getattr(_global_memory, 'session_id', None)
        if current_thread and current_thread != thread_id:
            print(f"🧠 [Memory] New thread detected ({thread_id}) - loading new thread memory")
            _global_memory = None
    
    # Clean up old tool results based on age (if memory exists)
    if _global_memory and hasattr(_global_memory, 'working_memory'):
        cutoff_time = datetime.now(UTC) - timedelta(minutes=30)
        before_count = len(_global_memory.working_memory.recent_tool_results)
        
        # ✅ FIX: Use safe timestamp parsing to handle timezone inconsistencies
        _global_memory.working_memory.recent_tool_results = [
            entry for entry in _global_memory.working_memory.recent_tool_results
            if _parse_timestamp_safe(entry["timestamp"]) > cutoff_time
        ]
        
        after_count = len(_global_memory.working_memory.recent_tool_results)
        if after_count < before_count:
            print(f" [Memory] Cleaned up {before_count - after_count} old tool results (older than 30 minutes)")
    
    if _global_memory is None:
        # CRITICAL: Get movesia.db path
        db_path = _get_movesia_db_path()
        
        # Check config override
        if config:
            configurable = config.get("configurable", {})
            
            # Check for sqlite_path (from Electron)
            if "sqlite_path" in configurable:
                electron_db_path = Path(configurable["sqlite_path"])
                if electron_db_path.exists():
                    db_path = electron_db_path
                    print(f" [Memory] Using database path from Electron: {db_path}")
            
            # Check for explicit memory_db_path
            elif "memory_db_path" in configurable:
                config_db_path = Path(configurable["memory_db_path"])
                if config_db_path.exists():
                    db_path = config_db_path
        
        if db_path:
            print(f" [Memory] Using database: {db_path}")
        else:
            print(f" [Memory] No database found - memory will not persist across sessions")
        
        # ✅ ADD: Verify session data in database (async-safe)
        if db_path and db_path.exists():
            try:
                # ✅ FIX: Use asyncio.to_thread for blocking SQLite operations
                def _verify_database():
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    # Check what sessions exist
                    cursor.execute("SELECT DISTINCT session_id FROM memory_working LIMIT 10")
                    existing_sessions = [row[0] for row in cursor.fetchall()]
                    print(f"📊 [Memory] Database has {len(existing_sessions)} sessions: {existing_sessions[:3]}")
                    
                    # Check if our session has data
                    cursor.execute("SELECT COUNT(*) FROM memory_working WHERE session_id = ?", (thread_id,))
                    count = cursor.fetchone()[0]
                    print(f"📊 [Memory] Session '{thread_id}' has {count} tool results in database")
                    
                    conn.close()
                
                # Execute in separate thread to avoid blocking
                await asyncio.to_thread(_verify_database)
                
            except Exception as e:
                print(f"⚠️  [Memory] Could not verify database: {e}")
        
        # FIX: Use asyncio.to_thread() to avoid blocking
        _global_memory = await asyncio.to_thread(
            MemoryManager,
            db_path=db_path,
            auto_persist=bool(db_path),
            session_id=thread_id
        )
        
        print(f"🧠 [Memory] Initialized global memory manager")
        print(f"   Thread ID: {_global_memory.session_id}")
        print(f"   Database: {db_path}")
        print(f"   Auto-persist: {bool(db_path)}")
        
        # Log loaded state
        if hasattr(_global_memory, 'working_memory'):
            tool_count = len(_global_memory.working_memory.recent_tool_results)
            print(f"   Loaded tool results: {tool_count}")
            if tool_count > 0:
                for i, tool in enumerate(_global_memory.working_memory.recent_tool_results[:3], 1):
                    print(f"      {i}. {tool['tool_name']}: {tool['summary'][:50]}")
        else:
            _global_memory._session_id = None
            print(f" [Memory] Initialized global memory manager (no session ID)")
        
        if db_path:
            print(f" [Memory] Loaded memory from database: {db_path}")
            print(f"   Persistent storage: {db_path}")
    
    return _global_memory


def reset_memory_manager():
    """
    Reset global memory manager (testing only).
    
    ⚠️ WARNING: This clears the global singleton.
    - Use only in tests or when explicitly restarting
    - Not thread-safe
    - Loses all in-memory data
    """
    global _global_memory
    _global_memory = None


# ============================================================================
# CONTEXT EXTRACTION UTILITIES
# ============================================================================

def extract_entities_from_state(state: State) -> List[str]:
    """
    Extract entities (files, classes) from current state.
    Works with both plan and messages.
    
    Args:
        state: Current agent state
        
    Returns:
        List of entity names
    """
    entities = set()
    
    # Extract from plan
    if state.plan:
        for step in state.plan.steps:
            desc = step.description.lower()
            # Look for file extensions
            words = desc.split()
            for word in words:
                if any(ext in word for ext in [".cs", ".unity", ".prefab", ".asset", ".mat", ".shader"]):
                    entities.add(word.strip(".,;:\"'()[]{}"))
                # Look for Unity class names
                elif any(cls in word for cls in ["Controller", "Manager", "System", "Handler", "Component"]):
                    entities.add(word.strip(".,;:\"'()[]{}"))
    
    # Extract from recent messages
    for msg in state.messages[-3:]:
        content = get_message_text(msg).lower()
        words = content.split()
        for word in words:
            if any(ext in word for ext in [".cs", ".unity", ".prefab", ".asset"]):
                entities.add(word.strip(".,;:\"'()[]{}"))
    
    return list(entities)


def extract_topics_from_state(state: State) -> List[str]:
    """
    Extract topics (Unity concepts) from current state.
    
    Args:
        state: Current agent state
        
    Returns:
        List of topic keywords
    """
    topics = set()
    
    # Unity/game dev topic keywords
    topic_keywords = {
        "movement": ["move", "movement", "walk", "run", "jump", "player"],
        "input": ["input", "keyboard", "mouse", "controller", "button"],
        "physics": ["physics", "rigidbody", "collider", "collision", "force"],
        "animation": ["animation", "animator", "anim", "animate"],
        "ui": ["ui", "menu", "button", "canvas", "panel", "text"],
        "audio": ["audio", "sound", "music", "sfx"],
        "camera": ["camera", "view", "perspective"],
        "ai": ["ai", "enemy", "npc", "pathfinding", "navmesh"],
        "scene": ["scene", "level", "load", "transition"],
        "prefab": ["prefab", "instantiate", "spawn"],
        "shader": ["shader", "material", "graphics", "rendering"],
        "error": ["error", "fix", "debug", "problem", "compilation"],
    }
    
    # Extract from plan
    if state.plan:
        goal_lower = state.plan.goal.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in goal_lower for kw in keywords):
                topics.add(topic)
        
        for step in state.plan.steps:
            desc_lower = step.description.lower()
            for topic, keywords in topic_keywords.items():
                if any(kw in desc_lower for kw in keywords):
                    topics.add(topic)
    
    # Extract from recent messages
    for msg in state.messages[-3:]:
        content_lower = get_message_text(msg).lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                topics.add(topic)
    
    return list(topics)


def extract_entities_from_request(request: str) -> List[str]:
    """Extract entities from a user request string."""
    entities = []
    
    # Common Unity file extensions
    extensions = [".cs", ".unity", ".prefab", ".asset", ".mat", ".shader"]
    
    words = request.split()
    for word in words:
        # Check if word contains a file extension
        if any(ext in word.lower() for ext in extensions):
            entities.append(word.strip(".,;:\"'()[]{}"))
        # Check for common Unity class names
        elif any(cls in word for cls in ["Controller", "Manager", "System", "Handler"]):
            entities.append(word)
    
    return entities


def extract_topics_from_request(request: str) -> List[str]:
    """Extract topics from a user request string."""
    topics = []
    
    topic_keywords = {
        "movement": ["move", "movement", "walk", "run", "jump", "player"],
        "input": ["input", "keyboard", "mouse", "controller", "button"],
        "physics": ["physics", "rigidbody", "collider", "collision"],
        "animation": ["animation", "animator", "anim"],
        "ui": ["ui", "menu", "button", "canvas"],
        "audio": ["audio", "sound", "music"],
        "camera": ["camera", "view"],
        "ai": ["ai", "enemy", "npc", "pathfinding"],
        "scene": ["scene", "level"],
        "prefab": ["prefab", "instantiate"],
    }
    
    request_lower = request.lower()
    for topic, keywords in topic_keywords.items():
        if any(kw in request_lower for kw in keywords):
            topics.append(topic)
    
    return topics


# ============================================================================
# MEMORY CONTEXT FORMATTING
# ============================================================================

def format_memory_context(memory_context: Dict[str, Any]) -> str:
    """Format memory context for LLM consumption."""
    parts = []
    
    # Working memory
    wm = memory_context.get("working_memory", {})
    if wm.get("summary") and wm["summary"] != "No active context":
        parts.append(f"**Current Focus**: {wm['summary']}")
    
    # Episodic memory - similar past tasks
    em = memory_context.get("episodic_memory", {})
    similar = em.get("similar_past_interactions", [])
    if similar:
        parts.append("\n**Similar Past Tasks**:")
        for i, interaction in enumerate(similar[:2], 1):
            outcome = "✓" if interaction.get("outcome") == "success" else "✗"
            goal = interaction.get("goal", 'Unknown')[:60]
            parts.append(f"{i}. {outcome} {goal}")
            if interaction.get("what_worked"):
                parts.append(f"   → {interaction['what_worked'][:80]}")
    
    # Success patterns
    patterns = em.get("success_patterns", {})
    if patterns.get("pattern") == "success" and patterns.get("sample_size", 0) > 0:
        parts.append(f"\n**Success Pattern**: Typically uses {patterns.get('typical_step_count', 'N/A')} steps")
        tools = patterns.get("preferred_tools", [])
        if tools:
            tool_names = [t[0] for t in tools[:3]]
            parts.append(f"   Preferred tools: {', '.join(tool_names)}")
    
    # Semantic memory - project knowledge
    sm = memory_context.get("semantic_memory", {})
    profile = sm.get("project_profile", {})
    if profile.get("common_scripts"):
        scripts = [s["name"] for s in profile["common_scripts"][:3]]
        parts.append(f"\n**Common Scripts**: {', '.join(scripts)}")
    
    # User preferences
    prefs = sm.get("user_preferences", {})
    if prefs.get("planning_style"):
        parts.append(f"\n**User Preference**: {prefs['planning_style']} planning")
    
    return "\n".join(parts) if parts else ""


async def inject_memory_context(
    state: State,
    base_prompt: str,
    config: Dict[str, Any]
) -> str:
    """
    Inject memory context into LLM prompt.
    
    ✅ FIXED: Now async to support async get_memory_context()
    
    Args:
        state: Current agent state
        base_prompt: Base system prompt
        config: RunnableConfig containing context
        
    Returns:
        Enhanced prompt with memory context
    """
    if not state.memory:
        return base_prompt
    
    # Check if memory context injection is enabled
    context = config.get("configurable", {})
    if not context.get("memory_inject_context", True):
        return base_prompt
    
    # Get memory context
    memory_context = await state.memory.get_memory_context(
        include_patterns=True,
        include_episodes=True
    )
    
    if not memory_context:
        return base_prompt
    
    # Inject memory context
    enhanced_prompt = f"""{base_prompt}

# Memory Context

{memory_context}

Use this memory context to inform your decisions and avoid repeating past mistakes.
"""
    
    return enhanced_prompt


async def inject_memory_into_prompt(
    base_prompt: str,
    state: 'State',
    include_patterns: bool = True,
    include_episodes: bool = True
) -> str:
    """
    Inject memory context into any LLM prompt.
    
    ✅ FIXED: Now async to support async get_memory_context()
    """
    logger.info(f"🧠 [MEMORY] inject_memory_into_prompt called")
    logger.info(f"🧠 [MEMORY]   State has memory: {state.memory is not None}")
    
    if not state.memory:
        logger.debug("🧠 [MEMORY] No memory available for injection")
        return base_prompt
    
    # Log memory state
    if hasattr(state.memory, 'working_memory'):
        tool_count = len(state.memory.working_memory.recent_tool_results)
        logger.info(f"🧠 [MEMORY]   Working memory tool results: {tool_count}")
        if tool_count > 0:
            for i, tool in enumerate(state.memory.working_memory.recent_tool_results, 1):
                logger.info(f"🧠 [MEMORY]     {i}. {tool['summary']}")
    
    # Get memory context string
    memory_context_str = await state.memory.get_memory_context(
        include_patterns=include_patterns,
        include_episodes=include_episodes
    )
    
    if not memory_context_str:
        logger.debug("🧠 [MEMORY] No memory context to inject")
        return base_prompt
    
    logger.info(f"🧠 [MEMORY] Injecting memory context:")
    logger.info(f"🧠 [MEMORY]   Patterns: {include_patterns}")
    logger.info(f"🧠 [MEMORY]   Episodes: {include_episodes}")
    logger.info(f"🧠 [MEMORY]   Context length: {len(memory_context_str)} chars")
    logger.info(f"🧠 [MEMORY]   Context preview: {memory_context_str[:200]}")
    
    # Inject memory context
    return f"""{base_prompt}

# Memory Context

{memory_context_str}

Use this memory context to inform your decisions and avoid repeating past mistakes.
"""


# ============================================================================
# MEMORY INSIGHTS
# ============================================================================

def get_memory_insights(state: State) -> Dict[str, Any]:
    """
    Get insights from memory for any node that needs it.
    
    Usage in any node:
        insights = get_memory_insights(state)
        if insights["relevant_patterns"]:
            # Use patterns in decision making
    """
    if not state.memory:
        return {
            "working_context": "No memory available",
            "relevant_patterns": [],
            "similar_tasks": []
        }
    
    insights = {
        "working_context": state.memory.get_working_context(),
        "relevant_patterns": [],
        "similar_tasks": [],
    }
    
    # Get relevant patterns
    if state.plan:
        patterns = state.memory.find_relevant_patterns(
            context=state.plan.goal,
            pattern_type="success",
            min_confidence=0.6,
            limit=3
        )
        
        insights["relevant_patterns"] = [
            {
                "context": p.context,
                "action": p.action,
                "confidence": p.confidence,
                "success_rate": p.success_rate,
            }
            for p in patterns
        ]
    
    # Get similar tasks
    entities = extract_entities_from_state(state)
    topics = extract_topics_from_state(state)
    
    if entities or topics:
        similar = state.memory.find_similar_tasks(entities, topics, limit=2)
        
        insights["similar_tasks"] = [
            {
                "task": ep.task_description,
                "success": ep.success,
                "outcome": ep.outcome_summary,
            }
            for ep in similar
        ]
    
    return insights


# ============================================================================
# MEMORY INITIALIZATION
# ============================================================================

async def initialize_memory_system(state: State, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize memory system at agent startup.
    
    CANONICAL ENTRY POINT - Called by classify.py node only.
    Other nodes should just check if state.memory exists.
    
    Args:
        state: Current agent state
        config: RunnableConfig containing context
    
    Returns:
        Dict with "memory" key containing MemoryManager instance
    """
    # Check if memory already initialized
    if state.memory is not None:
        return {}
    
    # Get context from config
    context = config.get("configurable", {})
    
    # Check if memory is enabled
    if not context.get("enable_memory", True):
        return {"memory_enabled": False}
    
    # Get or create global memory manager
    memory = await get_memory_manager(config)
    
    # Extract task from first user message
    task_description = "Unknown task"
    if state.messages:
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                task_description = get_message_text(msg)[:200]
                break
    
    # Start episode
    episode_id = memory.start_task(task_description)
    
    # Add initial messages to memory
    for msg in state.messages:
        memory.add_message(
            role=getattr(msg, 'type', 'unknown'),
            content=get_message_text(msg),
            metadata={"message_id": getattr(msg, 'id', None)}
        )
    
    # Extract and set initial focus
    entities = extract_entities_from_state(state)
    topics = extract_topics_from_state(state)
    if entities or topics:
        memory.update_focus(entities, topics)
    
    print(f"🧠 [Memory] Episode started: {episode_id}")
    print(f"   Task: {task_description[:60]}...")
    if entities:
        print(f"   Entities: {', '.join(entities[:3])}")
    if topics:
        print(f"   Topics: {', '.join(topics[:3])}")
    
    return {
        "memory": memory,
        "memory_enabled": True,
    }


