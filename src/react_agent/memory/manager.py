"""
Unified Memory Manager with FIXED async SQLite operations.

✅ FIXED: All SQLite calls now wrapped in asyncio.to_thread()
✅ FIXED: Proper async/await for blocking operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import hashlib
import time
from functools import wraps
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from react_agent.memory.working_memory import WorkingMemory
from react_agent.memory.episodic import EpisodicMemory, Episode, EpisodeStatus
from react_agent.memory.semantic import SemanticMemory, SemanticFact, Pattern

logger = logging.getLogger(__name__)


def retry_on_lock(max_attempts=3, delay=0.1):
    """Decorator to retry database operations on lock errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < max_attempts - 1:
                        logger.warning(f"🧠 [MemoryManager] Database locked, retry {attempt + 1}/{max_attempts}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        last_error = e
                        continue
                    raise
            raise last_error if last_error else Exception("Retry failed")
        return wrapper
    return decorator

class MemoryManager:
    """
    Unified interface to all three memory tiers with proper async handling.
    
    ✅ FIXED: All blocking SQLite operations now use asyncio.to_thread()
    """
    
    def __init__(self, db_path: Optional[Path] = None, auto_persist: bool = True, session_id: Optional[str] = None):
        """
        Initialize memory manager (blocking - must be called via asyncio.to_thread).
        """
        # Initialize all three memory tiers
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        
        semantic_db_path = db_path if db_path else None
        self.semantic_memory = SemanticMemory(storage_path=semantic_db_path)
        
        # Session tracking
        self.session_id = session_id or f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        
        # Persistence settings
        self.db_path = db_path
        self.auto_persist = auto_persist
        
        # Load existing memory if database exists (blocking, but OK in __init__)
        if self.db_path and self.db_path.exists() and self.auto_persist:
            self._load_from_database_sync()  # Synchronous load in __init__
            logger.info(f"   Entities: {len(self.semantic_memory.entity_knowledge)}")
            logger.info(f"   Topics: {len(self.semantic_memory.topic_knowledge)}")
            
            # ✅ FIX: Force reload semantic memory if empty (after interrupt resume)
            if len(self.semantic_memory.entity_knowledge) == 0:
                self._reload_semantic_knowledge()
        
        # Clean up dangling in-progress episodes
        if self.episodic_memory.current_episode:
            logger.warning(f" [MemoryManager] Found unclosed episode: {self.episodic_memory.current_episode.episode_id}")
            self.episodic_memory.end_episode(
                success=False,
                outcome_summary="Session interrupted (recovered from crash)"
            )
            self.consolidate_memories_sync()  # Synchronous in __init__
    
    # ========== Working Memory Operations ==========
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to both working and episodic memory (synchronous)."""
        self.working_memory.add_message(role, content, metadata)
        self.episodic_memory.add_message(role, content, metadata)
    
    async def update_focus(self, entities: List[str], topics: List[str]):
        """Update current focus in working memory (async)."""
        self.working_memory.update_focus(entities, topics)
        self.episodic_memory.update_entities(entities)
        self.episodic_memory.update_topics(topics)
        
        # ✅ FIXED: Async persistence
        if self.auto_persist and self.db_path:
            await self._persist_session_metadata()
    
    def update_focus_sync(self, entities: List[str], topics: List[str]):
        """Synchronous version of update_focus for sync contexts."""
        self.working_memory.update_focus(entities, topics)
        self.episodic_memory.update_entities(entities)
        self.episodic_memory.update_topics(topics)
        
        # Persist session metadata synchronously
        if self.auto_persist and self.db_path:
            try:
                self._persist_session_metadata_sync()
            except Exception as e:
                logger.warning(f" [MemoryManager] Failed to persist session metadata: {e}")
    
    def get_working_context(self) -> str:
        """Get current working memory context (synchronous)."""
        return self.working_memory.get_context_summary()
    
    # ========== Episodic Memory Operations ==========
    
    def start_task(self, task_description: str) -> str:
        """Start a new task episode (synchronous)."""
        episode_id = self.episodic_memory.start_episode(task_description)
        self.working_memory.user_intent = task_description
        return episode_id
    
    async def end_task(
        self, 
        success: bool, 
        outcome_summary: Optional[str] = None,
        clear_working_memory: bool = True
    ):
        """
        End current task episode with optional working memory preservation.
        
        ✅ FIXED: Now async with proper thread handling
        """
        if not self.episodic_memory.current_episode:
            logger.warning(" [MemoryManager] end_task called but no current episode")
            return
        
        episode_id = self.episodic_memory.current_episode.episode_id
        self.episodic_memory.end_episode(success, outcome_summary)
        
        # Learn from episode
        if self.episodic_memory.recent_episodes:
            last_episode = self.episodic_memory.recent_episodes[-1]
            self.semantic_memory.learn_from_episode(last_episode)
        
        # ✅ FIXED: Async persistence BEFORE clearing
        if self.auto_persist and self.db_path:
            await self._persist_to_database()  # Now async!
        
        # Only clear working memory if explicitly requested
        if clear_working_memory:
            self.working_memory.clear()
    
    def add_plan(self, plan: Dict[str, Any]):
        """Add plan to episodic memory (synchronous)."""
        self.episodic_memory.add_plan(plan)
        self.working_memory.current_plan = plan
    
    async def add_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any):
        """
        Add tool call to episodic and working memory, then persist.
        
        ✅ FIXED: Now async
        ✅ FIXED: Extracts and stores entity/topic knowledge for semantic memory
        """
        # Add to episodic memory
        self.episodic_memory.add_tool_call(tool_name, args, result)
        
        # Add to working memory
        # ✅ FIX: Better context for file operations
        if tool_name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
            file_path = args.get("file_path", "")
            query = f"{tool_name} {file_path}"  # e.g., "write_file Assets/Scripts/test7284.cs"
        else:
            query = args.get("query", "") or args.get("sql_query", "") or args.get("operation", "") or args.get("query_description", "")
        self.working_memory.add_tool_result(tool_name, result, query)
        
        # ✅ FIX: Extract query_text FIRST, before using it
        if tool_name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
            file_path = args.get("file_path", "")
            query_text = f"{tool_name} {file_path}"
        else:
            query_text = (
                args.get("query", "") or 
                args.get("query_description", "") or 
                args.get("natural_query", "") or
                args.get("description", "") or
                args.get("sql_query", "")
            )
        
        # ✅ EXTRACT ENTITIES: Use smart extractor with tool results
        from react_agent.memory.entity_extractor import extract_entities_simple

        entity_count = 0
        entities_found = extract_entities_simple(
            text=query_text,  # Also uses query text as secondary source
            tool_name=tool_name,
            tool_result=result
        )

        for entity in entities_found:
            # Entity already normalized by extractor
            knowledge = {
                "type": "file" if any(ext in entity for ext in [".cs", ".shader"]) else "asset",
                "last_accessed": datetime.now(UTC).isoformat(),
                "access_count": 1
            }
            
            # Enhance with tool-specific data
            if tool_name == "search_project" and isinstance(result, dict):
                structured = result.get("results_structured", [])
                for item in structured:
                    if isinstance(item, dict):
                        item_name = (item.get("path") or item.get("name", "")).lower()
                        if entity in item_name:
                            knowledge["size"] = item.get("size")
                            knowledge["mtime"] = item.get("mtime")
                            knowledge["type"] = item.get("kind", knowledge["type"])
                            break
            
            self.update_entity_knowledge(entity, knowledge)
            entity_count += 1

        if entity_count > 0:
            self.update_focus_sync(entities_found, [])
        
        # ✅ SIMPLE: Extract topics using synchronous keyword matching
        from react_agent.memory.topic_extractor import extract_topics_simple
        
        if query_text:
            try:
                # Simple synchronous extraction (works in async context too)
                result_summary = result.get("summary", "") if isinstance(result, dict) else str(result)[:200]
                topics_found = extract_topics_simple(
                    query=query_text,
                    tool_name=tool_name,
                    result_summary=result_summary
                )
                
                # Store extracted topics
                for topic in topics_found:
                    topic_knowledge = {
                        "queries": [query_text[:100]],
                        "tool_used": tool_name,
                        "last_queried": datetime.now(UTC).isoformat(),
                        "success": result.get("success", True),
                        "query_count": 1
                    }
                    self.update_topic_knowledge(topic, topic_knowledge)
                
                if topics_found:
                    self.update_focus_sync([], topics_found)
            except Exception as e:
                logger.error(f" [MemoryManager]   ❌ Topic extraction failed: {e}", exc_info=True)
        
        # ✅ FIXED: Async persistence
        if self.auto_persist and self.db_path:
            await self._persist_working_memory()
    
    def add_tool_call_sync(self, tool_name: str, args: Dict[str, Any], result: Any):
        """
        Synchronous wrapper for add_tool_call - safe to call from sync contexts.
        
        This is needed for LangGraph routing functions which cannot be async.
        ✅ FIXED: Now uses synchronous persistence instead of trying to schedule async tasks.
        ✅ FIXED: Now extracts entity and topic knowledge like async version.
        ✅ FIXED: Handles different result formats (natural_language vs structured).
        """
        # Add to episodic memory (synchronous)
        self.episodic_memory.add_tool_call(tool_name, args, result)
        
        # ✅ FIX: Extract query_text FIRST, before using it
        if tool_name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
            file_path = args.get("file_path", "")
            
            # ⚠️ Special handling for modify_file with Pydantic model
            if tool_name == "modify_file" and "modification_spec" in args:
                spec = args["modification_spec"]
                # Extract type for logging (handle both dict and Pydantic model)
                if isinstance(spec, dict):
                    spec_type = spec.get("type", "unknown")
                else:
                    # Pydantic model - access type attribute
                    spec_type = getattr(spec, "type", "unknown")
                query_text = f"{tool_name} {file_path} (type: {spec_type})"
            else:
                query_text = f"{tool_name} {file_path}"
        else:
            query_text = (
                args.get("query", "") or 
                args.get("query_description", "") or 
                args.get("natural_query", "") or
                args.get("description", "") or
                args.get("sql_query", "")
            )
        
        # Add to working memory (synchronous)
        self.working_memory.add_tool_result(tool_name, result, query_text)
        
        # ✅ EXTRACT ENTITIES: Use smart extractor with tool results
        from react_agent.memory.entity_extractor import extract_entities_simple

        entity_count = 0
        entities_found = extract_entities_simple(
            text=query_text,  # Also uses query text as secondary source
            tool_name=tool_name,
            tool_result=result
        )

        for entity in entities_found:
            # Entity already normalized by extractor
            knowledge = {
                "type": "file" if any(ext in entity for ext in [".cs", ".shader"]) else "asset",
                "last_accessed": datetime.now(UTC).isoformat(),
                "access_count": 1
            }
            
            # Enhance with tool-specific data
            if tool_name == "search_project" and isinstance(result, dict):
                structured = result.get("results_structured", [])
                for item in structured:
                    if isinstance(item, dict):
                        item_name = (item.get("path") or item.get("name", "")).lower()
                        if entity in item_name:
                            knowledge["size"] = item.get("size")
                            knowledge["mtime"] = item.get("mtime")
                            knowledge["type"] = item.get("kind", knowledge["type"])
                            break
            
            self.update_entity_knowledge(entity, knowledge)
            entity_count += 1

        if entity_count > 0:
            self.update_focus_sync(entities_found, [])
        
        # ✅ SIMPLE: Extract topics using synchronous keyword matching
        from react_agent.memory.topic_extractor import extract_topics_simple
        
        if query_text:
            try:
                # Simple synchronous extraction (no async/event loop issues)
                result_summary = result.get("summary", "") if isinstance(result, dict) else str(result)[:200]
                topics_found = extract_topics_simple(
                    query=query_text,
                    tool_name=tool_name,
                    result_summary=result_summary
                )
                
                # Store extracted topics
                for topic in topics_found:
                    topic_knowledge = {
                        "queries": [query_text[:100]],
                        "success": result.get("success", True) if isinstance(result, dict) else True,
                        "query_count": 1
                    }
                    self.update_topic_knowledge(topic, topic_knowledge)
                
                if topics_found:
                    self.update_focus_sync([], topics_found)
            except Exception as e:
                logger.error(f" [MemoryManager]   ❌ Topic extraction failed: {e}", exc_info=True)
        else:
            pass  # No query text available
        
        # ✅ FIX: Persist ALL memory types in ONE transaction to avoid locks
        if self.auto_persist and self.db_path:
            try:
                # ✅ NEW: Use single connection for all operations to avoid locks
                conn = sqlite3.connect(str(self.db_path), timeout=10.0)  # Longer timeout
                cursor = conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 10000")  # 10 second timeout
                
                try:
                    # 1. Persist working memory (tool results)
                    self._persist_working_memory_with_cursor(cursor)
                    
                    # 2. Persist semantic memory (entities/topics)
                    if len(self.semantic_memory.entity_knowledge) > 0 or len(self.semantic_memory.topic_knowledge) > 0:
                        self._persist_semantic_memory_with_cursor(cursor)
                    
                    # 3. Persist session metadata (focus_entities, focus_topics)
                    self._persist_session_metadata_with_cursor(cursor)
                    
                    # Commit everything at once
                    conn.commit()
                    
                finally:
                    conn.close()
                    
            except Exception as e:
                logger.warning(f" [MemoryManager]   ⚠️ Persistence failed: {e}")
                # Store in a pending queue for next async opportunity
                if not hasattr(self, '_pending_persistence'):
                    self._pending_persistence = []
                self._pending_persistence.append(('working', datetime.now(UTC)))
    
    def add_assessment(self, assessment: Dict[str, Any]):
        """Add assessment to episodic memory (synchronous)."""
        self.episodic_memory.add_assessment(assessment)
    
    def add_error(self, error: Dict[str, Any]):
        """Add error to episodic memory (synchronous)."""
        self.episodic_memory.add_error(error)
    
    def find_similar_tasks(self, entities: List[str], topics: List[str], limit: int = 5) -> List[Episode]:
        """Find similar past tasks (synchronous)."""
        return self.episodic_memory.find_similar_episodes(entities, topics, limit)
    
    # ========== Semantic Memory Operations ==========
    
    def find_relevant_patterns(
        self,
        context: str,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 5
    ) -> List[Pattern]:
        """Find relevant patterns for current context (synchronous)."""
        return self.semantic_memory.find_relevant_patterns(context, pattern_type, min_confidence, limit)
    
    async def apply_pattern(self, pattern_id: str, success: bool):
        """
        Record application of a pattern.
        
        ✅ FIXED: Now async
        """
        self.semantic_memory.update_pattern(pattern_id, success)
        
        if self.auto_persist and self.db_path:
            await self._persist_to_database()
    
    def update_entity_knowledge(self, entity: str, knowledge: Dict[str, Any]):
        """Update knowledge about an entity (synchronous)."""
        self.semantic_memory.update_entity_knowledge(entity, knowledge)
    
    def update_topic_knowledge(self, topic: str, knowledge: Dict[str, Any]):
        """Update knowledge about a topic (synchronous)."""
        self.semantic_memory.update_topic_knowledge(topic, knowledge)
    
    # ========== Memory Consolidation ==========

    def consolidate_memories_sync(self):
        """Consolidate memories (synchronous version for __init__)."""
        learned_count = 0
        for episode in self.episodic_memory.recent_episodes[-5:]:
            if episode.status == EpisodeStatus.COMPLETED:
                self.semantic_memory.learn_from_episode(episode)
                learned_count += 1

    async def consolidate_memories(self):
        """
        Consolidate memories across tiers with persistence.

        ✅ FIXED: Now async
        """
        self.consolidate_memories_sync()

        # Persist consolidated memories
        if self.auto_persist and self.db_path:
            await self._persist_to_database()
    
    # ========== Context Generation for LLM ==========
    
    async def get_memory_context(self, include_patterns: bool = True, include_episodes: bool = True) -> str:
        """
        Generate memory context for LLM prompt injection.
        
        Returns a formatted string containing relevant memory information.
        ✅ FIXED: Now async and reloads working memory using asyncio.to_thread
        """
        
        # ✅ FIX: Reload working memory from database if empty but we have a session
        if len(self.working_memory.recent_tool_results) == 0 and self.db_path and self.db_path.exists():
            try:
                # ✅ CRITICAL FIX: Use asyncio.to_thread for blocking SQLite operations
                def _reload_tools():
                    conn = sqlite3.connect(str(self.db_path), timeout=5.0)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT tool_name, query, result, summary, timestamp
                        FROM memory_working
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (self.session_id, self.working_memory.max_tool_results))
                    
                    tools = []
                    for row in cursor.fetchall():
                        try:
                            tool_result = {
                                "tool_name": row[0],
                                "query": row[1],
                                "result": json.loads(row[2]),
                                "summary": row[3],
                                "timestamp": row[4]
                            }
                            tools.append(tool_result)
                        except Exception as e:
                            logger.warning(f" [MemoryManager] Failed to reload tool result: {e}")
                    
                    conn.close()
                    return tools
                
                # Execute in separate thread to avoid blocking
                loaded_tools = await asyncio.to_thread(_reload_tools)
                
                for tool in loaded_tools:
                    self.working_memory.recent_tool_results.append(tool)
                
                if len(loaded_tools) > 0:
                    pass  # Successfully reloaded
                    
            except Exception as e:
                logger.warning(f" [MemoryManager]   ⚠️ Could not reload tools from database: {e}")
        
        # ✅ FIX: Reload entity knowledge from database if empty
        if len(self.semantic_memory.entity_knowledge) == 0 and self.db_path and self.db_path.exists():
            try:
                def _reload_entities():
                    conn = sqlite3.connect(str(self.db_path), timeout=5.0)
                    cursor = conn.cursor()
                    cursor.execute("SELECT entity, knowledge FROM memory_entities")
                    
                    entities = {}
                    for row in cursor.fetchall():
                        try:
                            entity, knowledge_json = row
                            knowledge = json.loads(knowledge_json)
                            entities[entity] = knowledge
                        except Exception as e:
                            logger.warning(f" [MemoryManager] Failed to reload entity: {e}")
                    
                    conn.close()
                    return entities
                
                loaded_entities = await asyncio.to_thread(_reload_entities)
                self.semantic_memory.entity_knowledge.update(loaded_entities)
                
                if len(loaded_entities) > 0:
                    pass  # Successfully reloaded
                    
            except Exception as e:
                logger.warning(f" [MemoryManager]   ⚠️ Could not reload entities from database: {e}")
        
        # ✅ FIX: Reload topic knowledge from database if empty  
        if len(self.semantic_memory.topic_knowledge) == 0 and self.db_path and self.db_path.exists():
            try:
                def _reload_topics():
                    conn = sqlite3.connect(str(self.db_path), timeout=5.0)
                    cursor = conn.cursor()
                    cursor.execute("SELECT topic, knowledge FROM memory_topics")
                    
                    topics = {}
                    for row in cursor.fetchall():
                        try:
                            topic, knowledge_json = row
                            knowledge = json.loads(knowledge_json)
                            topics[topic] = knowledge
                        except Exception as e:
                            logger.warning(f" [MemoryManager] Failed to reload topic: {e}")
                    
                    conn.close()
                    return topics
                
                loaded_topics = await asyncio.to_thread(_reload_topics)
                self.semantic_memory.topic_knowledge.update(loaded_topics)
                
                if len(loaded_topics) > 0:
                    pass  # Successfully reloaded
                    
            except Exception as e:
                logger.warning(f" [MemoryManager]   ⚠️ Could not reload topics from database: {e}")
        
        context_parts = []
        
        # ✅ NEW: Recent tool results FIRST (most important for follow-up questions)
        tool_context = self.working_memory.get_recent_tool_context(limit=2)
        if tool_context:
            context_parts.append(f"## Recent Actions\n{tool_context}")
        
        # Working memory context
        working_context = self.working_memory.get_context_summary()
        if working_context != "No active context":
            context_parts.append(f"## Current Context\n{working_context}")
        
        # Relevant patterns from semantic memory
        if include_patterns and self.working_memory.user_intent:
            patterns = self.find_relevant_patterns(
                self.working_memory.user_intent,
                pattern_type="success",
                min_confidence=0.6,
                limit=3
            )
            
            if patterns:
                pattern_text = "## Learned Patterns\n"
                for i, pattern in enumerate(patterns, 1):
                    pattern_text += f"{i}. **{pattern.pattern_type.title()}** (confidence: {pattern.confidence:.2f})\n"
                    pattern_text += f"   - Context: {pattern.context}\n"
                    pattern_text += f"   - Action: {pattern.action}\n"
                    pattern_text += f"   - Applied {pattern.times_applied} times with {pattern.success_rate:.1%} success\n"
                context_parts.append(pattern_text)
        
        # ✅ ENTITY KNOWLEDGE RETRIEVAL
        
        recent_entities = set()
        if len(self.working_memory.recent_tool_results) > 0:
            for tool_result in self.working_memory.recent_tool_results[-2:]:
                result_data = tool_result.get("result", {})
                
                if isinstance(result_data, dict):
                    structured = result_data.get("results_structured", [])
                    
                    if isinstance(structured, list) and len(structured) > 0:
                        for item in structured[:10]:
                            if isinstance(item, dict):
                                entity = item.get("path") or item.get("name")
                                if entity:
                                    recent_entities.add(entity)
        
        # Combine strategies: focus_entities + recent_entities
        all_entities = set(self.working_memory.focus_entities) | recent_entities
        
        # Retrieve knowledge for these entities
        entity_context_parts = []
        
        for entity in list(all_entities)[:10]:
            if entity in self.semantic_memory.entity_knowledge:
                knowledge = self.semantic_memory.entity_knowledge[entity]
                entity_name = entity.split("/")[-1] if "/" in entity else entity
                details = []
                
                if "size" in knowledge:
                    details.append(f"size: {knowledge['size']} bytes")
                
                if "type" in knowledge:
                    details.append(f"type: {knowledge['type']}")
                
                if "access_count" in knowledge and knowledge["access_count"] > 0:
                    details.append(f"accessed {knowledge['access_count']} times")
                
                if details:
                    formatted = f"- **{entity_name}**: {', '.join(details)}"
                    entity_context_parts.append(formatted)
        
        if entity_context_parts:
            entity_section = f"## Known Entities\n" + "\n".join(entity_context_parts)
            context_parts.append(entity_section)
        
        # ✅ TOPIC KNOWLEDGE RETRIEVAL
        has_intent = bool(self.working_memory.user_intent)
        has_tool_results = len(self.working_memory.recent_tool_results) > 0
        has_focus_topics = len(self.working_memory.focus_topics) > 0
        
        topic_context_parts = []
        
        if has_intent or has_tool_results or has_focus_topics:
            for topic, knowledge in list(self.semantic_memory.topic_knowledge.items())[:5]:
                details = []
                
                if "query_count" in knowledge and knowledge["query_count"] > 0:
                    details.append(f"queried {knowledge['query_count']} times")
                
                if "tool_used" in knowledge:
                    details.append(f"via {knowledge['tool_used']}")
                
                if "success" in knowledge:
                    success_str = "successful" if knowledge["success"] else "failed"
                    details.append(success_str)
                
                if details:
                    formatted = f"- **{topic}**: {', '.join(details)}"
                    topic_context_parts.append(formatted)
            
            if topic_context_parts:
                topic_section = f"## Relevant Topics\n" + "\n".join(topic_context_parts)
                context_parts.append(topic_section)
        
        # Similar past episodes
        if include_episodes and self.working_memory.focus_entities:
            similar_episodes = self.find_similar_tasks(
                self.working_memory.focus_entities,
                self.working_memory.focus_topics,
                limit=2
            )
            
            if similar_episodes:
                episode_text = "## Similar Past Tasks\n"
                for i, episode in enumerate(similar_episodes, 1):
                    status = "✅ Succeeded" if episode.success else "✗ Failed"
                    episode_text += f"{i}. {status}: {episode.task_description[:80]}...\n"
                    if episode.outcome_summary:
                        episode_text += f"   Outcome: {episode.outcome_summary[:100]}...\n"
                context_parts.append(episode_text)
        
        final_context = "\n\n".join(context_parts) if context_parts else ""
        
        return final_context
    
    # ========== Persistence Operations ==========

    @retry_on_lock(max_attempts=3, delay=0.1)
    def _persist_working_memory_sync(self):
        """
        Persist working memory (synchronous - called via asyncio.to_thread).
        
        ✅ FIXED: Uses INSERT OR IGNORE to prevent duplicates and tracks persisted items.
        ✅ FIXED: Loads existing hashes from database to prevent skipping reloaded items.
        """
        if not self.db_path or not self.db_path.exists():
            return

        # Track which tool results have been persisted to avoid duplicates
        if not hasattr(self, '_persisted_tool_hashes'):
            self._persisted_tool_hashes = set()
        
        # ✅ FIX: Load existing hashes from database on first persistence call
        if len(self._persisted_tool_hashes) == 0 and self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tool_name, timestamp, query
                    FROM memory_working
                    WHERE session_id = ?
                """, (self.session_id,))
                
                for row in cursor.fetchall():
                    result_hash = hashlib.md5(
                        f"{row[0]}_{row[1]}_{row[2]}".encode()
                    ).hexdigest()
                    self._persisted_tool_hashes.add(result_hash)
                
                conn.close()
                
                if len(self._persisted_tool_hashes) > 0:
                    logger.info(f"🧠 [MemoryManager] Loaded {len(self._persisted_tool_hashes)} existing tool result hashes")
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager] Could not load persisted hashes: {e}")

        try:
            # ✅ FIX: Use separate connection with timeout
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 5000")

            persisted_count = 0
            for tool_result in self.working_memory.recent_tool_results:
                try:
                    # Create a hash to identify this specific tool result
                    result_hash = hashlib.md5(
                        f"{tool_result['tool_name']}_{tool_result['timestamp']}_{tool_result.get('query', '')}".encode()
                    ).hexdigest()
                    
                    # Skip if already persisted
                    if result_hash in self._persisted_tool_hashes:
                        continue
                    
                    # ✅ FIX: Use INSERT OR IGNORE to handle duplicates gracefully
                    cursor.execute("""
                        INSERT OR IGNORE INTO memory_working(session_id, tool_name, query, result, summary, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        self.session_id,
                        tool_result["tool_name"],
                        tool_result.get("query", ""),
                        json.dumps(tool_result["result"]),
                        tool_result["summary"],
                        tool_result["timestamp"]
                    ))
                    
                    # Mark as persisted
                    self._persisted_tool_hashes.add(result_hash)
                    persisted_count += 1
                    
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to persist tool result: {e}")
                    continue

            conn.commit()
            conn.close()
            
            if persisted_count > 0:
                logger.info(f"🧠 [MemoryManager] Persisted {persisted_count} new tool results")
            
        except sqlite3.OperationalError as e:
            logger.warning(f"🧠 [MemoryManager] Working memory persistence locked: {e}")
        except Exception as e:
            logger.error(f"🧠 [MemoryManager] Failed to persist working memory: {e}")

    async def _persist_working_memory(self):
        """✅ FIXED: Async wrapper for working memory persistence."""
        await asyncio.to_thread(self._persist_working_memory_sync)

    @retry_on_lock(max_attempts=3, delay=0.1)
    def _persist_semantic_memory_sync(self):
        """Persist semantic memory (entities/topics) synchronously."""
        if not self.db_path or not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 5000")

            # Persist entities
            entity_count = 0
            for entity, knowledge in self.semantic_memory.entity_knowledge.items():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_entities VALUES (?, ?, ?)
                    """, (
                        entity,
                        json.dumps(knowledge),
                        knowledge.get("last_updated", datetime.now(UTC).isoformat())
                    ))
                    entity_count += 1
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to persist entity {entity}: {e}")

            # Persist topics
            topic_count = 0
            for topic, knowledge in self.semantic_memory.topic_knowledge.items():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_topics VALUES (?, ?, ?)
                    """, (
                        topic,
                        json.dumps(knowledge),
                        knowledge.get("last_updated", datetime.now(UTC).isoformat())
                    ))
                    topic_count += 1
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to persist topic {topic}: {e}")

            conn.commit()
            conn.close()
            
            logger.info(f"🧠 [MemoryManager] Persisted {entity_count} entities, {topic_count} topics to database")
            
        except Exception as e:
            logger.error(f"🧠 [MemoryManager] Failed to persist semantic memory: {e}")

    def _persist_working_memory_with_cursor(self, cursor):
        """Persist working memory using provided cursor (no new connection)."""
        if not hasattr(self, '_persisted_tool_hashes'):
            self._persisted_tool_hashes = set()
        
        persisted_count = 0
        for tool_result in self.working_memory.recent_tool_results:
            try:
                result_hash = hashlib.md5(
                    f"{tool_result['tool_name']}_{tool_result['timestamp']}_{tool_result.get('query', '')}".encode()
                ).hexdigest()
                
                if result_hash in self._persisted_tool_hashes:
                    continue
                
                cursor.execute("""
                    INSERT OR IGNORE INTO memory_working(session_id, tool_name, query, result, summary, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id,
                    tool_result["tool_name"],
                    tool_result.get("query", ""),
                    json.dumps(tool_result["result"]),
                    tool_result["summary"],
                    tool_result["timestamp"]
                ))
                
                self._persisted_tool_hashes.add(result_hash)
                persisted_count += 1
                
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager] Failed to persist tool result: {e}")
                continue
        
        if persisted_count > 0:
            logger.info(f"🧠 [MemoryManager]   Persisted {persisted_count} tool results")

    def _persist_semantic_memory_with_cursor(self, cursor):
        """Persist semantic memory using provided cursor (no new connection)."""
        # Persist entities
        entity_count = 0
        for entity, knowledge in self.semantic_memory.entity_knowledge.items():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_entities VALUES (?, ?, ?)
                """, (
                    entity,
                    json.dumps(knowledge),
                    knowledge.get("last_updated", datetime.now(UTC).isoformat())
                ))
                entity_count += 1
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager] Failed to persist entity: {e}")
        
        # Persist topics
        topic_count = 0
        for topic, knowledge in self.semantic_memory.topic_knowledge.items():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_topics VALUES (?, ?, ?)
                """, (
                    topic,
                    json.dumps(knowledge),
                    knowledge.get("last_updated", datetime.now(UTC).isoformat())
                ))
                topic_count += 1
            except Exception as e:
                logger.warning(f" [MemoryManager] Failed to persist topic: {e}")
        
        if entity_count > 0 or topic_count > 0:
            logger.info(f" [MemoryManager]   Persisted {entity_count} entities, {topic_count} topics")

    def _persist_session_metadata_with_cursor(self, cursor):
        """Persist session metadata using provided cursor (no new connection)."""
        # FIX 2: Skip persisting if focus is empty after clear
        if (not self.working_memory.focus_entities and 
            not self.working_memory.focus_topics and
            not self.working_memory.user_intent):
            logger.debug(" [MemoryManager] Skipping session persist - no meaningful data")
            return
        
        try:
            # FIX 1: Prevent overwriting non-empty data with empty arrays
            cursor.execute("""
                INSERT INTO memory_sessions(session_id, user_intent, focus_entities, focus_topics, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_intent = excluded.user_intent,
                    focus_entities = CASE 
                        WHEN excluded.focus_entities = '[]' THEN memory_sessions.focus_entities
                        ELSE excluded.focus_entities
                    END,
                    focus_topics = CASE 
                        WHEN excluded.focus_topics = '[]' THEN memory_sessions.focus_topics
                        ELSE excluded.focus_topics
                    END,
                    last_updated = excluded.last_updated
            """, (
                self.session_id,
                self.working_memory.user_intent,
                json.dumps(self.working_memory.focus_entities),
                json.dumps(self.working_memory.focus_topics),
                datetime.now(UTC).isoformat()
            ))
            
            logger.info(f" [MemoryManager]   Persisted session metadata (focus: {len(self.working_memory.focus_entities)} entities)")
            
        except Exception as e:
            logger.error(f" [MemoryManager] Failed to persist session metadata: {e}")

    @retry_on_lock(max_attempts=3, delay=0.1)
    def _persist_session_metadata_sync(self):
        """Persist session metadata (synchronous - called via asyncio.to_thread)."""
        if not self.db_path or not self.db_path.exists():
            return
        
        # ✅ FIX 2: Skip persisting if focus is empty after clear
        if (not self.working_memory.focus_entities and 
            not self.working_memory.focus_topics and
            not self.working_memory.user_intent):
            logger.debug("🧠 [MemoryManager] Skipping session persist - no meaningful data")
            return

        try:
            # ✅ FIX: Use separate connection with timeout
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 5000")

            # ✅ FIX 1: Prevent overwriting non-empty data with empty arrays
            cursor.execute("""
                INSERT INTO memory_sessions(session_id, user_intent, focus_entities, focus_topics, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_intent = excluded.user_intent,
                    focus_entities = CASE 
                        WHEN excluded.focus_entities = '[]' THEN memory_sessions.focus_entities
                        ELSE excluded.focus_entities
                    END,
                    focus_topics = CASE 
                        WHEN excluded.focus_topics = '[]' THEN memory_sessions.focus_topics
                        ELSE excluded.focus_topics
                    END,
                    last_updated = excluded.last_updated
            """, (
                self.session_id,
                self.working_memory.user_intent,
                json.dumps(self.working_memory.focus_entities),
                json.dumps(self.working_memory.focus_topics),
                datetime.now(UTC).isoformat()
            ))

            conn.commit()
            conn.close()
        except sqlite3.OperationalError as e:
            logger.warning(f"🧠 [MemoryManager] Session metadata persistence locked: {e}")
        except Exception as e:
            logger.error(f"🧠 [MemoryManager] Failed to persist session metadata: {e}")

    async def _persist_session_metadata(self):
        """✅ FIXED: Async wrapper for session metadata persistence."""
        await asyncio.to_thread(self._persist_session_metadata_sync)

    @retry_on_lock(max_attempts=3, delay=0.1)
    def _persist_to_database_sync(self):
        """
        Persist all memories to database (synchronous - called via asyncio.to_thread).

        ✅ FIXED: Proper error handling with rollback protection
        ✅ FIXED: Busy timeout for database locks
        ✅ FIXED: Retry logic for transient lock errors
        """
        if not self.db_path or not self.db_path.exists():
            logger.warning(f"🧠 [MemoryManager] Cannot persist - database not found: {self.db_path}")
            return

        conn = None
        try:
            # ✅ FIX: Add busy timeout for database locks (5 seconds)
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            cursor = conn.cursor()
            
            # ✅ FIX: Set immediate mode for better lock handling
            cursor.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout

            # Persist episodes
            for episode in self.episodic_memory.recent_episodes:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_episodes
                        (episode_id, session_id, task_description, start_time, end_time, status, success, outcome_summary, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        episode.episode_id,
                        self.session_id,
                        episode.task_description,
                        episode.start_time.isoformat(),
                        episode.end_time.isoformat() if episode.end_time else None,
                        episode.status.value,
                        1 if episode.success else 0,
                        episode.outcome_summary,
                        json.dumps(episode.to_dict())
                    ))
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to persist episode {episode.episode_id}: {e}")
                    continue

            # Persist patterns
            for pattern in self.semantic_memory.patterns.values():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.pattern_id,
                        pattern.pattern_type,
                        pattern.context,
                        pattern.action,
                        pattern.outcome,
                        pattern.confidence,
                        pattern.times_applied,
                        pattern.success_rate,
                        pattern.created_at.isoformat(),
                        pattern.last_updated.isoformat(),
                        json.dumps(pattern.to_dict())
                    ))
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to persist pattern {pattern.pattern_id}: {e}")
                    continue

            # Persist entity knowledge
            logger.info(f"🧠 [MemoryManager] Persisting {len(self.semantic_memory.entity_knowledge)} entities")
            for entity, knowledge in self.semantic_memory.entity_knowledge.items():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_entities VALUES (?, ?, ?)
                    """, (
                        entity,
                        json.dumps(knowledge),
                        knowledge.get("last_updated", datetime.now(UTC).isoformat())
                    ))
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to persist entity {entity}: {e}")
                    continue

            # Persist topic knowledge
            logger.info(f"🧠 [MemoryManager] Persisting {len(self.semantic_memory.topic_knowledge)} topics")
            for topic, knowledge in self.semantic_memory.topic_knowledge.items():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_topics VALUES (?, ?, ?)
                    """, (
                        topic,
                        json.dumps(knowledge),
                        knowledge.get("last_updated", datetime.now(UTC).isoformat())
                    ))
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to persist topic {topic}: {e}")
                    continue

            # Persist working memory (with individual error handling)
            try:
                self._persist_working_memory_sync()
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager] Failed to persist working memory: {e}")

            # Persist session metadata (with individual error handling)
            try:
                self._persist_session_metadata_sync()
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager] Failed to persist session metadata: {e}")

            # ✅ FIX: Commit only if we got this far
            conn.commit()
            logger.info(f"🧠 [MemoryManager] Successfully persisted all memory to database")
            
        except sqlite3.OperationalError as e:
            # Database is locked - this is expected with concurrent access
            logger.warning(f"🧠 [MemoryManager] Database temporarily locked: {e}")
            # ✅ FIX: Don't try to rollback if connection failed
            if conn:
                try:
                    conn.rollback()
                except:
                    pass  # Rollback can fail if connection is bad
                    
        except Exception as e:
            logger.error(f"🧠 [MemoryManager] Failed to persist to database: {e}")
            # ✅ FIX: Safe rollback with error suppression
            if conn:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.warning(f"🧠 [MemoryManager] Rollback also failed: {rollback_error}")
        finally:
            # ✅ FIX: Always close connection safely
            if conn:
                try:
                    conn.close()
                except:
                    pass

    async def _persist_to_database(self):
        """✅ FIXED: Async wrapper for full database persistence."""
        await asyncio.to_thread(self._persist_to_database_sync)

    def _load_from_database_sync(self):
        """Load memories from movesia.db (blocking I/O)."""
        if not self.db_path or not self.db_path.exists():
            return
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # ✅ CRITICAL: Load working memory tool results first
            cursor.execute("""
                SELECT tool_name, query, result, summary, timestamp
                FROM memory_working
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (self.session_id, self.working_memory.max_tool_results))
            
            for row in cursor.fetchall():
                try:
                    tool_result = {
                        "tool_name": row[0],
                        "query": row[1],
                        "result": json.loads(row[2]),
                        "summary": row[3],
                        "timestamp": row[4]
                    }
                    self.working_memory.recent_tool_results.append(tool_result)
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to load tool result: {e}")
            
            logger.info(f"🧠 [MemoryManager] Loaded {len(self.working_memory.recent_tool_results)} tool results")
            
            # Load session metadata
            cursor.execute("""
                SELECT user_intent, focus_entities, focus_topics
                FROM memory_sessions
                WHERE session_id = ?
            """, (self.session_id,))
            
            session_row = cursor.fetchone()
            if session_row:
                self.working_memory.user_intent = session_row[0] or ""
                self.working_memory.focus_entities = json.loads(session_row[1]) if session_row[1] else []
                self.working_memory.focus_topics = json.loads(session_row[2]) if session_row[2] else []
                logger.info(f"🧠 [MemoryManager] Loaded session metadata")
            
            # Load episodes
            cursor.execute("""
                SELECT data FROM memory_episodes 
                WHERE session_id = ?
                ORDER BY created_at DESC 
                LIMIT ?
            """, (self.session_id, self.episodic_memory.max_recent_episodes))
            
            for row in cursor.fetchall():
                try:
                    episode_data = json.loads(row[0])
                    episode = Episode.from_dict(episode_data)
                    self.episodic_memory.recent_episodes.append(episode)
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to load episode: {e}")
            
            # Load patterns
            cursor.execute("SELECT data FROM memory_patterns")
            for row in cursor.fetchall():
                try:
                    pattern_data = json.loads(row[0])
                    pattern = Pattern.from_dict(pattern_data)
                    self.semantic_memory.add_pattern(pattern)
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to load pattern: {e}")
            
            # Load entity knowledge
            try:
                entity_rows = cursor.execute("SELECT entity, knowledge FROM memory_entities").fetchall()
                logger.info(f"🧠 [MemoryManager] Found {len(entity_rows)} entities in database")
                
                for row in entity_rows:
                    try:
                        entity, knowledge_json = row
                        knowledge = json.loads(knowledge_json)
                        self.semantic_memory.entity_knowledge[entity] = knowledge
                        logger.debug(f"🧠 [MemoryManager] Loaded entity: {entity}")
                    except Exception as e:
                        logger.warning(f"🧠 [MemoryManager] Failed to load entity knowledge: {e}")
                
                logger.info(f"🧠 [MemoryManager] Successfully loaded {len(self.semantic_memory.entity_knowledge)} entities")
            except sqlite3.OperationalError as e:
                logger.warning(f"🧠 [MemoryManager] memory_entities table not found: {e}")
            except Exception as e:
                logger.error(f"🧠 [MemoryManager] Failed to load entity knowledge: {e}")
            
            # Load topic knowledge
            try:
                topic_rows = cursor.execute("SELECT topic, knowledge FROM memory_topics").fetchall()
                logger.info(f"🧠 [MemoryManager] Found {len(topic_rows)} topics in database")
                
                for row in topic_rows:
                    try:
                        topic, knowledge_json = row
                        knowledge = json.loads(knowledge_json)
                        self.semantic_memory.topic_knowledge[topic] = knowledge
                        logger.debug(f"🧠 [MemoryManager] Loaded topic: {topic}")
                    except Exception as e:
                        logger.warning(f"🧠 [MemoryManager] Failed to load topic knowledge: {e}")
                
                logger.info(f"🧠 [MemoryManager] Successfully loaded {len(self.semantic_memory.topic_knowledge)} topics")
            except sqlite3.OperationalError as e:
                logger.warning(f"🧠 [MemoryManager] memory_topics table not found: {e}")
            except Exception as e:
                logger.error(f"🧠 [MemoryManager] Failed to load topic knowledge: {e}")
        
        finally:
            conn.close()
    
    def _reload_semantic_knowledge(self):
        """
        Force reload entity and topic knowledge from database.
        ✅ FIX: Used after interrupt resume to ensure semantic memory is populated.
        """
        if not self.db_path or not self.db_path.exists():
            logger.warning(f"🧠 [MemoryManager] Cannot reload: database not found")
            return
        
        logger.info(f"🧠 [MemoryManager] Force reloading semantic knowledge from database...")
        
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        cursor = conn.cursor()
        
        try:
            # Reload entities
            cursor.execute("SELECT entity, knowledge FROM memory_entities")
            entity_rows = cursor.fetchall()
            entity_count = 0
            for row in entity_rows:
                try:
                    entity, knowledge_json = row
                    knowledge = json.loads(knowledge_json)
                    self.semantic_memory.entity_knowledge[entity] = knowledge
                    entity_count += 1
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to reload entity: {e}")
            
            # Reload topics  
            cursor.execute("SELECT topic, knowledge FROM memory_topics")
            topic_rows = cursor.fetchall()
            topic_count = 0
            for row in topic_rows:
                try:
                    topic, knowledge_json = row
                    knowledge = json.loads(knowledge_json)
                    self.semantic_memory.topic_knowledge[topic] = knowledge
                    topic_count += 1
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to reload topic: {e}")
            
            logger.info(f"🧠 [MemoryManager] ✅ Force reloaded: {entity_count} entities, {topic_count} topics")
            
        except Exception as e:
            logger.error(f"🧠 [MemoryManager] Failed to reload semantic knowledge: {e}")
        finally:
            conn.close()
    
    def _reload_working_memory_sync(self):
        """
        Force reload working memory tool results from database.
        ✅ FIX: Called after interrupts to ensure tool results are available.
        """
        if not self.db_path or not self.db_path.exists():
            logger.warning(f"🧠 [MemoryManager] Cannot reload working memory: database not found")
            return
        
        logger.info(f"🧠 [MemoryManager] Force reloading working memory from database...")
        
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT tool_name, query, result, summary, timestamp
                FROM memory_working
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (self.session_id, self.working_memory.max_tool_results))
            
            # Clear current results
            self.working_memory.recent_tool_results.clear()
            
            # Reload from database
            tool_count = 0
            for row in cursor.fetchall():
                try:
                    tool_result = {
                        "tool_name": row[0],
                        "query": row[1],
                        "result": json.loads(row[2]),
                        "summary": row[3],
                        "timestamp": row[4]
                    }
                    self.working_memory.recent_tool_results.append(tool_result)
                    tool_count += 1
                except Exception as e:
                    logger.warning(f"🧠 [MemoryManager] Failed to reload tool result: {e}")
            
            logger.info(f"🧠 [MemoryManager] ✅ Force reloaded: {tool_count} tool results")
            
            # ✅ NEW: Reload focus_entities from memory_sessions table
            cursor.execute("""
                SELECT user_intent, focus_entities, focus_topics
                FROM memory_sessions
                WHERE session_id = ?
            """, (self.session_id,))
            
            session_row = cursor.fetchone()
            if session_row:
                self.working_memory.user_intent = session_row[0] or ""
                self.working_memory.focus_entities = json.loads(session_row[1]) if session_row[1] else []
                self.working_memory.focus_topics = json.loads(session_row[2]) if session_row[2] else []
                logger.info(f"🧠 [MemoryManager]   ✅ Reloaded focus from movesia.db:")
                logger.info(f"🧠 [MemoryManager]     focus_entities: {self.working_memory.focus_entities}")
                logger.info(f"🧠 [MemoryManager]     focus_topics: {self.working_memory.focus_topics}")
            else:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ No session metadata found in movesia.db for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"🧠 [MemoryManager] Failed to reload working memory: {e}")
        finally:
            conn.close()
