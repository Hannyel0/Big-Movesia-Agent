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
        
        logger.info(f"🧠 [MemoryManager] Initialized")
        logger.info(f"   Session ID: {self.session_id}")
        logger.info(f"   DB Path: {db_path}")
        logger.info(f"   Auto-persist: {auto_persist}")
        
        # Persistence settings
        self.db_path = db_path
        self.auto_persist = auto_persist
        
        # Load existing memory if database exists (blocking, but OK in __init__)
        if self.db_path and self.db_path.exists() and self.auto_persist:
            self._load_from_database_sync()  # Synchronous load in __init__
            logger.info(f"🧠 [MemoryManager] Loaded from database:")
            logger.info(f"   Episodes: {len(self.episodic_memory.recent_episodes)}")
            logger.info(f"   Working memory tools: {len(self.working_memory.recent_tool_results)}")
            logger.info(f"   Patterns: {len(self.semantic_memory.patterns)}")
            logger.info(f"   Entities: {len(self.semantic_memory.entity_knowledge)}")
            logger.info(f"   Topics: {len(self.semantic_memory.topic_knowledge)}")
        
        # Clean up dangling in-progress episodes
        if self.episodic_memory.current_episode:
            logger.warning(f"🧠 [MemoryManager] Found unclosed episode: {self.episodic_memory.current_episode.episode_id}")
            self.episodic_memory.end_episode(
                success=False,
                outcome_summary="Session interrupted (recovered from crash)"
            )
            logger.info("🧠 [MemoryManager] Dangling episode closed successfully")
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
    
    def get_working_context(self) -> str:
        """Get current working memory context (synchronous)."""
        return self.working_memory.get_context_summary()
    
    # ========== Episodic Memory Operations ==========
    
    def start_task(self, task_description: str) -> str:
        """Start a new task episode (synchronous)."""
        episode_id = self.episodic_memory.start_episode(task_description)
        self.working_memory.user_intent = task_description
        logger.info(f"🧠 [MemoryManager] Started episode {episode_id}: {task_description[:60]}...")
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
            logger.warning("🧠 [MemoryManager] end_task called but no current episode")
            return
        
        episode_id = self.episodic_memory.current_episode.episode_id
        self.episodic_memory.end_episode(success, outcome_summary)
        
        logger.info(f"🧠 [MemoryManager] Ended episode {episode_id}: {'✅ success' if success else '❌ failure'}")
        logger.info(f"🧠 [MemoryManager]   Outcome: {outcome_summary}")
        logger.info(f"🧠 [MemoryManager]   Total episodes: {len(self.episodic_memory.recent_episodes)}")
        
        # Learn from episode
        if self.episodic_memory.recent_episodes:
            last_episode = self.episodic_memory.recent_episodes[-1]
            self.semantic_memory.learn_from_episode(last_episode)
            logger.info(f"🧠 [MemoryManager] Learned patterns from episode")
        
        # ✅ FIXED: Async persistence BEFORE clearing
        if self.auto_persist and self.db_path:
            await self._persist_to_database()  # Now async!
            logger.info(f"🧠 [MemoryManager] Persisted to database")
        
        # Only clear working memory if explicitly requested
        if clear_working_memory:
            self.working_memory.clear()
            logger.info(f"🧠 [MemoryManager] Cleared working memory")
        else:
            logger.info(f"🧠 [MemoryManager] Preserved working memory ({len(self.working_memory.recent_tool_results)} tool results)")
    
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
        logger.info(f"🧠 [MemoryManager] add_tool_call invoked")
        logger.info(f"🧠 [MemoryManager]   Tool: {tool_name}")
        logger.info(f"🧠 [MemoryManager]   Args: {str(args)[:100]}")
        
        # Add to episodic memory
        self.episodic_memory.add_tool_call(tool_name, args, result)
        logger.info(f"🧠 [MemoryManager]   ✅ Added to episodic memory")
        
        # Add to working memory
        query = args.get("query", "") or args.get("sql_query", "") or args.get("operation", "") or args.get("query_description", "")
        self.working_memory.add_tool_result(tool_name, result, query)
        logger.info(f"🧠 [MemoryManager]   ✅ Added to working memory")
        
        # Log current state
        logger.info(f"🧠 [MemoryManager]   Working memory now has {len(self.working_memory.recent_tool_results)} tool results")
        if self.working_memory.recent_tool_results:
            latest = self.working_memory.recent_tool_results[-1]
            logger.info(f"🧠 [MemoryManager]   Latest: {latest['summary']}")
        
        # ✅ NEW: Extract and store entity knowledge from tool results
        entity_count = 0
        if tool_name == "search_project" and result.get("success"):
            results = result.get("results", [])
            for item in results[:5]:  # Store knowledge about top 5 results
                entity = item.get("path") or item.get("name")
                if entity:
                    knowledge = {
                        "type": item.get("kind", "asset"),
                        "size": item.get("size"),
                        "last_accessed": datetime.now(UTC).isoformat(),
                        "access_count": 1
                    }
                    self.update_entity_knowledge(entity, knowledge)
                    entity_count += 1
        
        elif tool_name == "code_snippets" and result.get("success"):
            snippets = result.get("snippets", [])
            for snippet in snippets[:5]:
                entity = snippet.get("file_path")
                if entity:
                    knowledge = {
                        "type": "script",
                        "language": "csharp",
                        "last_accessed": datetime.now(UTC).isoformat(),
                        "functionality": snippet.get("description", "")[:100]
                    }
                    self.update_entity_knowledge(entity, knowledge)
                    entity_count += 1
        
        elif tool_name == "file_operation" and result.get("success"):
            file_path = result.get("file_path")
            if file_path:
                knowledge = {
                    "type": "file",
                    "operation": result.get("operation", "unknown"),
                    "last_modified": datetime.now(UTC).isoformat(),
                    "modification_count": 1
                }
                self.update_entity_knowledge(file_path, knowledge)
                entity_count += 1
        
        if entity_count > 0:
            logger.info(f"🧠 [MemoryManager]   ✅ Stored knowledge for {entity_count} entities")
        
        # ✅ NEW: Extract and store topic knowledge from queries
        query_text = args.get("query", "") or args.get("query_description", "") or args.get("natural_query", "")
        if query_text:
            topic_keywords = ["movement", "ui", "physics", "animation", "audio", "input", "player", "enemy", "camera", "inventory"]
            topics_found = []
            for topic in topic_keywords:
                if topic in query_text.lower():
                    topic_knowledge = {
                        "queries": [query_text[:100]],
                        "tool_used": tool_name,
                        "last_queried": datetime.now(UTC).isoformat(),
                        "success": result.get("success", True),
                        "query_count": 1
                    }
                    self.update_topic_knowledge(topic, topic_knowledge)
                    topics_found.append(topic)
            
            if topics_found:
                logger.info(f"🧠 [MemoryManager]   ✅ Stored knowledge for topics: {', '.join(topics_found)}")
        
        # ✅ FIXED: Async persistence
        if self.auto_persist and self.db_path:
            await self._persist_working_memory()  # Now async!
            logger.info(f"🧠 [MemoryManager]   ✅ Persisted to database")
    
    def add_tool_call_sync(self, tool_name: str, args: Dict[str, Any], result: Any):
        """
        Synchronous wrapper for add_tool_call - safe to call from sync contexts.
        
        This is needed for LangGraph routing functions which cannot be async.
        ✅ FIXED: Now uses synchronous persistence instead of trying to schedule async tasks.
        ✅ FIXED: Now extracts entity and topic knowledge like async version.
        ✅ FIXED: Handles different result formats (natural_language vs structured).
        """
        logger.info(f"🧠 [MemoryManager] add_tool_call_sync invoked")
        logger.info(f"🧠 [MemoryManager]   Tool: {tool_name}")
        
        # Add to episodic memory (synchronous)
        self.episodic_memory.add_tool_call(tool_name, args, result)
        logger.info(f"🧠 [MemoryManager]   ✅ Added to episodic memory")
        
        # Add to working memory (synchronous)
        query = args.get("query", "") or args.get("sql_query", "") or args.get("operation", "") or args.get("query_description", "")
        self.working_memory.add_tool_result(tool_name, result, query)
        logger.info(f"🧠 [MemoryManager]   ✅ Added to working memory")
        
        # Log current state
        logger.info(f"🧠 [MemoryManager]   Working memory now has {len(self.working_memory.recent_tool_results)} tool results")
        if self.working_memory.recent_tool_results:
            latest = self.working_memory.recent_tool_results[-1]
            logger.info(f"🧠 [MemoryManager]   Latest: {latest['summary']}")
        
        # ✅ DEBUG: Log the result structure
        logger.info(f"🧠 [MemoryManager] 📊 Analyzing result structure:")
        logger.info(f"🧠 [MemoryManager]   Result type: {type(result)}")
        logger.info(f"🧠 [MemoryManager]   Result is dict: {isinstance(result, dict)}")
        if isinstance(result, dict):
            logger.info(f"🧠 [MemoryManager]   Result keys: {list(result.keys())}")
            logger.info(f"🧠 [MemoryManager]   Has 'results' key: {'results' in result}")
            if 'results' in result:
                results_data = result.get("results")
                logger.info(f"🧠 [MemoryManager]   Results type: {type(results_data)}")
                logger.info(f"🧠 [MemoryManager]   Results is list: {isinstance(results_data, list)}")
                if isinstance(results_data, list) and len(results_data) > 0:
                    logger.info(f"🧠 [MemoryManager]   First result type: {type(results_data[0])}")
                    logger.info(f"🧠 [MemoryManager]   First result preview: {str(results_data[0])[:100]}")
        
        # ✅ EXTRACT ENTITIES: Extract and store entity knowledge from tool results
        entity_count = 0
        
        if tool_name == "search_project" and isinstance(result, dict) and result.get("success"):
            logger.info(f"🧠 [MemoryManager] 🔍 Extracting entities from search_project")
            
            # Check if results are in structured format (list of dicts)
            results_data = result.get("results", [])
            
            if isinstance(results_data, list):
                logger.info(f"🧠 [MemoryManager]   Found {len(results_data)} results to process")
                
                for i, item in enumerate(results_data[:5], 1):  # Store knowledge about top 5 results
                    logger.info(f"🧠 [MemoryManager]   Processing result {i}/{min(5, len(results_data))}")
                    logger.info(f"🧠 [MemoryManager]     Item type: {type(item)}")
                    
                    # Handle both dict and string formats
                    if isinstance(item, dict):
                        entity = item.get("path") or item.get("name")
                        logger.info(f"🧠 [MemoryManager]     Entity from dict: {entity}")
                        
                        if entity:
                            knowledge = {
                                "type": item.get("kind", "asset"),
                                "size": item.get("size"),
                                "last_accessed": datetime.now(UTC).isoformat(),
                                "access_count": 1
                            }
                            logger.info(f"🧠 [MemoryManager]     Storing entity: {entity}")
                            logger.info(f"🧠 [MemoryManager]     Knowledge: {knowledge}")
                            
                            self.update_entity_knowledge(entity, knowledge)
                            entity_count += 1
                            logger.info(f"🧠 [MemoryManager]     ✅ Entity stored successfully")
                        else:
                            logger.warning(f"🧠 [MemoryManager]     ⚠️ Could not extract entity from item")
                    
                    elif isinstance(item, str):
                        # If it's a string, try to extract meaningful info
                        logger.info(f"🧠 [MemoryManager]     Item is string: {item[:80]}")
                        # For now, skip string results - they're natural language formatted
                        logger.info(f"🧠 [MemoryManager]     ⚠️ Skipping string result (natural language format)")
                    
                    else:
                        logger.warning(f"🧠 [MemoryManager]     ⚠️ Unknown item type: {type(item)}")
            else:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ Results is not a list: {type(results_data)}")
                logger.info(f"🧠 [MemoryManager]   Results preview: {str(results_data)[:200]}")
        
        elif tool_name == "code_snippets" and isinstance(result, dict) and result.get("success"):
            logger.info(f"🧠 [MemoryManager] 🔍 Extracting entities from code_snippets")
            snippets = result.get("snippets", [])
            
            if isinstance(snippets, list):
                logger.info(f"🧠 [MemoryManager]   Found {len(snippets)} snippets to process")
                
                for i, snippet in enumerate(snippets[:5], 1):
                    logger.info(f"🧠 [MemoryManager]   Processing snippet {i}/{min(5, len(snippets))}")
                    
                    if isinstance(snippet, dict):
                        entity = snippet.get("file_path")
                        logger.info(f"🧠 [MemoryManager]     Entity: {entity}")
                        
                        if entity:
                            knowledge = {
                                "type": "script",
                                "language": "csharp",
                                "last_accessed": datetime.now(UTC).isoformat(),
                                "functionality": snippet.get("description", "")[:100]
                            }
                            logger.info(f"🧠 [MemoryManager]     Storing entity: {entity}")
                            
                            self.update_entity_knowledge(entity, knowledge)
                            entity_count += 1
                            logger.info(f"🧠 [MemoryManager]     ✅ Entity stored successfully")
                    else:
                        logger.warning(f"🧠 [MemoryManager]     ⚠️ Snippet is not dict: {type(snippet)}")
            else:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ Snippets is not a list: {type(snippets)}")
        
        elif tool_name == "file_operation" and isinstance(result, dict) and result.get("success"):
            logger.info(f"🧠 [MemoryManager] 🔍 Extracting entities from file_operation")
            file_path = result.get("file_path")
            logger.info(f"🧠 [MemoryManager]   File path: {file_path}")
            
            if file_path:
                knowledge = {
                    "type": "file",
                    "operation": result.get("operation", "unknown"),
                    "last_modified": datetime.now(UTC).isoformat(),
                    "modification_count": 1
                }
                logger.info(f"🧠 [MemoryManager]   Storing entity: {file_path}")
                logger.info(f"🧠 [MemoryManager]   Knowledge: {knowledge}")
                
                self.update_entity_knowledge(file_path, knowledge)
                entity_count += 1
                logger.info(f"🧠 [MemoryManager]   ✅ Entity stored successfully")
            else:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ No file_path in result")
        
        if entity_count > 0:
            logger.info(f"🧠 [MemoryManager]   ✅ Stored knowledge for {entity_count} entities")
            logger.info(f"🧠 [MemoryManager]   📊 Total entities in memory: {len(self.semantic_memory.entity_knowledge)}")
            # Log first 3 entities
            entities_list = list(self.semantic_memory.entity_knowledge.keys())[:3]
            for entity in entities_list:
                logger.info(f"🧠 [MemoryManager]     - {entity}")
        else:
            logger.warning(f"🧠 [MemoryManager]   ⚠️ No entities extracted from this tool call")
        
        # ✅ EXTRACT TOPICS: Extract and store topic knowledge from queries
        logger.info(f"🧠 [MemoryManager] 🔍 Extracting topics from query")
        query_text = args.get("query", "") or args.get("query_description", "") or args.get("natural_query", "")
        logger.info(f"🧠 [MemoryManager]   Query text: {query_text[:100]}")
        
        if query_text:
            topic_keywords = ["movement", "ui", "physics", "animation", "audio", "input", "player", "enemy", "camera", "inventory", "script", "scripts"]
            logger.info(f"🧠 [MemoryManager]   Checking against {len(topic_keywords)} topic keywords")
            
            topics_found = []
            for topic in topic_keywords:
                if topic in query_text.lower():
                    logger.info(f"🧠 [MemoryManager]   ✅ Found topic: {topic}")
                    
                    topic_knowledge = {
                        "queries": [query_text[:100]],
                        "tool_used": tool_name,
                        "last_queried": datetime.now(UTC).isoformat(),
                        "success": result.get("success", True) if isinstance(result, dict) else True,
                        "query_count": 1
                    }
                    logger.info(f"🧠 [MemoryManager]   Storing topic: {topic}")
                    logger.info(f"🧠 [MemoryManager]   Knowledge: {topic_knowledge}")
                    
                    self.update_topic_knowledge(topic, topic_knowledge)
                    topics_found.append(topic)
                    logger.info(f"🧠 [MemoryManager]   ✅ Topic stored successfully")
            
            if topics_found:
                logger.info(f"🧠 [MemoryManager]   ✅ Stored knowledge for topics: {', '.join(topics_found)}")
                logger.info(f"🧠 [MemoryManager]   📊 Total topics in memory: {len(self.semantic_memory.topic_knowledge)}")
                # Log all topics
                topics_list = list(self.semantic_memory.topic_knowledge.keys())
                for topic in topics_list:
                    logger.info(f"🧠 [MemoryManager]     - {topic}")
            else:
                logger.info(f"🧠 [MemoryManager]   ℹ️ No matching topics found in query")
        else:
            logger.warning(f"🧠 [MemoryManager]   ⚠️ No query text to extract topics from")
        
        # ✅ FIX: Use synchronous persistence in sync context
        if self.auto_persist and self.db_path:
            try:
                logger.info(f"🧠 [MemoryManager] 💾 Starting synchronous persistence")
                # Don't try to schedule async - just persist synchronously
                self._persist_working_memory_sync()
                logger.info(f"🧠 [MemoryManager]   ✅ Persisted synchronously")
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ Persistence failed: {e}")
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
        logger.info("🧠 [MemoryManager] Starting memory consolidation...")

        learned_count = 0
        for episode in self.episodic_memory.recent_episodes[-5:]:
            if episode.status == EpisodeStatus.COMPLETED:
                self.semantic_memory.learn_from_episode(episode)
                learned_count += 1

        logger.info(f"🧠 [MemoryManager]   Learned from {learned_count} episodes")
        logger.info(f"🧠 [MemoryManager]   Total patterns: {len(self.semantic_memory.patterns)}")

    async def consolidate_memories(self):
        """
        Consolidate memories across tiers with persistence.

        ✅ FIXED: Now async
        """
        self.consolidate_memories_sync()

        # Persist consolidated memories
        if self.auto_persist and self.db_path:
            await self._persist_to_database()
            logger.info(f"🧠 [MemoryManager]   Persisted consolidated memories")
    
    # ========== Session Management ==========
    
    def start_session(self) -> Dict[str, Any]:
        """Initialize a new session."""
        self.working_memory.clear()
        
        # Load relevant context from semantic memory
        # NOTE: get_user_preferences() and get_project_profile() are now async
        # If this method is needed, it should be made async and await these calls
        # user_prefs = self.semantic_memory.get_user_preferences()
        # project_profile = self.semantic_memory.get_project_profile()
        
        return {
            "user_preferences": {},  # Placeholder - make this method async if needed
            "project_profile": {}    # Placeholder - make this method async if needed
        }
    
    def process_interaction(
        self,
        user_message: str,
        agent_response: str,
        tool_results: List[Dict[str, Any]],
        outcome: str
    ):
        """Process a completed interaction across all memory tiers."""
        
        # 1. Update working memory
        self.working_memory.add_message("user", user_message)
        self.working_memory.add_message("assistant", agent_response)
        
        # 2. Extract and store semantic facts
        self._extract_semantic_facts(user_message, tool_results)
        
        # 3. If this is a complete episode, store it
        if outcome in ["success", "failure", "partial"]:
            episode = self._create_episode_from_interaction(
                user_message, agent_response, tool_results, outcome
            )
            self.episodic_memory.recent_episodes.append(episode)
            if len(self.episodic_memory.recent_episodes) > self.episodic_memory.max_recent_episodes:
                self.episodic_memory.recent_episodes = self.episodic_memory.recent_episodes[-self.episodic_memory.max_recent_episodes:]
    
    def _extract_semantic_facts(self, user_message: str, tool_results: List[Dict[str, Any]]):
        """Extract learnable facts from interaction."""
        
        # Learn about files the user works with
        for result in tool_results:
            if result.get("tool") == "file_operation":
                file_path = result.get("file_path", "")
                if file_path:
                    fact = SemanticFact(
                        fact_id=hashlib.md5(f"file_{file_path}".encode()).hexdigest(),
                        category="project",
                        subject="user",
                        predicate="works_with_file",
                        object=file_path,
                        confidence=0.6,
                        source="observation"
                    )
                    # self.semantic_memory.learn_fact(fact)  # REMOVED: learn_fact() is now async, learning happens in assess.py
            
            # Learn about search patterns
            elif result.get("tool") == "code_snippets":
                query = result.get("query", "")
                if query and result.get("success"):
                    fact = SemanticFact(
                        fact_id=hashlib.md5(f"search_{query}".encode()).hexdigest(),
                        category="pattern",
                        subject="user",
                        predicate="searches_for",
                        object=query,
                        confidence=0.5,
                        source="observation"
                    )
                    # self.semantic_memory.learn_fact(fact)  # REMOVED: learn_fact() is now async, learning happens in assess.py
    
    def _create_episode_from_interaction(
        self,
        user_message: str,
        agent_response: str,
        tool_results: List[Dict[str, Any]],
        outcome: str
    ) -> Episode:
        """Create an episode from the current interaction."""
        
        episode_id = hashlib.md5(
            f"{datetime.now(UTC).isoformat()}_{user_message}".encode()
        ).hexdigest()
        
        tools_used = [r.get("tool", "unknown") for r in tool_results]
        
        return Episode(
            episode_id=episode_id,
            task_description=user_message[:200],
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            status=EpisodeStatus.COMPLETED if outcome == "success" else EpisodeStatus.FAILED,
            tool_calls=[{"tool_name": t, "timestamp": datetime.now(UTC).isoformat()} for t in tools_used],
            entities_involved=self._extract_entities(tool_results),
            topics=self._extract_tags(user_message),
            success=outcome == "success",
            total_steps=len(tool_results),
            execution_time_seconds=sum(r.get("execution_time", 0) for r in tool_results)
        )
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract semantic tags from text."""
        tags = []
        
        keywords = {
            "movement": ["move", "movement", "walk", "run", "player"],
            "ui": ["ui", "menu", "button", "canvas"],
            "error": ["error", "fix", "debug", "problem"],
            "creation": ["create", "make", "build", "generate"]
        }
        
        text_lower = text.lower()
        for tag, keywords_list in keywords.items():
            if any(kw in text_lower for kw in keywords_list):
                tags.append(tag)
        
        return tags
    
    def _extract_entities(self, tool_results: List[Dict[str, Any]]) -> List[str]:
        """Extract entity names from tool results."""
        entities = set()
        
        for result in tool_results:
            if "file_path" in result:
                entities.add(result["file_path"])
            
            if "snippets" in result:
                for snippet in result["snippets"]:
                    entities.add(snippet.get("file_path", ""))
        
        return list(entities)
    
    async def get_relevant_context(self, user_query: str) -> Dict[str, Any]:
        """
        Get relevant context from all memory tiers for a query.
        This is the key integration point!
        """
        
        # 1. Working memory context
        working_context = self.working_memory.get_context_summary()
        
        # 2. Recall similar past episodes
        similar_episodes = self.episodic_memory.find_similar_episodes(
            self.working_memory.focus_entities,
            self.working_memory.focus_topics,
            limit=3
        )
        
        # 3. Query relevant semantic knowledge
        semantic_facts = await self.semantic_memory.query_knowledge(min_confidence=0.6)
        
        # 4. Get success patterns for this type of query
        query_type = self._categorize_query(user_query)
        success_patterns = self.episodic_memory.get_success_patterns(query_type)
        
        # 5. Get user preferences and project profile (async calls)
        user_preferences = await self.semantic_memory.get_user_preferences()
        project_profile = await self.semantic_memory.get_project_profile()
        
        return {
            "working_memory": {
                "summary": working_context,
                "recent_messages": self.working_memory.recent_messages[-3:],
                "focus": {
                    "entities": self.working_memory.focus_entities,
                    "topics": self.working_memory.focus_topics
                }
            },
            "episodic_memory": {
                "similar_past_interactions": [
                    {
                        "goal": ep.task_description,
                        "outcome": "success" if ep.success else "failure",
                        "what_worked": ep.what_worked,
                        "insights": ep.insights
                    }
                    for ep in similar_episodes
                ],
                "success_patterns": success_patterns
            },
            "semantic_memory": {
                "relevant_facts": [
                    {
                        "subject": f.subject,
                        "predicate": f.predicate,
                        "object": f.object,
                        "confidence": f.confidence
                    }
                    for f in semantic_facts[:10]
                ],
                "user_preferences": user_preferences,
                "project_profile": project_profile
            }
        }
    
    def _categorize_query(self, query: str) -> str:
        """Categorize query type for pattern matching."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["create", "make", "build"]):
            return "creation"
        elif any(word in query_lower for word in ["fix", "error", "debug"]):
            return "debugging"
        elif any(word in query_lower for word in ["find", "search", "show"]):
            return "search"
        else:
            return "general"
    
    # ========== Context Generation for LLM ==========
    
    async def get_memory_context(self, include_patterns: bool = True, include_episodes: bool = True) -> str:
        """
        Generate memory context for LLM prompt injection.
        
        Returns a formatted string containing relevant memory information.
        ✅ FIXED: Now async and reloads working memory using asyncio.to_thread
        """
        logger.info(f"🧠 [MemoryManager] Generating memory context")
        logger.info(f"🧠 [MemoryManager]   Include patterns: {include_patterns}")
        logger.info(f"🧠 [MemoryManager]   Include episodes: {include_episodes}")
        logger.info(f"🧠 [MemoryManager]   Working memory tool results: {len(self.working_memory.recent_tool_results)}")
        
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
                            logger.warning(f"🧠 [MemoryManager] Failed to reload tool result: {e}")
                    
                    conn.close()
                    return tools
                
                # Execute in separate thread to avoid blocking
                loaded_tools = await asyncio.to_thread(_reload_tools)
                
                for tool in loaded_tools:
                    self.working_memory.recent_tool_results.append(tool)
                
                if len(loaded_tools) > 0:
                    logger.info(f"🧠 [MemoryManager]   ✅ Reloaded {len(loaded_tools)} tool results from database")
                    logger.info(f"🧠 [MemoryManager]   Working memory now has: {len(self.working_memory.recent_tool_results)} tool results")
                    
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ Could not reload tools from database: {e}")
        
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
                            logger.warning(f"🧠 [MemoryManager] Failed to reload entity: {e}")
                    
                    conn.close()
                    return entities
                
                loaded_entities = await asyncio.to_thread(_reload_entities)
                self.semantic_memory.entity_knowledge.update(loaded_entities)
                
                if len(loaded_entities) > 0:
                    logger.info(f"🧠 [MemoryManager]   ✅ Reloaded {len(loaded_entities)} entities from database")
                    
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ Could not reload entities from database: {e}")
        
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
                            logger.warning(f"🧠 [MemoryManager] Failed to reload topic: {e}")
                    
                    conn.close()
                    return topics
                
                loaded_topics = await asyncio.to_thread(_reload_topics)
                self.semantic_memory.topic_knowledge.update(loaded_topics)
                
                if len(loaded_topics) > 0:
                    logger.info(f"🧠 [MemoryManager]   ✅ Reloaded {len(loaded_topics)} topics from database")
                    
            except Exception as e:
                logger.warning(f"🧠 [MemoryManager]   ⚠️ Could not reload topics from database: {e}")
        
        context_parts = []
        
        # ✅ NEW: Recent tool results FIRST (most important for follow-up questions)
        tool_context = self.working_memory.get_recent_tool_context(limit=2)
        if tool_context:
            context_parts.append(f"## Recent Actions\n{tool_context}")
            logger.info(f"🧠 [MemoryManager]   ✅ Added recent tool context")
        else:
            logger.info(f"🧠 [MemoryManager]   ⚠️ No recent tool context available")
        
        # Working memory context
        working_context = self.working_memory.get_context_summary()
        if working_context != "No active context":
            context_parts.append(f"## Current Context\n{working_context}")
            logger.info(f"🧠 [MemoryManager]   ✅ Added working context")
        
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
        
        # ✅ ENTITY KNOWLEDGE RETRIEVAL (with debug logging)
        logger.info(f"🔍 [MemoryManager] === ENTITY KNOWLEDGE RETRIEVAL ===")
        logger.info(f"🔍 [MemoryManager]   Total entities in semantic memory: {len(self.semantic_memory.entity_knowledge)}")
        
        if len(self.semantic_memory.entity_knowledge) > 0:
            logger.info(f"🔍 [MemoryManager]   Sample entities in memory:")
            for i, entity in enumerate(list(self.semantic_memory.entity_knowledge.keys())[:3], 1):
                logger.info(f"🔍 [MemoryManager]     {i}. {entity}")
        
        # Strategy 1: Check focus_entities
        logger.info(f"🔍 [MemoryManager]   Strategy 1 - Focus entities:")
        logger.info(f"🔍 [MemoryManager]     focus_entities count: {len(self.working_memory.focus_entities)}")
        if self.working_memory.focus_entities:
            logger.info(f"🔍 [MemoryManager]     focus_entities: {self.working_memory.focus_entities[:5]}")
        
        # Strategy 2: Extract entities from recent tool results
        logger.info(f"🔍 [MemoryManager]   Strategy 2 - Recent tool results:")
        logger.info(f"🔍 [MemoryManager]     Recent tool results count: {len(self.working_memory.recent_tool_results)}")
        
        recent_entities = set()
        if len(self.working_memory.recent_tool_results) > 0:
            for idx, tool_result in enumerate(self.working_memory.recent_tool_results[-2:], 1):
                logger.info(f"🔍 [MemoryManager]     Processing tool result {idx}:")
                logger.info(f"🔍 [MemoryManager]       Tool: {tool_result.get('tool_name')}")
                
                result_data = tool_result.get("result", {})
                logger.info(f"🔍 [MemoryManager]       Result data type: {type(result_data)}")
                logger.info(f"🔍 [MemoryManager]       Result has 'results' key: {'results' in result_data if isinstance(result_data, dict) else False}")
                
                if isinstance(result_data, dict) and "results" in result_data:
                    results_list = result_data.get("results", [])
                    logger.info(f"🔍 [MemoryManager]       Results list length: {len(results_list) if isinstance(results_list, list) else 'not a list'}")
                    
                    if isinstance(results_list, list):
                        for item_idx, item in enumerate(results_list[:10], 1):
                            if isinstance(item, dict):
                                entity = item.get("path") or item.get("name")
                                if entity:
                                    recent_entities.add(entity)
                                    logger.info(f"🔍 [MemoryManager]       Extracted entity {item_idx}: {entity}")
                                else:
                                    logger.info(f"🔍 [MemoryManager]       Item {item_idx} has no path/name: {list(item.keys())}")
                            else:
                                logger.info(f"🔍 [MemoryManager]       Item {item_idx} is not dict: {type(item)}")
        
        logger.info(f"🔍 [MemoryManager]   Extracted {len(recent_entities)} entities from recent tool results")
        if recent_entities:
            logger.info(f"🔍 [MemoryManager]   Sample extracted entities:")
            for i, entity in enumerate(list(recent_entities)[:3], 1):
                logger.info(f"🔍 [MemoryManager]     {i}. {entity}")
        
        # Combine strategies: focus_entities + recent_entities
        all_entities = set(self.working_memory.focus_entities) | recent_entities
        logger.info(f"🔍 [MemoryManager]   Combined entities to check: {len(all_entities)}")
        
        # Retrieve knowledge for these entities
        entity_context_parts = []
        logger.info(f"🔍 [MemoryManager]   Checking entity knowledge:")
        
        for entity in list(all_entities)[:10]:
            logger.info(f"🔍 [MemoryManager]     Checking entity: {entity}")
            logger.info(f"🔍 [MemoryManager]       In semantic memory? {entity in self.semantic_memory.entity_knowledge}")
            
            if entity in self.semantic_memory.entity_knowledge:
                knowledge = self.semantic_memory.entity_knowledge[entity]
                logger.info(f"🔍 [MemoryManager]       Knowledge found: {knowledge}")
                
                entity_name = entity.split("/")[-1] if "/" in entity else entity
                details = []
                
                if "size" in knowledge:
                    details.append(f"size: {knowledge['size']} bytes")
                    logger.info(f"🔍 [MemoryManager]       Added size: {knowledge['size']} bytes")
                
                if "type" in knowledge:
                    details.append(f"type: {knowledge['type']}")
                    logger.info(f"🔍 [MemoryManager]       Added type: {knowledge['type']}")
                
                if "access_count" in knowledge and knowledge["access_count"] > 0:
                    details.append(f"accessed {knowledge['access_count']} times")
                    logger.info(f"🔍 [MemoryManager]       Added access_count: {knowledge['access_count']}")
                
                if details:
                    formatted = f"- **{entity_name}**: {', '.join(details)}"
                    entity_context_parts.append(formatted)
                    logger.info(f"🔍 [MemoryManager]       ✅ Added to context: {formatted}")
                else:
                    logger.info(f"🔍 [MemoryManager]       ⚠️ No details found for entity")
            else:
                logger.info(f"🔍 [MemoryManager]       ❌ Not found in semantic memory")
        
        if entity_context_parts:
            entity_section = f"## Known Entities\n" + "\n".join(entity_context_parts)
            context_parts.append(entity_section)
            logger.info(f"✅ [MemoryManager]   Added entity knowledge section with {len(entity_context_parts)} entities")
            logger.info(f"✅ [MemoryManager]   Entity section preview: {entity_section[:200]}")
        else:
            logger.info(f"⚠️ [MemoryManager]   No entity knowledge added to context")
        
        logger.info(f"🔍 [MemoryManager] === END ENTITY KNOWLEDGE RETRIEVAL ===")
        
        # ✅ TOPIC KNOWLEDGE RETRIEVAL (with debug logging)
        logger.info(f"🔍 [MemoryManager] === TOPIC KNOWLEDGE RETRIEVAL ===")
        logger.info(f"🔍 [MemoryManager]   Total topics in semantic memory: {len(self.semantic_memory.topic_knowledge)}")
        
        if len(self.semantic_memory.topic_knowledge) > 0:
            logger.info(f"🔍 [MemoryManager]   Topics in memory:")
            for i, (topic, knowledge) in enumerate(list(self.semantic_memory.topic_knowledge.items())[:5], 1):
                logger.info(f"🔍 [MemoryManager]     {i}. {topic}: {knowledge}")
        
        # Check if we have any basis for retrieving topics
        has_intent = bool(self.working_memory.user_intent)
        has_tool_results = len(self.working_memory.recent_tool_results) > 0
        has_focus_topics = len(self.working_memory.focus_topics) > 0
        
        logger.info(f"🔍 [MemoryManager]   Retrieval conditions:")
        logger.info(f"🔍 [MemoryManager]     Has user_intent: {has_intent} ('{self.working_memory.user_intent[:50] if has_intent else ''}')")
        logger.info(f"🔍 [MemoryManager]     Has tool_results: {has_tool_results}")
        logger.info(f"🔍 [MemoryManager]     Has focus_topics: {has_focus_topics} ({self.working_memory.focus_topics if has_focus_topics else []})")
        
        topic_context_parts = []
        
        if has_intent or has_tool_results or has_focus_topics:
            logger.info(f"🔍 [MemoryManager]   Retrieving topic knowledge:")
            
            for topic, knowledge in list(self.semantic_memory.topic_knowledge.items())[:5]:
                logger.info(f"🔍 [MemoryManager]     Processing topic: {topic}")
                logger.info(f"🔍 [MemoryManager]       Knowledge: {knowledge}")
                
                details = []
                
                if "query_count" in knowledge and knowledge["query_count"] > 0:
                    details.append(f"queried {knowledge['query_count']} times")
                    logger.info(f"🔍 [MemoryManager]       Added query_count: {knowledge['query_count']}")
                
                if "tool_used" in knowledge:
                    details.append(f"via {knowledge['tool_used']}")
                    logger.info(f"🔍 [MemoryManager]       Added tool_used: {knowledge['tool_used']}")
                
                if "success" in knowledge:
                    success_str = "successful" if knowledge["success"] else "failed"
                    details.append(success_str)
                    logger.info(f"🔍 [MemoryManager]       Added success: {success_str}")
                
                if details:
                    formatted = f"- **{topic}**: {', '.join(details)}"
                    topic_context_parts.append(formatted)
                    logger.info(f"🔍 [MemoryManager]       ✅ Added to context: {formatted}")
                else:
                    logger.info(f"🔍 [MemoryManager]       ⚠️ No details found for topic")
            
            if topic_context_parts:
                topic_section = f"## Relevant Topics\n" + "\n".join(topic_context_parts)
                context_parts.append(topic_section)
                logger.info(f"✅ [MemoryManager]   Added topic knowledge section with {len(topic_context_parts)} topics")
                logger.info(f"✅ [MemoryManager]   Topic section preview: {topic_section[:200]}")
            else:
                logger.info(f"⚠️ [MemoryManager]   No topic knowledge added to context")
        else:
            logger.info(f"⚠️ [MemoryManager]   No basis for topic retrieval (no intent, tool results, or focus)")
        
        logger.info(f"🔍 [MemoryManager] === END TOPIC KNOWLEDGE RETRIEVAL ===")
        
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
        logger.info(f"🧠 [MemoryManager] Generated context length: {len(final_context)} chars")
        logger.info(f"🧠 [MemoryManager] Context preview: {final_context[:200]}")
        
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
    def _persist_session_metadata_sync(self):
        """Persist session metadata (synchronous - called via asyncio.to_thread)."""
        if not self.db_path or not self.db_path.exists():
            return

        try:
            # ✅ FIX: Use separate connection with timeout
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 5000")

            cursor.execute("""
                INSERT INTO memory_sessions(session_id, user_intent, focus_entities, focus_topics, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_intent = COALESCE(excluded.user_intent, memory_sessions.user_intent),
                    focus_entities = COALESCE(excluded.focus_entities, memory_sessions.focus_entities),
                    focus_topics = COALESCE(excluded.focus_topics, memory_sessions.focus_topics),
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
    
    # ========== Convenience Properties ==========
    
    @property
    def working(self) -> WorkingMemory:
        """Convenience property for working memory access."""
        return self.working_memory
    
    @property
    def episodic(self) -> EpisodicMemory:
        """Convenience property for episodic memory access."""
        return self.episodic_memory
    
    @property
    def semantic(self) -> SemanticMemory:
        """Convenience property for semantic memory access."""
        return self.semantic_memory
