"""
TIER 1: Working Memory (Short-term, current task context).

✅ FIXED: Tool result summaries now work correctly
✅ FIXED: Better handling of different tool result formats
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemory:
    """
    Working memory stores current task context and recent interactions.
    
    ✅ CHANGES:
    - Fixed tool result summary generation
    - Better handling of different result formats
    - More robust error handling
    """
    
    # Current context
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    user_intent: str = ""
    current_plan: Optional[Dict[str, Any]] = None
    
    # Focus tracking
    focus_entities: List[str] = field(default_factory=list)
    focus_topics: List[str] = field(default_factory=list)
    
    # Tool results storage
    recent_tool_results: List[Dict[str, Any]] = field(default_factory=list)
    max_tool_results: int = 10
    tool_result_ttl_minutes: int = 30
    
    # Limits
    max_recent_messages: int = 10
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to working memory with automatic pruning."""
        self.recent_messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": metadata or {}
        })
        
        # Prune old messages
        if len(self.recent_messages) > self.max_recent_messages:
            self.recent_messages = self.recent_messages[-self.max_recent_messages:]
    
    def add_tool_result(self, tool_name: str, result: Dict[str, Any], query: str = ""):
        """
        Add tool result to working memory with automatic pruning.
        
        ✅ FIXED: Now creates proper summaries for all tool types
        """
        tool_entry = {
            "tool_name": tool_name,
            "result": result,
            "result_raw": result,  # ✅ ADD: Keep full original result
            "query": query,
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": self._create_tool_result_summary(tool_name, result, query)
        }
        
        self.recent_tool_results.append(tool_entry)
        
        # Prune by age first (remove results older than TTL)
        cutoff_time = datetime.now(UTC) - timedelta(minutes=self.tool_result_ttl_minutes)
        self.recent_tool_results = [
            entry for entry in self.recent_tool_results
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
        
        # Then prune by count
        if len(self.recent_tool_results) > self.max_tool_results:
            self.recent_tool_results = self.recent_tool_results[-self.max_tool_results:]
    
    def _create_tool_result_summary(self, tool_name: str, result: Dict[str, Any], query: str = "") -> str:
        """
        Create a human-readable summary of tool result for context.
        
        ✅ FIXED: Now handles all result formats correctly
        """
        # Handle case where result might not be a dict
        if not isinstance(result, dict):
            return f"{tool_name} completed"
        
        # Check for explicit failure
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown error")
            return f"{tool_name} failed: {error_msg[:50]}"
        
        # Tool-specific success summaries
        if tool_name == "search_project":
            # ✅ FIXED: Properly extract result count
            result_count = result.get("result_count", 0)
            results = result.get("results", [])
            
            # ✅ Check for structured results with size data
            structured = result.get("results_structured", [])
            if structured and isinstance(structured, list):
                # Log that we have size data available
                has_size_data = any('size' in item for item in structured if isinstance(item, dict))
                if has_size_data:
                    logger.info(f"📊 [WorkingMemory] Size data available for {len(structured)} items")
            
            if result_count > 0 or results:
                count = result_count if result_count > 0 else len(results)
                
                # Get a few item names if available
                item_names = []
                if isinstance(results, list) and len(results) > 0:
                    for item in results[:3]:
                        if isinstance(item, dict):
                            name = item.get("name") or item.get("path") or item.get("guid", "")
                            if name:
                                # Get just the filename if it's a path
                                if "/" in name:
                                    name = name.split("/")[-1]
                                item_names.append(name)
                
                if item_names:
                    return f"Found {count} project items: {', '.join(item_names[:3])}"
                else:
                    return f"Found {count} project items"
            else:
                return "No project items found"
        
        elif tool_name == "code_snippets":
            # ✅ FIXED: Properly extract snippet count
            snippets = result.get("snippets", [])
            total_found = result.get("total_found", len(snippets))
            
            if total_found > 0 or snippets:
                count = total_found if total_found > 0 else len(snippets)
                
                # Get file names if available
                file_names = []
                if isinstance(snippets, list) and len(snippets) > 0:
                    for snippet in snippets[:3]:
                        if isinstance(snippet, dict):
                            file_path = snippet.get("file_path", "")
                            if file_path:
                                # Get just the filename
                                if "/" in file_path:
                                    file_path = file_path.split("/")[-1]
                                file_names.append(file_path)
                
                if file_names:
                    return f"Found {count} C# scripts: {', '.join(file_names[:3])}"
                else:
                    return f"Found {count} C# scripts"
            else:
                return "No scripts found"
        
        elif tool_name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
            # ✅ FIX: Check both top-level and pending_operation for file path
            file_path = result.get("file_path") or result.get("pending_operation", {}).get("rel_path", "unknown")
            
            # Get just the filename
            if "/" in file_path:
                file_path = file_path.split("/")[-1]
            
            if tool_name == "read_file":
                line_count = result.get("line_count", 0)
                return f"Read {file_path} ({line_count} lines)"
            elif tool_name == "write_file":
                line_count = result.get("line_count", 0)
                return f"Wrote {file_path} ({line_count} lines)"
            elif tool_name == "modify_file":
                modifications = result.get("modifications_applied", 0)
                return f"Modified {file_path} ({modifications} changes)"
            elif tool_name == "delete_file":
                return f"Deleted {file_path}"
            elif tool_name == "move_file":
                to_path = result.get("to_path", "unknown")
                if "/" in to_path:
                    to_path = to_path.split("/")[-1]
                return f"Moved {file_path} to {to_path}"
            else:
                return f"{tool_name}: {file_path}"
        
        elif tool_name == "web_search":
            results = result.get("results", [])
            result_count = result.get("result_count", len(results))

            if result_count > 0 or results:
                count = result_count if result_count > 0 else len(results)
                return f"Found {count} web resources"
            else:
                return "No web resources found"

        elif tool_name == "unity_docs":
            # ✅ ADD THIS CASE
            results = result.get("results", [])
            result_count = result.get("result_count", len(results))

            if result_count > 0:
                # Get titles of top results
                titles = []
                for doc in results[:3]:
                    if isinstance(doc, dict):
                        title = doc.get("title", "")
                        if title:
                            titles.append(title)

                if titles:
                    return f"Found {result_count} Unity docs: {', '.join(titles[:3])}"
                else:
                    return f"Found {result_count} Unity documentation pages"
            else:
                return "No Unity docs found"

        # Generic fallback
        return f"{tool_name} completed successfully"
    
    def get_recent_tool_context(self, limit: int = 3) -> str:
        """
        Get formatted context from recent tool results.
        
        ✅ FIXED: Now returns properly formatted context
        """
        if not self.recent_tool_results:
            return ""
        
        context_parts = []
        for entry in self.recent_tool_results[-limit:]:
            context_parts.append(f"- {entry['summary']}")
        
        return "Recent tool results:\n" + "\n".join(context_parts)
    
    def update_focus(self, entities: List[str], topics: List[str]):
        """Update current focus (what we're working on now)."""
        # Merge with existing focus, keeping unique items
        for entity in entities:
            if entity not in self.focus_entities:
                self.focus_entities.append(entity)
        
        for topic in topics:
            if topic not in self.focus_topics:
                self.focus_topics.append(topic)
        
        # Keep focus manageable (top 5 of each)
        self.focus_entities = self.focus_entities[-5:]
        self.focus_topics = self.focus_topics[-5:]
    
    def get_context_summary(self) -> str:
        """Get a human-readable summary of current context."""
        if not self.user_intent and not self.focus_entities and not self.focus_topics:
            return "No active context"
        
        parts = []
        
        if self.user_intent:
            intent_preview = self.user_intent[:60] + "..." if len(self.user_intent) > 60 else self.user_intent
            parts.append(f"Working on: {intent_preview}")
        
        if self.focus_entities:
            entities_str = ", ".join(self.focus_entities[:3])
            if len(self.focus_entities) > 3:
                entities_str += f" (+{len(self.focus_entities) - 3} more)"
            parts.append(f"Focused on: {entities_str}")
        
        if self.focus_topics:
            topics_str = ", ".join(self.focus_topics[:3])
            if len(self.focus_topics) > 3:
                topics_str += f" (+{len(self.focus_topics) - 3} more)"
            parts.append(f"Topics: {topics_str}")
        
        # ✅ NEW: Include recent tool results in summary
        if self.recent_tool_results:
            latest = self.recent_tool_results[-1]
            parts.append(f"Last action: {latest['summary']}")
        
        return " | ".join(parts)
    
    def clear(self):
        """
        Clear working memory (called between major tasks).
        
        ✅ NOTE: Tool results persist and decay naturally based on TTL
        """
        self.recent_messages = []
        self.user_intent = ""
        self.current_plan = None
        self.focus_entities = []
        self.focus_topics = []
        # Tool results are NOT cleared - they persist across tasks
    
    def get_recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent messages for context injection."""
        return self.recent_messages[-limit:] if self.recent_messages else []
    
    def has_active_context(self) -> bool:
        """Check if there's active context worth considering."""
        return bool(
            self.user_intent or 
            self.focus_entities or 
            self.focus_topics or 
            self.current_plan or
            self.recent_tool_results  # ✅ NEW: Check tool results too
        )
