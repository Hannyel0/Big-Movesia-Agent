"""
TIER 3: Semantic Memory (Long-term, learned patterns).

Analogous to human semantic memory (facts, concepts, skills).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from collections import defaultdict
import asyncio
import hashlib
import sqlite3
import json

from react_agent.memory.episodic import Episode, EpisodeStatus


@dataclass
class Pattern:
    """A learned pattern from past experiences."""
    
    pattern_id: str
    pattern_type: Literal["success", "failure", "optimization"]
    
    # Pattern description
    context: str  # When does this pattern apply?
    action: str   # What action to take?
    outcome: str  # What's the expected outcome?
    
    # Evidence
    supporting_episodes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    times_applied: int = 0
    success_rate: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "supporting_episodes": self.supporting_episodes,
            "confidence": self.confidence,
            "times_applied": self.times_applied,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Pattern:
        """Create pattern from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            context=data["context"],
            action=data["action"],
            outcome=data["outcome"],
            supporting_episodes=data.get("supporting_episodes", []),
            confidence=data.get("confidence", 0.5),
            times_applied=data.get("times_applied", 0),
            success_rate=data.get("success_rate", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            tags=data.get("tags", []),
        )


@dataclass
class SemanticFact:
    """A learned fact about the project or user preferences."""
    
    fact_id: str
    category: Literal["project", "user_preference", "domain_knowledge", "pattern"]
    
    subject: str  # What this is about
    predicate: str  # The relationship/property
    object: str  # The value/target
    
    confidence: float  # 0.0 to 1.0
    evidence_count: int = 1
    
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    source: str = "observation"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "category": self.category,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticFact:
        data["first_seen"] = datetime.fromisoformat(data["first_seen"])
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class SemanticMemory:
    """
    Semantic memory stores learned patterns and knowledge.
    
    Characteristics:
    - Long-term storage (persistent)
    - Stores abstract patterns, not specific events
    - Enables "I know that X usually leads to Y"
    - Continuously refined through experience
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./agent_memory/semantic.db")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory structures
        self.patterns: Dict[str, Pattern] = {}
        self.success_patterns: List[str] = []
        self.failure_patterns: List[str] = []
        self.optimization_patterns: List[str] = []
        
        # Knowledge bases
        self.entity_knowledge: Dict[str, Dict[str, Any]] = {}
        self.topic_knowledge: Dict[str, Dict[str, Any]] = {}
        
        self._init_storage()
        self._load_into_memory()
    
    def add_pattern(self, pattern: Pattern):
        """Add a new pattern to semantic memory."""
        self.patterns[pattern.pattern_id] = pattern
        
        # Categorize
        if pattern.pattern_type == "success":
            if pattern.pattern_id not in self.success_patterns:
                self.success_patterns.append(pattern.pattern_id)
        elif pattern.pattern_type == "failure":
            if pattern.pattern_id not in self.failure_patterns:
                self.failure_patterns.append(pattern.pattern_id)
        elif pattern.pattern_type == "optimization":
            if pattern.pattern_id not in self.optimization_patterns:
                self.optimization_patterns.append(pattern.pattern_id)
    
    def update_pattern(self, pattern_id: str, success: bool):
        """Update pattern based on application result."""
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        pattern.times_applied += 1
        
        # Update success rate
        if pattern.times_applied == 1:
            pattern.success_rate = 1.0 if success else 0.0
        else:
            # Exponential moving average
            alpha = 0.3
            pattern.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * pattern.success_rate
        
        # Update confidence based on success rate and number of applications
        pattern.confidence = min(0.95, pattern.success_rate * (1 - 1 / (pattern.times_applied + 1)))
        pattern.last_updated = datetime.now(UTC)
    
    def find_relevant_patterns(
        self,
        context: str,
        pattern_type: Optional[Literal["success", "failure", "optimization"]] = None,
        min_confidence: float = 0.5,
        limit: int = 5
    ) -> List[Pattern]:
        """Find patterns relevant to current context."""
        relevant = []
        
        # Filter by type if specified
        if pattern_type:
            pattern_ids = getattr(self, f"{pattern_type}_patterns", [])
            candidates = [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
        else:
            candidates = list(self.patterns.values())
        
        # Filter by confidence
        candidates = [p for p in candidates if p.confidence >= min_confidence]
        
        # Simple relevance scoring (could be enhanced with embeddings)
        context_words = set(context.lower().split())
        
        for pattern in candidates:
            pattern_words = set(pattern.context.lower().split())
            overlap = len(context_words & pattern_words)
            
            if overlap > 0:
                score = overlap * pattern.confidence * (1 + pattern.times_applied * 0.1)
                relevant.append((score, pattern))
        
        # Sort by score and return top N
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in relevant[:limit]]
    
    def learn_from_episode(self, episode: Episode):
        """Extract patterns from a completed episode."""
        # This is a simplified version - could be enhanced with LLM-based pattern extraction
        
        if episode.success and len(episode.tool_calls) > 0:
            # Create success pattern
            pattern_id = self._generate_pattern_id(episode.task_description, "success")
            
            # Check if pattern already exists
            if pattern_id in self.patterns:
                # Update existing pattern
                pattern = self.patterns[pattern_id]
                if episode.episode_id not in pattern.supporting_episodes:
                    pattern.supporting_episodes.append(episode.episode_id)
                pattern.confidence = min(0.95, pattern.confidence + 0.05)
                pattern.last_updated = datetime.now(UTC)
                return
            
            # Extract common tool sequence
            tool_sequence = " -> ".join([tc["tool_name"] for tc in episode.tool_calls[:3]])
            
            pattern = Pattern(
                pattern_id=pattern_id,
                pattern_type="success",
                context=f"Task: {episode.task_description[:100]}",
                action=f"Use tool sequence: {tool_sequence}",
                outcome="Task completed successfully",
                supporting_episodes=[episode.episode_id],
                confidence=0.6,
                tags=episode.topics[:3]
            )
            
            self.add_pattern(pattern)
        
        elif not episode.success and len(episode.errors_encountered) > 0:
            # Create failure pattern
            pattern_id = self._generate_pattern_id(episode.task_description, "failure")
            
            # Check if pattern already exists
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                if episode.episode_id not in pattern.supporting_episodes:
                    pattern.supporting_episodes.append(episode.episode_id)
                pattern.confidence = min(0.95, pattern.confidence + 0.05)
                pattern.last_updated = datetime.now(UTC)
                return
            
            # Extract common error
            error_types = [e.get("error", {}).get("type", "unknown") for e in episode.errors_encountered]
            most_common_error = max(set(error_types), key=error_types.count) if error_types else "unknown"
            
            pattern = Pattern(
                pattern_id=pattern_id,
                pattern_type="failure",
                context=f"Task: {episode.task_description[:100]}",
                action=f"Avoid: {most_common_error}",
                outcome="Task failed",
                supporting_episodes=[episode.episode_id],
                confidence=0.5,
                tags=episode.topics[:3]
            )
            
            self.add_pattern(pattern)
    
    def update_entity_knowledge(self, entity: str, knowledge: Dict[str, Any]):
        """
        Update knowledge about an entity.
        
        ✅ FIXED: Properly merges counters and lists instead of overwriting.
        """
        if entity not in self.entity_knowledge:
            self.entity_knowledge[entity] = {}
        
        existing = self.entity_knowledge[entity]
        
        # Merge knowledge intelligently
        for key, value in knowledge.items():
            if key == "access_count" or key == "modification_count" or key == "query_count":
                # Increment counters
                existing[key] = existing.get(key, 0) + value
            elif key == "queries" and isinstance(value, list):
                # Append to query list (keep last 10)
                existing_queries = existing.get("queries", [])
                existing_queries.extend(value)
                existing[key] = existing_queries[-10:]
            else:
                # Overwrite other fields
                existing[key] = value
        
        existing["last_updated"] = datetime.now(UTC).isoformat()
    
    def update_topic_knowledge(self, topic: str, knowledge: Dict[str, Any]):
        """
        Update knowledge about a topic.
        
        ✅ FIXED: Properly merges counters and lists instead of overwriting.
        """
        if topic not in self.topic_knowledge:
            self.topic_knowledge[topic] = {}
        
        existing = self.topic_knowledge[topic]
        
        # Merge knowledge intelligently
        for key, value in knowledge.items():
            if key == "query_count":
                # Increment counter
                existing[key] = existing.get(key, 0) + value
            elif key == "queries" and isinstance(value, list):
                # Append to query list (keep last 10)
                existing_queries = existing.get("queries", [])
                existing_queries.extend(value)
                existing[key] = existing_queries[-10:]
            else:
                # Overwrite other fields
                existing[key] = value
        
        existing["last_updated"] = datetime.now(UTC).isoformat()
    
    def _init_storage(self):
        """Verify database connection - tables are created by frontend."""
        if not self.storage_path or not self.storage_path.exists():
            logger.warning(f"Database not found at {self.storage_path}")
            return
        
        # Just verify we can connect - don't create any tables
        conn = sqlite3.connect(str(self.storage_path))
        conn.close()
    
    def _load_into_memory(self):
        """Load patterns and knowledge into in-memory structures."""
        conn = sqlite3.connect(str(self.storage_path))
        conn.row_factory = sqlite3.Row
        
        # Load patterns
        cursor = conn.execute("SELECT * FROM memory_patterns")
        for row in cursor:
            try:
                pattern_data = json.loads(row["data"])
                pattern = Pattern.from_dict(pattern_data)
                self.add_pattern(pattern)
            except Exception:
                pass
        
        # ✅ NEW: Load entity knowledge from memory_entities table
        try:
            cursor = conn.execute("SELECT entity, knowledge FROM memory_entities")
            entity_count = 0
            for row in cursor:
                try:
                    entity, knowledge_json = row
                    knowledge = json.loads(knowledge_json)
                    self.entity_knowledge[entity] = knowledge
                    entity_count += 1
                except Exception as e:
                    pass  # Skip malformed entries
            
            if entity_count > 0:
                print(f"🧠 [SemanticMemory] Loaded {entity_count} entities from database")
        except sqlite3.OperationalError:
            # Table doesn't exist yet - that's fine
            pass
        
        # ✅ NEW: Load topic knowledge from memory_topics table
        try:
            cursor = conn.execute("SELECT topic, knowledge FROM memory_topics")
            topic_count = 0
            for row in cursor:
                try:
                    topic, knowledge_json = row
                    knowledge = json.loads(knowledge_json)
                    self.topic_knowledge[topic] = knowledge
                    topic_count += 1
                except Exception as e:
                    pass  # Skip malformed entries
            
            if topic_count > 0:
                print(f"🧠 [SemanticMemory] Loaded {topic_count} topics from database")
        except sqlite3.OperationalError:
            # Table doesn't exist yet - that's fine
            pass
        
        conn.close()
    
    def _generate_pattern_id(self, context: str, pattern_type: str) -> str:
        """Generate unique pattern ID."""
        # Use a stable hash based on context and type (not timestamp)
        # This allows similar patterns to be recognized and merged
        content = f"{context[:100]}_{pattern_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
