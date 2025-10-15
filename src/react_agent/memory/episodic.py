"""
TIER 2: Episodic Memory (Medium-term, task episodes).

Analogous to human episodic memory (remembering specific events).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from collections import defaultdict
import hashlib
import sqlite3
import json


class EpisodeStatus(str, Enum):
    """Status of a task episode."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class Episode:
    """
    An episode represents a complete task execution from start to finish.
    """
    
    episode_id: str
    task_description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: EpisodeStatus = EpisodeStatus.IN_PROGRESS
    
    # Episode content
    messages: List[Dict[str, Any]] = field(default_factory=list)
    plans: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    assessments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Episode metadata
    entities_involved: List[str] = field(default_factory=list)  # Files, classes, functions
    topics: List[str] = field(default_factory=list)  # Themes, concepts
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)
    
    # Outcome
    success: bool = False
    outcome_summary: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    # Metrics
    total_steps: int = 0
    total_retries: int = 0
    execution_time_seconds: float = 0.0
    
    # Insights
    what_worked: str = ""
    what_failed: str = ""
    insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for storage."""
        return {
            "episode_id": self.episode_id,
            "task_description": self.task_description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "messages": self.messages,
            "plans": self.plans,
            "tool_calls": self.tool_calls,
            "assessments": self.assessments,
            "entities_involved": self.entities_involved,
            "topics": self.topics,
            "errors_encountered": self.errors_encountered,
            "success": self.success,
            "outcome_summary": self.outcome_summary,
            "lessons_learned": self.lessons_learned,
            "total_steps": self.total_steps,
            "total_retries": self.total_retries,
            "execution_time_seconds": self.execution_time_seconds,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "insights": self.insights,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Episode:
        """Create episode from dictionary."""
        return cls(
            episode_id=data["episode_id"],
            task_description=data["task_description"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=EpisodeStatus(data["status"]),
            messages=data.get("messages", []),
            plans=data.get("plans", []),
            tool_calls=data.get("tool_calls", []),
            assessments=data.get("assessments", []),
            entities_involved=data.get("entities_involved", []),
            topics=data.get("topics", []),
            errors_encountered=data.get("errors_encountered", []),
            success=data.get("success", False),
            outcome_summary=data.get("outcome_summary"),
            lessons_learned=data.get("lessons_learned", []),
            total_steps=data.get("total_steps", 0),
            total_retries=data.get("total_retries", 0),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            what_worked=data.get("what_worked", ""),
            what_failed=data.get("what_failed", ""),
            insights=data.get("insights", []),
        )


@dataclass
class EpisodicMemory:
    """
    Episodic memory stores recent task episodes for context and learning.
    
    Characteristics:
    - Medium-term storage (days to weeks)
    - Stores complete task episodes
    - Enables learning from recent experiences
    - Supports "remember when I did X" queries
    """
    
    current_episode: Optional[Episode] = None
    recent_episodes: List[Episode] = field(default_factory=list)
    max_recent_episodes: int = 20
    
    def start_episode(self, task_description: str) -> str:
        """Start a new episode."""
        episode_id = self._generate_episode_id(task_description)
        self.current_episode = Episode(
            episode_id=episode_id,
            task_description=task_description,
            start_time=datetime.now(UTC),
        )
        return episode_id
    
    def end_episode(self, success: bool, outcome_summary: Optional[str] = None):
        """End the current episode."""
        if not self.current_episode:
            return
        
        self.current_episode.end_time = datetime.now(UTC)
        self.current_episode.status = EpisodeStatus.COMPLETED if success else EpisodeStatus.FAILED
        self.current_episode.success = success
        self.current_episode.outcome_summary = outcome_summary
        
        # Move to recent episodes
        self.recent_episodes.append(self.current_episode)
        
        # Prune old episodes
        if len(self.recent_episodes) > self.max_recent_episodes:
            self.recent_episodes = self.recent_episodes[-self.max_recent_episodes:]
        
        self.current_episode = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to current episode."""
        if self.current_episode:
            self.current_episode.messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": metadata or {}
            })
    
    def add_plan(self, plan: Dict[str, Any]):
        """Add plan to current episode."""
        if self.current_episode:
            self.current_episode.plans.append({
                "plan": plan,
                "timestamp": datetime.now(UTC).isoformat()
            })
    
    def add_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any):
        """Add tool call to current episode."""
        if self.current_episode:
            self.current_episode.tool_calls.append({
                "tool_name": tool_name,
                "args": args,
                "result": result,
                "timestamp": datetime.now(UTC).isoformat()
            })
    
    def add_assessment(self, assessment: Dict[str, Any]):
        """Add assessment to current episode."""
        if self.current_episode:
            self.current_episode.assessments.append({
                "assessment": assessment,
                "timestamp": datetime.now(UTC).isoformat()
            })
    
    def add_error(self, error: Dict[str, Any]):
        """Add error to current episode."""
        if self.current_episode:
            self.current_episode.errors_encountered.append({
                "error": error,
                "timestamp": datetime.now(UTC).isoformat()
            })
    
    def update_entities(self, entities: List[str]):
        """Update entities involved in current episode."""
        if self.current_episode:
            # Add new entities, avoid duplicates
            for entity in entities:
                if entity not in self.current_episode.entities_involved:
                    self.current_episode.entities_involved.append(entity)
    
    def update_topics(self, topics: List[str]):
        """Update topics in current episode."""
        if self.current_episode:
            for topic in topics:
                if topic not in self.current_episode.topics:
                    self.current_episode.topics.append(topic)
    
    def find_similar_episodes(self, entities: List[str], topics: List[str], limit: int = 5) -> List[Episode]:
        """Find episodes similar to current context."""
        scored_episodes = []
        
        for episode in self.recent_episodes:
            score = 0
            
            # Score based on entity overlap
            entity_overlap = len(set(entities) & set(episode.entities_involved))
            score += entity_overlap * 2
            
            # Score based on topic overlap
            topic_overlap = len(set(topics) & set(episode.topics))
            score += topic_overlap * 1.5
            
            # Boost successful episodes
            if episode.success:
                score *= 1.2
            
            if score > 0:
                scored_episodes.append((score, episode))
        
        # Sort by score and return top N
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored_episodes[:limit]]
    
    def get_success_patterns(self, goal_type: str) -> Dict[str, Any]:
        """
        Analyze successful episodes to extract patterns.
        Returns what typically works for this type of goal.
        """
        # Filter successful episodes matching goal type
        matching_episodes = [
            ep for ep in self.recent_episodes
            if ep.success and (
                goal_type.lower() in ep.task_description.lower() or
                any(goal_type.lower() in topic.lower() for topic in ep.topics)
            )
        ]
        
        if not matching_episodes:
            return {"pattern": "no_data", "confidence": 0.0}
        
        # Analyze patterns
        tool_frequency = defaultdict(int)
        common_steps = defaultdict(int)
        avg_steps = sum(e.total_steps for e in matching_episodes) / len(matching_episodes)
        avg_retries = sum(e.total_retries for e in matching_episodes) / len(matching_episodes)
        
        for episode in matching_episodes:
            # Count tool usage
            for tool_call in episode.tool_calls:
                tool_name = tool_call.get("tool_name", "unknown")
                tool_frequency[tool_name] += 1
            
            # Count step patterns
            if episode.plans:
                for plan in episode.plans:
                    plan_data = plan.get("plan", {})
                    for step in plan_data.get("steps", []):
                        step_tool = step.get("tool_name", "unknown")
                        common_steps[step_tool] += 1
        
        return {
            "pattern": "success",
            "confidence": min(1.0, len(matching_episodes) / 10.0),  # Confidence based on sample size
            "preferred_tools": sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True)[:3],
            "common_step_sequence": sorted(common_steps.items(), key=lambda x: x[1], reverse=True)[:5],
            "typical_step_count": round(avg_steps),
            "typical_retry_count": round(avg_retries),
            "sample_size": len(matching_episodes)
        }
    
    def _generate_episode_id(self, task_description: str) -> str:
        """Generate unique episode ID."""
        timestamp = datetime.now(UTC).isoformat()
        content = f"{task_description}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
