"""Enhanced context configuration for the ReAct agent with plan adherence."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Dict, Optional

from react_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Enhanced context for the ReAct agent with planning and assessment capabilities."""

    # Core model configuration
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai:gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider:model-name (colon format for better compatibility)."
        },
    )
    
    # Planning configuration
    planning_model: Optional[str] = field(
        default=None,
        metadata={
            "description": "Optional separate model for planning tasks. "
            "If not specified, uses the main model. Can be a cheaper/faster model."
        },
    )
    
    assessment_model: Optional[str] = field(
        default=None,
        metadata={
            "description": "Optional separate model for assessment tasks. "
            "If not specified, uses the main model."
        },
    )
    
    # Execution parameters
    max_retries_per_step: int = field(
        default=3,
        metadata={
            "description": "Maximum number of retries allowed for each plan step "
            "before triggering replanning."
        },
    )
    
    max_plan_revisions: int = field(
        default=2,
        metadata={
            "description": "Maximum number of times the plan can be revised "
            "during execution before failing."
        },
    )
    
    max_total_steps: int = field(
        default=10,
        metadata={
            "description": "Maximum total number of steps allowed in any plan "
            "to prevent overly complex execution paths."
        },
    )
    
    # Tool configuration
    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
    
    tool_timeout_seconds: int = field(
        default=30,
        metadata={
            "description": "Timeout in seconds for individual tool executions."
        },
    )
    
    parallel_tool_execution: bool = field(
        default=True,
        metadata={
            "description": "Whether to execute multiple tool calls in parallel when possible."
        },
    )
    
    # Assessment configuration
    strict_success_criteria: bool = field(
        default=True,
        metadata={
            "description": "Whether to strictly enforce success criteria in assessments. "
            "If False, allows more lenient interpretation of success."
        },
    )
    
    assessment_confidence_threshold: float = field(
        default=0.7,
        metadata={
            "description": "Minimum confidence level required for assessment decisions. "
            "Assessments below this threshold trigger re-assessment or human review."
        },
    )
    
    # Planning strategies
    planning_strategy: str = field(
        default="minimal",
        metadata={
            "description": "Planning strategy: 'minimal' for simple plans, "
            "'comprehensive' for detailed plans, 'adaptive' for context-aware planning."
        },
    )
    
    prefer_specific_tools: bool = field(
        default=True,
        metadata={
            "description": "Whether the planner should prefer specifying exact tools "
            "for each step when possible."
        },
    )
    
    # Error handling
    continue_on_tool_error: bool = field(
        default=True,
        metadata={
            "description": "Whether to continue execution when a tool returns an error, "
            "allowing the assessor to decide on retry/replan."
        },
    )
    
    capture_detailed_errors: bool = field(
        default=True,
        metadata={
            "description": "Whether to capture and store detailed error information "
            "for debugging and improvement."
        },
    )
    
    
    enable_step_dependencies: bool = field(
        default=True,
        metadata={
            "description": "Whether to respect step dependencies in the execution plan. "
            "If False, executes steps strictly in order."
        },
    )
    
    # Logging and monitoring
    verbose_logging: bool = field(
        default=False,
        metadata={
            "description": "Enable detailed logging of planning, execution, and assessment steps."
        },
    )
    
    
    # Runtime metadata
    runtime_metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional runtime metadata that can be used by nodes."
        },
    )
    
    # Feature flags
    enable_replanning: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable automatic replanning when steps fail."
        },
    )
    
    
    use_chain_of_thought: bool = field(
        default=True,
        metadata={
            "description": "Whether to request chain-of-thought reasoning in planning and assessment."
        },
    )
    
    # Memory system configuration
    enable_memory: bool = field(
        default=True,
        metadata={
            "description": "Enable the 3-tier memory system (working, episodic, semantic)."
        },
    )
    
    memory_db_path: Optional[str] = field(
        default=None,
        metadata={
            "description": "Path to SQLite database for persistent memory storage. "
            "If None, uses in-memory storage only."
        },
    )
    
    memory_inject_context: bool = field(
        default=True,
        metadata={
            "description": "Whether to inject memory context (patterns, episodes) into LLM prompts."
        },
    )
    
    memory_max_recent_episodes: int = field(
        default=20,
        metadata={
            "description": "Maximum number of recent episodes to keep in episodic memory."
        },
    )
    
    memory_pattern_confidence_threshold: float = field(
        default=0.6,
        metadata={
            "description": "Minimum confidence threshold for using learned patterns."
        },
    )

    # Prompt caching configuration
    enable_prompt_cache: bool = field(
        default=True,
        metadata={
            "description": "Enable prompt caching for supported models. "
            "- OpenAI (gpt-4o, gpt-4-turbo, gpt-3.5-turbo): Automatic prefix matching, no code changes needed "
            "- Anthropic (Claude): Explicit cache_control markers, ~90% cost reduction for repeated prompts "
            "This flag only affects Anthropic models (OpenAI caching is always automatic)."
        },
    )

    prompt_cache_ttl_seconds: int = field(
        default=300,  # 5 minutes (Anthropic's ephemeral cache TTL)
        metadata={
            "description": "Time-to-live for cached prompts in seconds. "
            "Only applies to Anthropic models (5 min ephemeral cache). "
            "OpenAI models use automatic 5-10 min TTL (not configurable)."
        },
    )

    def __post_init__(self) -> None:
        """Post-initialization setup and validation."""
        # Fetch env vars for attributes that were not passed as args
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_var = os.environ.get(f.name.upper())
                if env_var is not None:
                    # Type conversion based on field type
                    field_type = f.type
                    if field_type == bool:
                        setattr(self, f.name, env_var.lower() in ('true', '1', 'yes'))
                    elif field_type == int:
                        setattr(self, f.name, int(env_var))
                    elif field_type == float:
                        setattr(self, f.name, float(env_var))
                    else:
                        setattr(self, f.name, env_var)
        
        # Set default models if not specified
        if not self.planning_model:
            self.planning_model = self.model
        if not self.assessment_model:
            self.assessment_model = self.model
        
        # Validate configuration
        if self.assessment_confidence_threshold < 0 or self.assessment_confidence_threshold > 1:
            raise ValueError("assessment_confidence_threshold must be between 0 and 1")
        
        if self.max_retries_per_step < 0:
            raise ValueError("max_retries_per_step must be non-negative")
        
        if self.planning_strategy not in ["minimal", "comprehensive", "adaptive"]:
            raise ValueError("planning_strategy must be one of: minimal, comprehensive, adaptive")
        
        # Validate memory configuration
        if self.memory_pattern_confidence_threshold < 0 or self.memory_pattern_confidence_threshold > 1:
            raise ValueError("memory_pattern_confidence_threshold must be between 0 and 1")