"""Define the state structures for the enhanced ReAct agent with plan adherence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List, Optional, Literal, Dict, Any
from enum import Enum

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ConfigDict  # Fixed: Use pydantic directly


# Type-constrained tool names based on available tools
ToolName = Literal[
    "search",
    "get_project_info", 
    "create_asset",
    "write_file",
    "edit_project_config",
    "get_script_snippets",
    "compile_and_test",
    "scene_management"
]


class StepStatus(str, Enum):
    """Status of a plan step execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanStep(BaseModel):
    """Individual step in the execution plan."""
    model_config = ConfigDict(extra="forbid")  # Prevent schema drift
    
    description: str = Field(description="Clear description of what this step accomplishes")
    tool_name: Optional[ToolName] = Field(default=None, description="Specific tool to use for this step")
    success_criteria: str = Field(description="Measurable criteria to determine if step succeeded")
    dependencies: List[int] = Field(default_factory=list, description="Indices of steps this depends on")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Current status of the step")
    attempts: int = Field(default=0, description="Number of attempts made for this step")
    error_messages: List[str] = Field(default_factory=list, description="Error messages from failed attempts")


class ExecutionPlan(BaseModel):
    """Structured execution plan with verifiable steps."""
    model_config = ConfigDict(extra="forbid")
    
    goal: str = Field(description="Overall goal to achieve")
    steps: List[PlanStep] = Field(description="Ordered list of steps to execute")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional planning metadata")


class AssessmentOutcome(BaseModel):
    """Assessment result for a step execution."""
    model_config = ConfigDict(extra="forbid")
    
    outcome: Literal["success", "retry", "blocked"] = Field(
        description="Whether the step succeeded, should be retried, or is blocked"
    )
    reason: str = Field(description="Explanation of the assessment")
    fix: Optional[str] = Field(default=None, description="Suggested fix if retry is needed")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in assessment")


@dataclass
class InputState:
    """Input state for the agent."""    
    
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """Messages tracking the primary execution state of the agent."""


@dataclass
class State(InputState):
    """Complete state of the enhanced ReAct agent with plan tracking."""
    
    # Plan management
    plan: Optional[ExecutionPlan] = field(default=None)
    """Current execution plan with structured steps."""
    
    step_index: int = field(default=0)
    """Current step being executed (0-based index)."""
    
    # Execution tracking
    current_assessment: Optional[AssessmentOutcome] = field(default=None)
    """Most recent assessment of the current step."""
    
    retry_count: int = field(default=0)
    """Number of retries for the current step."""
    
    max_retries_per_step: int = field(default=3)
    """Maximum retries allowed per step before replanning."""
    
    # Tool execution context
    last_tool_result: Optional[Dict[str, Any]] = field(default=None)
    """Result from the most recent tool execution."""
    
    tool_errors: List[Dict[str, Any]] = field(default_factory=list)
    """History of tool execution errors for debugging."""
    
    # Progress tracking
    completed_steps: List[int] = field(default_factory=list)
    """Indices of successfully completed steps."""
    
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    """Detailed history of all execution attempts."""
    
    # Control flags
    should_replan: bool = field(default=False)
    """Flag indicating whether replanning is needed."""
    
    plan_revision_count: int = field(default=0)
    """Number of times the plan has been revised."""
    
    is_last_step: IsLastStep = field(default=False)
    """Managed variable: True when approaching recursion limit."""
    
    # Performance metrics
    total_tool_calls: int = field(default=0)
    """Total number of tool calls made."""
    
    total_assessments: int = field(default=0)
    """Total number of step assessments performed."""
    
    # Additional context
    user_constraints: Dict[str, Any] = field(default_factory=dict)
    """Any constraints or preferences specified by the user."""
    
    runtime_context: Dict[str, Any] = field(default_factory=dict)
    """Runtime context information for tools and assessment."""
    
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    """Runtime metadata that can be used by nodes for routing and classification."""
    
    # --- Error recovery flags/state ---
    needs_error_recovery: bool = field(default=False)
    """Flag indicating that error recovery is needed."""
    
    error_recovery_active: bool = field(default=False)
    """Flag indicating that error recovery is currently active."""
    
    error_context: Optional[Dict[str, Any]] = field(default=None)
    """Context information about the error that needs recovery."""


# Structured output schemas for LLM responses
class StructuredPlanStep(BaseModel):
    """Structured step for planning output with type-constrained tools."""
    description: str = Field(description="Clear description of what this step accomplishes")
    tool_name: ToolName = Field(description="Specific tool to use for this step")  # Type-constrained!
    success_criteria: str = Field(description="Measurable criteria to determine if step succeeded")
    dependencies: List[int] = Field(default_factory=list, description="Indices of steps this depends on")


class StructuredExecutionPlan(BaseModel):
    """Structured execution plan for native output with type-constrained tools."""
    goal: str = Field(description="Overall goal to achieve")
    steps: List[StructuredPlanStep] = Field(description="Ordered list of steps to execute")


class StructuredAssessment(BaseModel):
    """Structured assessment for native output."""
    outcome: Literal["success", "retry", "blocked"] = Field(
        description="Whether the step succeeded, should be retried, or is blocked"
    )
    reason: str = Field(description="Specific explanation of the assessment")
    fix: str = Field(default="", description="Suggested fix if retry is needed")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in assessment")