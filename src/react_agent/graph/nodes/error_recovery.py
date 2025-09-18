"""Fixed error recovery system with proper step insertion and routing."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from react_agent.context import Context
from react_agent.state import State, PlanStep, ExecutionPlan, StepStatus
from react_agent.utils import get_model, get_message_text
from react_agent.tools import TOOLS


class ErrorCategory(str, Enum):
    """Categories of errors that can occur."""
    DEPENDENCY_MISSING = "dependency_missing"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    PERMISSION_ERROR = "permission_error"
    NETWORK_ERROR = "network_error"
    INVALID_PARAMETER = "invalid_parameter"
    TOOL_MALFUNCTION = "tool_malfunction"
    PROJECT_STATE_ERROR = "project_state_error"
    BUILD_ERROR = "build_error"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ErrorDiagnosis:
    """Detailed error diagnosis."""
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    root_cause: str
    suggested_fixes: List[str]
    required_tools: List[str]
    confidence: float
    can_auto_fix: bool
    workaround_available: bool


class StructuredErrorRecovery(BaseModel):
    """Structured error recovery for LLM output."""
    diagnosis: str = Field(description="Clear diagnosis of what went wrong")
    recovery_steps: List[Dict[str, Any]] = Field(description="Steps to fix the error")
    fallback_approach: str = Field(default="", description="Alternative approach if fix fails")
    estimated_complexity: str = Field(description="Simple, moderate, or complex recovery")


async def diagnose_error(
    error_info: Dict[str, Any],
    failed_step: PlanStep,
    state: State,
    runtime: Runtime[Context]
) -> ErrorDiagnosis:
    """Intelligently diagnose what went wrong with a tool call."""
    context = runtime.context
    model = get_model(context.model)
    
    # Extract error details
    error_message = error_info.get("error", "Unknown error")
    tool_name = failed_step.tool_name
    step_description = failed_step.description
    
    # Build diagnostic context
    diagnostic_prompt = f"""Analyze this Unity/game development tool error and provide a comprehensive diagnosis:

FAILED OPERATION:
- Tool: {tool_name}
- Step: {step_description}
- Error: {error_message}
- Retry Count: {state.retry_count}

PROJECT CONTEXT:
- Goal: {state.plan.goal if state.plan else 'Unknown'}
- Step {state.step_index + 1} of {len(state.plan.steps) if state.plan else 'Unknown'}

RECENT CONTEXT:
{_extract_recent_context(state)}

Provide a detailed diagnosis considering:
1. What specifically failed and why
2. Whether this is a configuration, dependency, or implementation issue  
3. If this is related to Unity/Unreal project state
4. Whether the error indicates a missing prerequisite
5. If the tool parameters were incorrect
6. Whether there's a systemic project issue

Focus on actionable, specific diagnosis that can guide automatic error resolution."""

    try:
        diagnostic_response = await model.ainvoke([
            {"role": "system", "content": "You are a Unity/Unreal Engine development expert diagnosing tool failures. Provide specific, actionable diagnosis focused on what can be automatically fixed."},
            {"role": "user", "content": diagnostic_prompt}
        ])
        
        diagnosis_text = get_message_text(diagnostic_response)
        
        # Categorize the error based on patterns
        category = _categorize_error(error_message, tool_name, diagnosis_text)
        severity = _assess_severity(error_message, tool_name, state.retry_count)
        
        # Determine if we can auto-fix
        can_auto_fix = _can_auto_fix(category, tool_name, error_message)
        
        # Generate suggested fixes
        suggested_fixes = _generate_fix_suggestions(category, tool_name, error_message, diagnosis_text)
        
        return ErrorDiagnosis(
            category=category,
            severity=severity,
            description=diagnosis_text,
            root_cause=_extract_root_cause(diagnosis_text, error_message),
            suggested_fixes=suggested_fixes,
            required_tools=_determine_required_tools(category, tool_name),
            confidence=0.8,
            can_auto_fix=can_auto_fix,
            workaround_available=_has_workaround(category, tool_name)
        )
        
    except Exception as e:
        # Fallback diagnosis
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            description=f"Error occurred: {error_message}",
            root_cause="Unable to determine root cause",
            suggested_fixes=["Retry with different parameters", "Check project configuration"],
            required_tools=[tool_name],
            confidence=0.3,
            can_auto_fix=False,
            workaround_available=True
        )


async def create_error_recovery_plan(
    diagnosis: ErrorDiagnosis,
    failed_step: PlanStep,
    state: State,
    runtime: Runtime[Context]
) -> List[PlanStep]:
    """Create recovery steps to fix the error."""
    context = runtime.context
    model = get_model(context.model)
    
    recovery_request = f"""Create specific error recovery steps for this Unity/game development failure:

ERROR DIAGNOSIS:
- Category: {diagnosis.category}
- Severity: {diagnosis.severity}
- Root Cause: {diagnosis.root_cause}
- Can Auto-Fix: {diagnosis.can_auto_fix}

FAILED STEP:
- Description: {failed_step.description}
- Tool: {failed_step.tool_name}
- Success Criteria: {failed_step.success_criteria}

SUGGESTED FIXES:
{chr(10).join(f"- {fix}" for fix in diagnosis.suggested_fixes)}

Create 1-3 focused recovery steps that systematically resolve the issue.

VALID TOOLS: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

Example recovery patterns:
- Configuration Error: get_project_info → edit_project_config
- Missing Dependency: search → create_asset → compile_and_test  
- Build Error: compile_and_test → search → write_file
- Resource Missing: get_project_info → create_asset

Provide recovery steps that address the root cause."""

    try:
        structured_model = model.with_structured_output(StructuredErrorRecovery)
        recovery_response = await structured_model.ainvoke([
            {"role": "system", "content": "You are creating Unity/Unreal error recovery plans. Focus on systematic problem-solving that addresses root causes."},
            {"role": "user", "content": recovery_request}
        ])
        
        # Convert to internal format
        recovery_steps = []
        for i, step_data in enumerate(recovery_response.recovery_steps):
            step = PlanStep(
                description=step_data.get("description", f"Recovery step {i+1}"),
                tool_name=step_data.get("tool_name"),
                success_criteria=step_data.get("success_criteria", "Step completed successfully"),
                dependencies=step_data.get("dependencies", [])
            )
            recovery_steps.append(step)
        
        return recovery_steps
        
    except Exception as e:
        # Fallback recovery steps
        return _create_fallback_recovery_steps(diagnosis, failed_step)


async def execute_error_recovery(
    state: State,
    runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Main error recovery execution node."""
    
    # Extract error information - use error_context set by assess node
    if not state.error_context:
        return {"messages": [AIMessage(content="No error context found for recovery.")]}
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {"messages": [AIMessage(content="Cannot recover: invalid plan state.")]}
    
    current_step = state.plan.steps[state.step_index]
    
    # Step 1: Diagnose the error using the stored error context
    error_info = {
        "error": state.error_context.get("assessment").reason if state.error_context.get("assessment") else "Unknown error",
        "tool_result": state.error_context.get("tool_result", {}),
        "tool_name": state.error_context.get("tool_name"),
        "step_description": state.error_context.get("step_description")
    }
    
    diagnosis = await diagnose_error(error_info, current_step, state, runtime)
    
    # Step 2: Create recovery steps
    recovery_steps = await create_error_recovery_plan(diagnosis, current_step, state, runtime)
    
    # Step 3: Insert recovery steps into execution plan BEFORE the failed step
    updated_plan = _insert_recovery_steps_before_failed_step(state.plan, state.step_index, recovery_steps)
    
    # Step 4: Create user communication
    recovery_message = _create_recovery_narration(diagnosis, recovery_steps, state.retry_count)
    
    return {
        "plan": updated_plan,
        "step_index": state.step_index,  # Start executing the first recovery step (inserted at current position)
        "retry_count": 0,  # Reset retry count
        "current_assessment": None,
        "error_recovery_active": True,
        "needs_error_recovery": False,  # Clear the flag
        "error_context": None,  # Clear error context
        "messages": [AIMessage(content=recovery_message)],
        "runtime_metadata": {
            **state.runtime_metadata,
            "error_diagnosis": diagnosis.__dict__,
            "original_failed_step": state.step_index + len(recovery_steps),  # Original step is now after recovery steps
            "recovery_steps_count": len(recovery_steps)
        }
    }


def _insert_recovery_steps_before_failed_step(original_plan: ExecutionPlan, failed_step_index: int, recovery_steps: List[PlanStep]) -> ExecutionPlan:
    """Insert recovery steps BEFORE the failed step in the plan."""
    if not original_plan or not recovery_steps:
        return original_plan
    
    new_steps = []
    
    # Add all steps up to (but not including) the failed step
    for i in range(failed_step_index):
        new_steps.append(original_plan.steps[i])
    
    # Insert recovery steps at the failed step position
    for recovery_step in recovery_steps:
        new_steps.append(recovery_step)
    
    # Add the original failed step and all remaining steps after
    for i in range(failed_step_index, len(original_plan.steps)):
        new_steps.append(original_plan.steps[i])
    
    return ExecutionPlan(
        goal=original_plan.goal,
        steps=new_steps,
        metadata={
            **original_plan.metadata,
            "recovery_inserted": True,
            "original_failed_step": failed_step_index + len(recovery_steps),  # Adjust index after insertion
            "recovery_steps_count": len(recovery_steps)
        }
    )


# Helper functions

def _extract_recent_context(state: State) -> str:
    """Extract recent context for error diagnosis."""
    recent_messages = state.messages[-3:] if len(state.messages) >= 3 else state.messages
    context_parts = []
    
    for msg in recent_messages:
        content = get_message_text(msg)[:100]
        if content:
            context_parts.append(content)
    
    return "; ".join(context_parts) if context_parts else "No recent context"


def _categorize_error(error_message: str, tool_name: str, diagnosis: str) -> ErrorCategory:
    """Categorize error based on patterns."""
    error_lower = error_message.lower()
    
    if any(term in error_lower for term in ["not found", "missing", "does not exist"]):
        return ErrorCategory.RESOURCE_NOT_FOUND
    elif any(term in error_lower for term in ["permission", "access", "forbidden"]):
        return ErrorCategory.PERMISSION_ERROR
    elif any(term in error_lower for term in ["config", "setting", "invalid parameter"]):
        return ErrorCategory.CONFIGURATION_ERROR
    elif any(term in error_lower for term in ["network", "connection", "timeout"]):
        return ErrorCategory.NETWORK_ERROR
    elif any(term in error_lower for term in ["compile", "build", "syntax"]):
        return ErrorCategory.BUILD_ERROR
    elif any(term in error_lower for term in ["dependency", "reference", "import"]):
        return ErrorCategory.DEPENDENCY_MISSING
    elif tool_name in ["get_project_info", "scene_management"]:
        return ErrorCategory.PROJECT_STATE_ERROR
    else:
        return ErrorCategory.UNKNOWN


def _assess_severity(error_message: str, tool_name: str, retry_count: int) -> ErrorSeverity:
    """Assess error severity."""
    if retry_count >= 2:
        return ErrorSeverity.HIGH
    
    error_lower = error_message.lower()
    if any(term in error_lower for term in ["critical", "fatal", "crash"]):
        return ErrorSeverity.CRITICAL
    elif any(term in error_lower for term in ["warning", "minor"]):
        return ErrorSeverity.LOW
    else:
        return ErrorSeverity.MEDIUM


def _can_auto_fix(category: ErrorCategory, tool_name: str, error_message: str) -> bool:
    """Determine if error can be automatically fixed."""
    auto_fixable = {
        ErrorCategory.CONFIGURATION_ERROR,
        ErrorCategory.DEPENDENCY_MISSING,
        ErrorCategory.INVALID_PARAMETER,
        ErrorCategory.PROJECT_STATE_ERROR
    }
    return category in auto_fixable


def _generate_fix_suggestions(category: ErrorCategory, tool_name: str, error_message: str, diagnosis: str) -> List[str]:
    """Generate specific fix suggestions."""
    fixes_map = {
        ErrorCategory.RESOURCE_NOT_FOUND: [
            "Create the missing resource or asset",
            "Check project structure and file paths",
            "Verify asset references and naming"
        ],
        ErrorCategory.CONFIGURATION_ERROR: [
            "Update project configuration settings",
            "Check build settings and player settings",
            "Verify tool-specific configuration"
        ],
        ErrorCategory.DEPENDENCY_MISSING: [
            "Install required Unity packages",
            "Add missing script references",
            "Create prerequisite assets or scripts"
        ],
        ErrorCategory.BUILD_ERROR: [
            "Fix compilation errors in scripts",
            "Check for syntax and reference errors",
            "Update script dependencies"
        ],
        ErrorCategory.PROJECT_STATE_ERROR: [
            "Check Unity project is open and accessible",
            "Verify scene state and active objects",
            "Refresh project structure"
        ]
    }
    
    return fixes_map.get(category, ["Retry with adjusted parameters", "Check tool configuration"])


def _determine_required_tools(category: ErrorCategory, original_tool: str) -> List[str]:
    """Determine what tools are needed to fix this error."""
    tool_map = {
        ErrorCategory.CONFIGURATION_ERROR: ["get_project_info", "edit_project_config"],
        ErrorCategory.DEPENDENCY_MISSING: ["search", "create_asset", "compile_and_test"],
        ErrorCategory.RESOURCE_NOT_FOUND: ["get_project_info", "create_asset"],
        ErrorCategory.BUILD_ERROR: ["compile_and_test", "search", "write_file"],
        ErrorCategory.PROJECT_STATE_ERROR: ["get_project_info", "scene_management"]
    }
    
    return tool_map.get(category, ["get_project_info", original_tool])


def _has_workaround(category: ErrorCategory, tool_name: str) -> bool:
    """Check if a workaround exists for this error type."""
    return category != ErrorCategory.CRITICAL


def _extract_root_cause(diagnosis: str, error_message: str) -> str:
    """Extract root cause from diagnosis."""
    sentences = diagnosis.split('.')
    for sentence in sentences:
        if any(word in sentence.lower() for word in ["because", "due to", "caused by", "root"]):
            return sentence.strip()
    
    return error_message[:100]


def _create_fallback_recovery_steps(diagnosis: ErrorDiagnosis, failed_step: PlanStep) -> List[PlanStep]:
    """Create fallback recovery steps."""
    recovery_steps = []
    
    # Always start with project inspection
    recovery_steps.append(PlanStep(
        description="Analyze project state to understand the error context",
        tool_name="get_project_info",
        success_criteria="Retrieved current project information and identified potential issues"
    ))
    
    # Category-specific recovery
    if diagnosis.category == ErrorCategory.BUILD_ERROR:
        recovery_steps.append(PlanStep(
            description="Test compilation to identify specific build issues",
            tool_name="compile_and_test",
            success_criteria="Identified compilation errors and warnings",
            dependencies=[0]
        ))
    elif diagnosis.category == ErrorCategory.CONFIGURATION_ERROR:
        recovery_steps.append(PlanStep(
            description="Update project configuration to resolve the issue",
            tool_name="edit_project_config", 
            success_criteria="Applied necessary configuration changes",
            dependencies=[0]
        ))
    else:
        recovery_steps.append(PlanStep(
            description="Research solutions for the encountered problem",
            tool_name="search",
            success_criteria="Found relevant information to address the issue",
            dependencies=[0]
        ))
    
    return recovery_steps


def _create_recovery_narration(diagnosis: ErrorDiagnosis, recovery_steps: List[PlanStep], retry_count: int) -> str:
    """Create user-friendly narration for error recovery."""
    severity_phrases = {
        ErrorSeverity.CRITICAL: "critical issue",
        ErrorSeverity.HIGH: "significant problem", 
        ErrorSeverity.MEDIUM: "issue",
        ErrorSeverity.LOW: "minor problem"
    }
    
    severity = severity_phrases.get(diagnosis.severity, "issue")
    retry_context = f" (after {retry_count} attempts)" if retry_count > 0 else ""
    
    narration = f"I've identified a {severity} that needs resolution{retry_context}. "
    narration += f"Let me implement a systematic solution to fix this.\n\n"
    narration += f"**Issue:** {diagnosis.root_cause}\n"
    narration += f"**Recovery approach:** {len(recovery_steps)} step{'s' if len(recovery_steps) != 1 else ''} to resolve this properly."
    
    return narration