"""Fixed error recovery system with proper context handling and Pydantic schemas."""

from __future__ import annotations

import json
import logging
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

# Set up logging
logger = logging.getLogger(__name__)


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


class RecoveryStep(BaseModel):
    """Individual recovery step with proper Pydantic schema."""
    description: str = Field(description="What this recovery step does")
    tool_name: str = Field(description="Tool to use for this step")
    success_criteria: str = Field(description="How to know if this step succeeded")


class StructuredErrorRecovery(BaseModel):
    """Fixed structured error recovery for OpenAI compatibility."""
    diagnosis: str = Field(description="Clear diagnosis of what went wrong")
    recovery_steps: List[RecoveryStep] = Field(description="Steps to fix the error")
    fallback_approach: str = Field(default="", description="Alternative approach if fix fails")
    estimated_complexity: str = Field(description="Simple, moderate, or complex recovery")


async def diagnose_error(
    error_info: Dict[str, Any],
    failed_step: PlanStep,
    state: State,
    runtime: Runtime[Context]
) -> ErrorDiagnosis:
    """Intelligently diagnose what went wrong with a tool call."""
    logger.info("Starting error diagnosis")
    
    context = runtime.context
    model = get_model(context.model)
    
    # Extract error details
    error_message = error_info.get("error", "Unknown error")
    tool_name = failed_step.tool_name
    step_description = failed_step.description
    
    logger.info(f"Diagnosing error: {error_message}")
    logger.info(f"Failed tool: {tool_name}")
    
    # SHORTENED diagnostic prompt for concise analysis
    diagnostic_prompt = f"""Analyze this Unity development tool failure and provide a CONCISE diagnosis:

FAILURE DETAILS:
- Tool: {tool_name}
- Step: {step_description}
- Error: {error_message}

Provide a brief diagnosis (2-3 sentences) that identifies:
1. What specifically went wrong
2. The likely root cause (missing dependency, config issue, etc.)
3. Whether this is automatically fixable

Keep your response focused and under 150 words."""

    try:
        diagnostic_response = await model.ainvoke([
            {"role": "system", "content": "You are a Unity development expert providing concise error diagnosis. Be specific but brief - focus on actionable insights in 2-3 sentences maximum."},
            {"role": "user", "content": diagnostic_prompt}
        ])
        
        diagnosis_text = get_message_text(diagnostic_response)
        logger.info(f"LLM diagnosis: {diagnosis_text}")
        
        # Categorize the error based on patterns
        category = _categorize_error(error_message, tool_name, diagnosis_text)
        severity = _assess_severity(error_message, tool_name, state.retry_count)
        
        # Determine if we can auto-fix
        can_auto_fix = _can_auto_fix(category, tool_name, error_message)
        
        # Generate suggested fixes
        suggested_fixes = _generate_fix_suggestions(category, tool_name, error_message, diagnosis_text)
        
        root_cause = _extract_root_cause(diagnosis_text, error_message)
        logger.info(f"Extracted root cause: {root_cause}")
        
        return ErrorDiagnosis(
            category=category,
            severity=severity,
            description=diagnosis_text,  # This will now be shorter
            root_cause=root_cause,
            suggested_fixes=suggested_fixes,
            required_tools=_determine_required_tools(category, tool_name),
            confidence=0.8,
            can_auto_fix=can_auto_fix,
            workaround_available=_has_workaround(category, tool_name)
        )
        
    except Exception as e:
        logger.error(f"Error in diagnosis: {str(e)}")
        # Fallback diagnosis
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            description=f"Tool {tool_name} encountered an error: {error_message}",
            root_cause=f"Tool {tool_name} failed: {error_message}",
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
    """Create recovery steps to fix the error and replace the failed step."""
    logger.info("Creating error recovery plan")
    
    context = runtime.context
    model = get_model(context.model)
    
    # ENHANCED recovery request that includes the original step's intent
    recovery_request = f"""Create specific error recovery steps for this Unity/game development failure:

ERROR DIAGNOSIS:
- Category: {diagnosis.category}
- Severity: {diagnosis.severity}
- Root Cause: {diagnosis.root_cause}
- Can Auto-Fix: {diagnosis.can_auto_fix}

ORIGINAL FAILED STEP:
- Description: {failed_step.description}
- Tool: {failed_step.tool_name}
- Success Criteria: {failed_step.success_criteria}
- Intent: What this step was trying to accomplish

SUGGESTED FIXES:
{chr(10).join(f"- {fix}" for fix in diagnosis.suggested_fixes)}

IMPORTANT: Create recovery steps that will REPLACE the failed step entirely:
1. Fix the underlying issue that caused the failure
2. Include a final recovery step that accomplishes the SAME GOAL as the original failed step
3. Do NOT duplicate the original tool call - the recovery plan should complete the original intent

VALID TOOLS: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

Example recovery patterns that REPLACE the failed step:
- Failed compile_and_test: search for solution → edit_project_config to fix issue → compile_and_test (completes original intent)
- Failed write_file: get_project_info to understand structure → write_file with corrected approach (completes original intent)
- Failed get_script_snippets: search for code examples → create_asset with found patterns (alternative way to complete intent)

Create a recovery plan that fixes the issue AND completes the original step's purpose."""

    try:
        # Use function_calling method to avoid OpenAI structured output issues
        structured_model = model.with_structured_output(StructuredErrorRecovery, method="function_calling")
        recovery_response = await structured_model.ainvoke([
            {"role": "system", "content": "You are creating Unity/Unreal error recovery plans that REPLACE failed steps. The recovery plan must fix the issue AND accomplish the original step's goal. Never duplicate tool calls."},
            {"role": "user", "content": recovery_request}
        ])
        
        logger.info(f"Created {len(recovery_response.recovery_steps)} recovery steps")
        
        # Convert to internal format
        recovery_steps = []
        for i, recovery_step in enumerate(recovery_response.recovery_steps):
            step = PlanStep(
                description=recovery_step.description,
                tool_name=recovery_step.tool_name,
                success_criteria=recovery_step.success_criteria,
                dependencies=[]
            )
            recovery_steps.append(step)
            logger.info(f"Recovery step {i+1}: {recovery_step.description} using {recovery_step.tool_name}")
        
        # Validate that the recovery plan doesn't just duplicate the failed tool
        _validate_recovery_plan(recovery_steps, failed_step)
        
        return recovery_steps
        
    except Exception as e:
        logger.error(f"Error creating recovery plan: {str(e)}")
        # Fallback recovery steps that replace the original
        return _create_replacement_recovery_steps(diagnosis, failed_step)


async def execute_error_recovery(
    state: State,
    runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Main error recovery execution node."""
    logger.info("=== EXECUTING ERROR RECOVERY ===")
    
    # Extract error information - use error_context set by assess node
    if not state.error_context:
        logger.error("No error context found")
        return {"messages": [AIMessage(content="No error context found for recovery.")]}
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        logger.error("Invalid plan state")
        return {"messages": [AIMessage(content="Cannot recover: invalid plan state.")]}
    
    current_step = state.plan.steps[state.step_index]
    logger.info(f"Recovering from failure in step: {current_step.description}")
    
    # Step 1: Diagnose the error using the stored error context
    assessment = state.error_context.get("assessment")
    error_reason = assessment.reason if assessment else "Unknown error"
    
    error_info = {
        "error": error_reason,
        "tool_result": state.error_context.get("tool_result", {}),
        "tool_name": state.error_context.get("tool_name"),
        "step_description": state.error_context.get("step_description")
    }
    
    diagnosis = await diagnose_error(error_info, current_step, state, runtime)
    logger.info(f"Diagnosis complete. Root cause: {diagnosis.root_cause}")
    
    # Step 2: Create recovery steps that REPLACE the failed step
    recovery_steps = await create_error_recovery_plan(diagnosis, current_step, state, runtime)
    
    if not recovery_steps:
        logger.warning("No recovery steps created - using fallback")
        recovery_steps = _create_replacement_recovery_steps(diagnosis, current_step)
    
    # Step 3: REPLACE the failed step with recovery steps (no duplication)
    updated_plan = _replace_failed_step_with_recovery(state.plan, state.step_index, recovery_steps)
    
    # Step 4: Create user-facing message with VISIBLE diagnosis
    recovery_message = _create_comprehensive_recovery_message(diagnosis, recovery_steps, state.retry_count)
    
    logger.info(f"Recovery plan created with {len(recovery_steps)} steps replacing the failed step")
    
    return {
        "plan": updated_plan,
        "step_index": state.step_index,  # Start executing the first recovery step (at current position)
        "retry_count": 0,  # Reset retry count
        "current_assessment": None,
        "error_recovery_active": True,
        "needs_error_recovery": False,  # Clear the flag
        "error_context": None,  # Clear error context
        "messages": [AIMessage(content=recovery_message)],
        "runtime_metadata": {
            **state.runtime_metadata,
            "error_diagnosis": {
                "category": diagnosis.category.value,
                "severity": diagnosis.severity.value,
                "root_cause": diagnosis.root_cause,
                "can_auto_fix": diagnosis.can_auto_fix
            },
            "recovery_replaced_step": True,  # Indicate replacement, not insertion
            "recovery_steps_count": len(recovery_steps)
        }
    }


def _replace_failed_step_with_recovery(original_plan: ExecutionPlan, failed_step_index: int, recovery_steps: List[PlanStep]) -> ExecutionPlan:
    """Replace the failed step with recovery steps (eliminates duplication)."""
    if not original_plan or not recovery_steps:
        return original_plan
    
    new_steps = []
    
    # Add all steps up to (but not including) the failed step
    for i in range(failed_step_index):
        new_steps.append(original_plan.steps[i])
    
    # Replace the failed step with recovery steps
    for recovery_step in recovery_steps:
        new_steps.append(recovery_step)
    
    # Add all remaining steps after the failed step (skip the failed step entirely)
    for i in range(failed_step_index + 1, len(original_plan.steps)):
        new_steps.append(original_plan.steps[i])
    
    logger.info(f"Plan updated: Original had {len(original_plan.steps)} steps, new plan has {len(new_steps)} steps")
    logger.info(f"Replaced step at index {failed_step_index} with {len(recovery_steps)} recovery steps")
    
    return ExecutionPlan(
        goal=original_plan.goal,
        steps=new_steps,
        metadata={
            **original_plan.metadata,
            "recovery_replaced": True,  # Different from "recovery_inserted"
            "replaced_step_index": failed_step_index,
            "recovery_steps_count": len(recovery_steps)
        }
    )


def _validate_recovery_plan(recovery_steps: List[PlanStep], failed_step: PlanStep) -> None:
    """Validate that recovery plan properly replaces the failed step without duplication."""
    if not recovery_steps:
        logger.warning("Empty recovery plan provided")
        return
    
    # Check if any recovery step uses the exact same tool as the failed step
    failed_tool = failed_step.tool_name
    recovery_tools = [step.tool_name for step in recovery_steps]
    
    if failed_tool in recovery_tools:
        logger.info(f"Recovery plan includes {failed_tool} - this is OK as it's replacing the failed attempt")
    else:
        logger.warning(f"Recovery plan doesn't include {failed_tool} - may not complete original intent")
    
    # Log the replacement strategy
    logger.info(f"Recovery validation: {len(recovery_steps)} steps will replace failed '{failed_step.description}'")


def _create_replacement_recovery_steps(diagnosis: ErrorDiagnosis, failed_step: PlanStep) -> List[PlanStep]:
    """Create fallback recovery steps that REPLACE the failed step."""
    recovery_steps = []
    
    # Strategy: Fix the issue, then retry the original intent
    
    # Step 1: Always diagnose the current state
    recovery_steps.append(PlanStep(
        description="Analyze project state to understand the error context",
        tool_name="get_project_info",
        success_criteria="Retrieved current project information and identified potential issues"
    ))
    
    # Step 2: Category-specific fix
    if diagnosis.category == ErrorCategory.BUILD_ERROR:
        recovery_steps.append(PlanStep(
            description="Research solution for the compilation issue",
            tool_name="search",
            success_criteria="Found specific solution for the build problem",
            dependencies=[]
        ))
        # Step 3: Retry the original intent (compile)
        recovery_steps.append(PlanStep(
            description=f"Complete the original task: {failed_step.description}",
            tool_name=failed_step.tool_name,  # Same tool as failed step
            success_criteria=failed_step.success_criteria,
            dependencies=[]
        ))
        
    elif diagnosis.category == ErrorCategory.DEPENDENCY_MISSING:
        recovery_steps.append(PlanStep(
            description="Install or configure the missing dependency",
            tool_name="edit_project_config",
            success_criteria="Applied configuration changes to resolve dependency issue",
            dependencies=[]
        ))
        # Step 3: Retry the original intent
        recovery_steps.append(PlanStep(
            description=f"Complete the original task: {failed_step.description}",
            tool_name=failed_step.tool_name,
            success_criteria=failed_step.success_criteria,
            dependencies=[]
        ))
        
    elif diagnosis.category == ErrorCategory.CONFIGURATION_ERROR:
        recovery_steps.append(PlanStep(
            description="Update project configuration to resolve the issue",
            tool_name="edit_project_config", 
            success_criteria="Applied necessary configuration changes",
            dependencies=[]
        ))
        # Step 3: Retry the original intent
        recovery_steps.append(PlanStep(
            description=f"Complete the original task: {failed_step.description}",
            tool_name=failed_step.tool_name,
            success_criteria=failed_step.success_criteria,
            dependencies=[]
        ))
        
    else:
        # Generic recovery: research + retry
        recovery_steps.append(PlanStep(
            description="Research solutions for the encountered problem",
            tool_name="search",
            success_criteria="Found relevant information to address the issue",
            dependencies=[]
        ))
        # Step 3: Retry the original intent
        recovery_steps.append(PlanStep(
            description=f"Complete the original task with new approach: {failed_step.description}",
            tool_name=failed_step.tool_name,
            success_criteria=failed_step.success_criteria,
            dependencies=[]
        ))
    
    logger.info(f"Created {len(recovery_steps)} replacement recovery steps ending with {recovery_steps[-1].tool_name}")
    return recovery_steps


def _insert_recovery_steps_before_failed_step(original_plan: ExecutionPlan, failed_step_index: int, recovery_steps: List[PlanStep]) -> ExecutionPlan:
    """DEPRECATED: Use _replace_failed_step_with_recovery instead to avoid duplication."""
    logger.warning("Using deprecated insertion method - this may cause tool duplication")
    return _replace_failed_step_with_recovery(original_plan, failed_step_index, recovery_steps)


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
    """Extract root cause from diagnosis with better parsing."""
    if not diagnosis:
        return f"Tool failure: {error_message[:100]}"
    
    # Look for specific patterns in the diagnosis
    sentences = diagnosis.split('.')
    
    # First, look for sentences with causal keywords
    for sentence in sentences:
        sentence = sentence.strip()
        if any(word in sentence.lower() for word in ["because", "due to", "caused by", "root cause", "the issue is"]):
            return sentence[:200]  # Limit length
    
    # If no causal sentences, look for dependency/configuration issues
    for sentence in sentences:
        sentence = sentence.strip()
        if any(word in sentence.lower() for word in ["missing", "package", "dependency", "not found", "configuration"]):
            return sentence[:200]
    
    # If no specific patterns, use first substantial sentence
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            return sentence[:200]
    
    # Fallback
    return f"Tool {error_message.split(':')[0] if ':' in error_message else 'execution'} failed: {error_message[:100]}"


def _create_fallback_recovery_steps(diagnosis: ErrorDiagnosis, failed_step: PlanStep) -> List[PlanStep]:
    """DEPRECATED: Use _create_replacement_recovery_steps instead."""
    logger.warning("Using deprecated fallback recovery steps")
    return _create_replacement_recovery_steps(diagnosis, failed_step)


def _create_comprehensive_recovery_message(diagnosis: ErrorDiagnosis, recovery_steps: List[PlanStep], retry_count: int) -> str:
    """Create comprehensive user-facing recovery message with visible diagnosis."""
    severity_phrases = {
        ErrorSeverity.CRITICAL: "critical issue",
        ErrorSeverity.HIGH: "significant problem", 
        ErrorSeverity.MEDIUM: "issue",
        ErrorSeverity.LOW: "minor problem"
    }
    
    severity = severity_phrases.get(diagnosis.severity, "issue")
    retry_context = f" (after {retry_count} attempts)" if retry_count > 0 else ""
    
    # Create comprehensive message that shows the diagnosis and replacement strategy
    message = f"I've encountered a {severity}{retry_context} that I can systematically resolve.\n\n"
    message += f"**Problem Analysis:**\n{diagnosis.description}\n\n"
    message += f"**Recovery Strategy:** I'll replace the failed step with a {len(recovery_steps)}-step solution:\n"
    
    for i, step in enumerate(recovery_steps, 1):
        # Add context for the final step if it's retrying the original tool
        if i == len(recovery_steps) and recovery_steps[-1].tool_name in step.description:
            message += f"{i}. {step.description} (completing the original goal)\n"
        else:
            message += f"{i}. {step.description}\n"
    
    message += f"\nThis approach will fix the issue and complete the original objective without duplication."
    
    return message


def _create_recovery_narration(diagnosis: ErrorDiagnosis, recovery_steps: List[PlanStep], retry_count: int) -> str:
    """Create user-friendly narration for error recovery (kept for backward compatibility)."""
    # This is the old function - now just calls the comprehensive one
    return _create_comprehensive_recovery_message(diagnosis, recovery_steps, retry_count)