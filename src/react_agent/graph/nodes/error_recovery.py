"""Enhanced error recovery with detailed error context analysis and multi-error handling."""

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
    """Detailed error diagnosis with specific error context."""
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    root_cause: str
    specific_errors: List[Dict[str, Any]]  # NEW: Detailed error breakdown
    error_count: int  # NEW: Number of distinct errors
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
    error_context: Optional[str] = Field(default="", description="Specific error context this step addresses")


class ContextAwareErrorRecovery(BaseModel):
    """Context-aware error recovery that scales with error complexity."""
    diagnosis: str = Field(description="Clear diagnosis of what went wrong")
    error_breakdown: List[str] = Field(description="List of specific errors identified")
    recovery_steps: List[RecoveryStep] = Field(
        description="Steps to fix all identified errors", 
        max_items=6  # Increased cap to handle multiple errors properly
    )
    fallback_approach: str = Field(default="", description="Alternative approach if fix fails")
    estimated_complexity: str = Field(description="Simple, moderate, or complex recovery")


# ENHANCED TIER 1: Context-aware template fixes
CONTEXT_AWARE_TEMPLATE_FIXES = {
    ErrorCategory.BUILD_ERROR: {
        "single_error": {
            "fix_step": {
                "description": "Fix the compilation error in the script",
                "tool_name": "write_file",
                "success_criteria": "Resolved the specific compilation issue"
            },
            "max_steps": 2
        },
        "multiple_errors": {
            "analyze_first": True,
            "fix_per_error": True,
            "max_steps": 4  # Allow more steps for multiple errors
        }
    },
    
    ErrorCategory.DEPENDENCY_MISSING: {
        "single_error": {
            "fix_step": {
                "description": "Install or configure missing dependency",
                "tool_name": "edit_project_config",
                "success_criteria": "Applied configuration to resolve dependency issue"
            },
            "max_steps": 2
        }
    },
    
    ErrorCategory.CONFIGURATION_ERROR: {
        "single_error": {
            "fix_step": {
                "description": "Update project configuration to resolve issue",
                "tool_name": "edit_project_config", 
                "success_criteria": "Applied necessary configuration changes"
            },
            "max_steps": 2
        }
    }
}

# TIER 0: Transient errors that should get micro-retries
TRANSIENT_ERROR_CATEGORIES = {
    ErrorCategory.NETWORK_ERROR,
    ErrorCategory.TOOL_MALFUNCTION
}


def extract_detailed_error_context(tool_result: Dict[str, Any], tool_name: str) -> List[Dict[str, Any]]:
    """Extract detailed error information from tool results."""
    detailed_errors = []
    
    if tool_name == "compile_and_test":
        # Extract compilation errors
        compilation_errors = tool_result.get("compilation_errors", [])
        for error in compilation_errors:
            detailed_errors.append({
                "type": "compilation_error",
                "file": error.get("file", ""),
                "line": error.get("line", 0),
                "error": error.get("error", ""),
                "severity": "error"
            })
        
        # Extract detailed errors from nested structure
        if "details" in tool_result and "errors" in str(tool_result["details"]):
            details = tool_result["details"]
            if isinstance(details, dict) and "errors" in details:
                for error in details.get("errors", []):
                    if isinstance(error, dict):
                        detailed_errors.append({
                            "type": "compilation_error",
                            "file": error.get("file", ""),
                            "line": error.get("line", 0),
                            "error": error.get("error", ""),
                            "severity": "error"
                        })
        
        # Also check warnings if no errors found
        if not detailed_errors:
            warnings = tool_result.get("warnings", [])
            for warning in warnings:
                detailed_errors.append({
                    "type": "compilation_warning",
                    "file": warning.get("file", ""),
                    "line": warning.get("line", 0),
                    "error": warning.get("warning", ""),
                    "severity": "warning"
                })
    
    elif tool_name == "write_file":
        # Extract file-specific errors
        error_msg = tool_result.get("error", "")
        if error_msg:
            detailed_errors.append({
                "type": "file_error",
                "file": tool_result.get("attempted_path", ""),
                "error": error_msg,
                "severity": "error"
            })
    
    elif tool_name == "search":
        # Extract search-specific errors
        error_msg = tool_result.get("error", "")
        if error_msg:
            detailed_errors.append({
                "type": "search_error", 
                "query": tool_result.get("query", ""),
                "error": error_msg,
                "severity": "error"
            })
    
    else:
        # Generic error extraction
        error_msg = tool_result.get("error", "")
        if error_msg:
            detailed_errors.append({
                "type": "tool_error",
                "tool": tool_name,
                "error": error_msg,
                "severity": "error"
            })
    
    return detailed_errors


def should_micro_retry(error_category: ErrorCategory, retry_count: int) -> bool:
    """Determine if error should get micro-retry instead of recovery planning."""
    return (error_category in TRANSIENT_ERROR_CATEGORIES and retry_count < 2)


def create_context_aware_template_recovery(error_category: ErrorCategory, failed_step: PlanStep, specific_errors: List[Dict[str, Any]]) -> List[PlanStep]:
    """Create context-aware template-based recovery that scales with error complexity."""
    if error_category not in CONTEXT_AWARE_TEMPLATE_FIXES:
        return []
    
    template_config = CONTEXT_AWARE_TEMPLATE_FIXES[error_category]
    error_count = len(specific_errors)
    
    recovery_steps = []
    
    # Handle build errors with multiple compilation issues
    if error_category == ErrorCategory.BUILD_ERROR and error_count > 1:
        # Multiple errors - create targeted fixes for each major issue
        unique_files = set()
        error_types = set()
        
        for error in specific_errors:
            if error.get("file"):
                unique_files.add(error["file"])
            
            error_msg = error.get("error", "").lower()
            if "inputsystem" in error_msg or "input" in error_msg:
                error_types.add("input_system")
            elif "charactercontroller" in error_msg or "character" in error_msg:
                error_types.add("character_controller")
            elif "namespace" in error_msg or "using" in error_msg:
                error_types.add("namespace")
            elif "reference" in error_msg or "not found" in error_msg:
                error_types.add("missing_reference")
        
        # Create specific fix steps based on error analysis
        if "input_system" in error_types:
            recovery_steps.append(PlanStep(
                description="Fix Input System namespace and reference issues in PlayerController.cs",
                tool_name="write_file",
                success_criteria="Resolved Input System compilation errors",
                dependencies=[]
            ))
        
        if "character_controller" in error_types:
            recovery_steps.append(PlanStep(
                description="Fix CharacterController type and namespace issues",
                tool_name="write_file", 
                success_criteria="Resolved CharacterController compilation errors",
                dependencies=[]
            ))
        
        if "missing_reference" in error_types or "namespace" in error_types:
            recovery_steps.append(PlanStep(
                description="Add missing using statements and namespace references",
                tool_name="write_file",
                success_criteria="Added all required namespace references",
                dependencies=[]
            ))
        
        # If no specific patterns found, create generic file-based fixes
        if not recovery_steps and unique_files:
            for i, file in enumerate(list(unique_files)[:2]):  # Max 2 files
                recovery_steps.append(PlanStep(
                    description=f"Fix compilation errors in {file}",
                    tool_name="write_file",
                    success_criteria=f"Resolved compilation issues in {file}",
                    dependencies=[]
                ))
        
        # Always end with compilation test
        recovery_steps.append(PlanStep(
            description="Verify all compilation errors are resolved",
            tool_name="compile_and_test",
            success_criteria="Clean compilation with no errors",
            dependencies=list(range(len(recovery_steps)))
        ))
        
        logger.info(f"Created context-aware recovery for {error_count} build errors: {len(recovery_steps)} steps")
        return recovery_steps[:4]  # Cap at 4 steps max
    
    # Single error or other categories - use simple template
    elif error_count == 1 or error_category != ErrorCategory.BUILD_ERROR:
        template = template_config.get("single_error", {})
        if template:
            fix_step_data = template["fix_step"]
            
            # Customize the description based on specific error
            description = fix_step_data["description"]
            if specific_errors and error_category == ErrorCategory.BUILD_ERROR:
                error_detail = specific_errors[0].get("error", "")
                if "InputSystem" in error_detail:
                    description = "Fix Input System namespace and import issues"
                elif "CharacterController" in error_detail:
                    description = "Fix CharacterController type reference issues"
            
            fix_step = PlanStep(
                description=description,
                tool_name=fix_step_data["tool_name"],
                success_criteria=fix_step_data["success_criteria"],
                dependencies=[]
            )
            
            rerun_step = PlanStep(
                description=f"Complete the original task: {failed_step.description}",
                tool_name=failed_step.tool_name,
                success_criteria=failed_step.success_criteria,
                dependencies=[0]
            )
            
            logger.info(f"Created template recovery for single {error_category}: 2 steps")
            return [fix_step, rerun_step]
    
    return []


async def diagnose_error(
    error_info: Dict[str, Any],
    failed_step: PlanStep,
    state: State,
    runtime: Runtime[Context]
) -> ErrorDiagnosis:
    """Enhanced error diagnosis with detailed error context extraction."""
    error_message = error_info.get("error", "Unknown error")
    tool_name = failed_step.tool_name
    tool_result = error_info.get("tool_result", {})
    
    logger.info(f"Diagnosing error with detailed context: {error_message}")
    
    # Extract detailed error information from tool result
    specific_errors = extract_detailed_error_context(tool_result, tool_name)
    error_count = len(specific_errors)
    
    logger.info(f"Extracted {error_count} specific errors: {[e.get('error', '')[:50] for e in specific_errors]}")
    
    # Quick categorization
    category = _categorize_error_fast(error_message, tool_name)
    severity = _assess_severity_fast(error_message, tool_name, state.retry_count, error_count)
    
    # Enhanced diagnosis with error details
    if specific_errors:
        error_summary = "; ".join([f"{e.get('file', tool_name)}: {e.get('error', '')[:50]}" for e in specific_errors[:3]])
        description = f"Tool {tool_name} failed with {error_count} errors: {error_summary}"
        
        # Create specific root cause based on errors
        if category == ErrorCategory.BUILD_ERROR:
            error_types = []
            for error in specific_errors:
                error_msg = error.get("error", "").lower()
                if "inputsystem" in error_msg:
                    error_types.append("Input System namespace issues")
                elif "charactercontroller" in error_msg:
                    error_types.append("CharacterController type issues")
                elif "namespace" in error_msg or "using" in error_msg:
                    error_types.append("Missing namespace references")
            
            root_cause = f"Compilation errors: {', '.join(set(error_types))}" if error_types else f"{error_count} compilation errors requiring fixes"
        else:
            root_cause = f"Tool {tool_name} failed: {error_message[:100]}"
    else:
        description = f"Tool {tool_name} failed: {error_message[:100]}"
        root_cause = f"Tool {tool_name} failed: {error_message[:100]}"
    
    return ErrorDiagnosis(
        category=category,
        severity=severity,
        description=description,
        root_cause=root_cause,
        specific_errors=specific_errors,
        error_count=error_count,
        suggested_fixes=[],
        required_tools=[tool_name],
        confidence=0.8,
        can_auto_fix=category in CONTEXT_AWARE_TEMPLATE_FIXES,
        workaround_available=True
    )


async def create_error_recovery_plan(
    diagnosis: ErrorDiagnosis,
    failed_step: PlanStep,
    state: State,
    runtime: Runtime[Context]
) -> List[PlanStep]:
    """ENHANCED tiered recovery plan creation with context awareness."""
    
    # TIER 1: Context-aware template-based recovery (preferred for known categories)
    if diagnosis.category in CONTEXT_AWARE_TEMPLATE_FIXES:
        logger.info(f"Using Tier 1 context-aware template recovery for {diagnosis.category} with {diagnosis.error_count} errors")
        template_recovery = create_context_aware_template_recovery(diagnosis.category, failed_step, diagnosis.specific_errors)
        if template_recovery:
            return template_recovery
    
    # TIER 2: Enhanced LLM recovery with error context
    logger.info(f"Using Tier 2 enhanced LLM recovery for {diagnosis.category} with {diagnosis.error_count} errors")
    
    context = runtime.context
    model = get_model(context.model)
    
    # Create detailed error context for LLM
    error_context = ""
    if diagnosis.specific_errors:
        error_context = "\nSPECIFIC ERRORS FOUND:\n"
        for i, error in enumerate(diagnosis.specific_errors[:5], 1):
            file = error.get("file", "")
            line = error.get("line", "")
            error_msg = error.get("error", "")
            error_context += f"{i}. {file}:{line} - {error_msg}\n"
    
    # ENHANCED recovery request with specific error details
    enhanced_recovery_request = f"""Create a targeted recovery plan for this Unity development failure:

ERROR DIAGNOSIS:
- Category: {diagnosis.category}
- Error Count: {diagnosis.error_count}
- Root Cause: {diagnosis.root_cause}

FAILED STEP: {failed_step.description}
TOOL: {failed_step.tool_name}

{error_context}

REQUIREMENTS:
1. Address ALL {diagnosis.error_count} specific errors identified above
2. Create steps that fix the actual problems, not generic solutions
3. If there are multiple compilation errors, fix them systematically
4. Maximum {4 if diagnosis.error_count > 2 else 2} steps total
5. Final step should verify the solution works

VALID TOOLS: search, get_project_info, create_asset, write_file, edit_project_config, get_script_snippets, compile_and_test, scene_management

Create a recovery plan that addresses the specific errors listed above, not just the general error category."""

    try:
        # Use enhanced schema that scales with error complexity
        structured_model = model.with_structured_output(ContextAwareErrorRecovery, method="function_calling")
        recovery_response = await structured_model.ainvoke([
            {"role": "system", "content": f"Create targeted recovery plans that address ALL specific errors. For {diagnosis.error_count} errors, create appropriate steps to fix each issue. Always end with verification."},
            {"role": "user", "content": enhanced_recovery_request}
        ])
        
        # Convert to internal format
        recovery_steps = []
        for step_data in recovery_response.recovery_steps:
            step = PlanStep(
                description=step_data.description,
                tool_name=step_data.tool_name,
                success_criteria=step_data.success_criteria,
                dependencies=[] if len(recovery_steps) == 0 else [len(recovery_steps) - 1]
            )
            recovery_steps.append(step)
        
        # Validate and adjust step count based on error complexity
        max_steps = 4 if diagnosis.error_count > 2 else 2
        if len(recovery_steps) > max_steps:
            logger.warning(f"LLM returned {len(recovery_steps)} steps, truncating to {max_steps}")
            recovery_steps = recovery_steps[:max_steps]
        
        # Ensure the last step is verification if dealing with compilation errors
        if diagnosis.category == ErrorCategory.BUILD_ERROR and recovery_steps:
            last_step = recovery_steps[-1]
            if last_step.tool_name != "compile_and_test":
                # Replace last step with verification
                recovery_steps[-1] = PlanStep(
                    description="Verify all compilation errors are resolved",
                    tool_name="compile_and_test",
                    success_criteria="Clean compilation with no errors",
                    dependencies=[len(recovery_steps) - 2] if len(recovery_steps) > 1 else []
                )
        
        logger.info(f"Created enhanced LLM recovery: {len(recovery_steps)} steps for {diagnosis.error_count} errors")
        return recovery_steps
        
    except Exception as e:
        logger.error(f"Enhanced LLM recovery failed: {str(e)}")
        return _create_context_aware_fallback_recovery(failed_step, diagnosis)


def _create_context_aware_fallback_recovery(failed_step: PlanStep, diagnosis: ErrorDiagnosis) -> List[PlanStep]:
    """Create context-aware fallback recovery based on error analysis."""
    recovery_steps = []
    
    if diagnosis.category == ErrorCategory.BUILD_ERROR and diagnosis.error_count > 1:
        # Multiple compilation errors - create systematic fix
        recovery_steps.append(PlanStep(
            description=f"Fix all {diagnosis.error_count} compilation errors systematically",
            tool_name="write_file",
            success_criteria=f"Resolved all {diagnosis.error_count} compilation issues"
        ))
        
        recovery_steps.append(PlanStep(
            description="Verify compilation errors are resolved",
            tool_name="compile_and_test",
            success_criteria="Clean compilation with no errors",
            dependencies=[0]
        ))
    else:
        # Standard fallback: diagnose + rerun
        recovery_steps.append(PlanStep(
            description="Analyze project state to understand the issue context",
            tool_name="get_project_info",
            success_criteria="Retrieved current project information"
        ))
        
        recovery_steps.append(PlanStep(
            description=f"Retry original task: {failed_step.description}",
            tool_name=failed_step.tool_name,
            success_criteria=failed_step.success_criteria,
            dependencies=[0]
        ))
    
    return recovery_steps


async def execute_error_recovery(
    state: State,
    runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Main error recovery execution with enhanced context awareness."""
    logger.info("=== EXECUTING CONTEXT-AWARE ERROR RECOVERY ===")
    
    if not state.error_context:
        logger.error("No error context found")
        return {"messages": [AIMessage(content="No error context found for recovery.")]}
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        logger.error("Invalid plan state")
        return {"messages": [AIMessage(content="Cannot recover: invalid plan state.")]}
    
    current_step = state.plan.steps[state.step_index]
    logger.info(f"Recovering from failure in step: {current_step.description}")
    
    # Enhanced diagnosis with detailed error context
    assessment = state.error_context.get("assessment")
    error_reason = assessment.reason if assessment else "Unknown error"
    
    error_info = {
        "error": error_reason,
        "tool_result": state.error_context.get("tool_result", {}),
        "tool_name": state.error_context.get("tool_name"),
        "step_description": state.error_context.get("step_description")
    }
    
    diagnosis = await diagnose_error(error_info, current_step, state, runtime)
    logger.info(f"Enhanced diagnosis: {diagnosis.category} - {diagnosis.error_count} errors - {diagnosis.root_cause}")
    
    # Create context-aware recovery plan
    recovery_steps = await create_error_recovery_plan(diagnosis, current_step, state, runtime)
    
    if not recovery_steps:
        logger.warning("No recovery steps created - using context-aware fallback")
        recovery_steps = _create_context_aware_fallback_recovery(current_step, diagnosis)
    
    # Replace failed step with recovery steps
    updated_plan = _replace_failed_step_with_recovery(state.plan, state.step_index, recovery_steps)
    
    # Create detailed recovery message
    recovery_message = _create_detailed_recovery_message(diagnosis, recovery_steps, state.retry_count)
    
    logger.info(f"Context-aware recovery plan: {len(recovery_steps)} steps for {diagnosis.error_count} errors")
    
    return {
        "plan": updated_plan,
        "step_index": state.step_index,
        "retry_count": 0,
        "current_assessment": None,
        "error_recovery_active": True,
        "needs_error_recovery": False,
        "error_context": None,
        "messages": [AIMessage(content=recovery_message)],
        "runtime_metadata": {
            **state.runtime_metadata,
            "error_diagnosis": {
                "category": diagnosis.category.value,
                "error_count": diagnosis.error_count,
                "recovery_tier": "template" if diagnosis.category in CONTEXT_AWARE_TEMPLATE_FIXES else "llm",
                "steps_created": len(recovery_steps)
            }
        }
    }


def _replace_failed_step_with_recovery(original_plan: ExecutionPlan, failed_step_index: int, recovery_steps: List[PlanStep]) -> ExecutionPlan:
    """Replace the failed step with context-aware recovery steps."""
    if not original_plan or not recovery_steps:
        return original_plan
    
    new_steps = []
    
    # Add all steps up to (but not including) the failed step
    for i in range(failed_step_index):
        new_steps.append(original_plan.steps[i])
    
    # Replace with recovery steps
    for recovery_step in recovery_steps:
        new_steps.append(recovery_step)
    
    # Add remaining steps after failed step
    for i in range(failed_step_index + 1, len(original_plan.steps)):
        new_steps.append(original_plan.steps[i])
    
    logger.info(f"Plan updated: {len(original_plan.steps)} → {len(new_steps)} steps (replaced with {len(recovery_steps)})")
    
    return ExecutionPlan(
        goal=original_plan.goal,
        steps=new_steps,
        metadata={
            **original_plan.metadata,
            "context_aware_recovery": True,
            "recovery_steps_count": len(recovery_steps)
        }
    )


def _create_detailed_recovery_message(diagnosis: ErrorDiagnosis, recovery_steps: List[PlanStep], retry_count: int) -> str:
    """Create detailed recovery message that explains the specific issues found."""
    retry_context = f" (after {retry_count} attempts)" if retry_count > 0 else ""
    
    if diagnosis.error_count > 1:
        message = f"I found {diagnosis.error_count} specific compilation errors{retry_context} that need systematic fixes:\n\n"
        
        # Show specific errors found
        for i, error in enumerate(diagnosis.specific_errors[:3], 1):
            file = error.get("file", "").split("/")[-1]  # Just filename
            error_msg = error.get("error", "")[:60]
            message += f"{i}. {file}: {error_msg}...\n"
        
        if diagnosis.error_count > 3:
            message += f"...and {diagnosis.error_count - 3} more errors\n"
        
        message += f"\n**Recovery Plan:** {len(recovery_steps)} targeted steps to fix all issues systematically."
    else:
        error_detail = ""
        if diagnosis.specific_errors:
            error = diagnosis.specific_errors[0]
            file = error.get("file", "").split("/")[-1]
            error_detail = f" in {file}: {error.get('error', '')[:60]}"
        
        message = f"I'll resolve this {diagnosis.category.value} issue{retry_context}{error_detail} with a targeted fix."
    
    return message


# Fast categorization functions (enhanced)
def _categorize_error_fast(error_message: str, tool_name: str) -> ErrorCategory:
    """Fast error categorization without LLM."""
    error_lower = error_message.lower()
    
    if any(term in error_lower for term in ["not found", "missing", "does not exist"]):
        return ErrorCategory.RESOURCE_NOT_FOUND
    elif any(term in error_lower for term in ["permission", "access", "forbidden"]):
        return ErrorCategory.PERMISSION_ERROR
    elif any(term in error_lower for term in ["config", "setting", "invalid parameter"]):
        return ErrorCategory.CONFIGURATION_ERROR
    elif any(term in error_lower for term in ["network", "connection", "timeout"]):
        return ErrorCategory.NETWORK_ERROR
    elif any(term in error_lower for term in ["compile", "build", "syntax", "cs0103", "cs0246"]):
        return ErrorCategory.BUILD_ERROR
    elif any(term in error_lower for term in ["dependency", "reference", "import"]):
        return ErrorCategory.DEPENDENCY_MISSING
    elif any(term in error_lower for term in ["invalid", "parameter", "argument"]):
        return ErrorCategory.INVALID_PARAMETER
    else:
        return ErrorCategory.UNKNOWN


def _assess_severity_fast(error_message: str, tool_name: str, retry_count: int, error_count: int = 1) -> ErrorSeverity:
    """Fast severity assessment considering error count."""
    if retry_count >= 2 or error_count > 3:
        return ErrorSeverity.HIGH
    
    error_lower = error_message.lower()
    if any(term in error_lower for term in ["critical", "fatal", "crash"]):
        return ErrorSeverity.CRITICAL
    elif any(term in error_lower for term in ["warning", "minor"]) and error_count == 1:
        return ErrorSeverity.LOW
    elif error_count > 1:
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.MEDIUM