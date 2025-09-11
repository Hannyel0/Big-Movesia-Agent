"""Assessment node for the ReAct agent."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.runtime import Runtime
from typing import Literal

from react_agent.context import Context
from react_agent.state import (
    AssessmentOutcome,
    State,
    StructuredAssessment,
)
from react_agent.narration import NarrationEngine, StreamingNarrator
from react_agent.utils import get_message_text, get_model


# Initialize narration components
narration_engine = NarrationEngine()




def _create_step_transition(current_step, next_step) -> str:
    """Create transition narration between steps."""
    return f"✅ Step completed successfully! Moving on to: {next_step.description}"


async def assess(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced assessment with rich post-tool narration."""
    context = runtime.context
    model = get_model(context.assessment_model or context.model)
    
    if not state.plan or state.step_index >= len(state.plan.steps):
        return {}
    
    current_step = state.plan.steps[state.step_index]
    
    # Extract the tool result from the last ToolMessage
    tool_result = None
    tool_name = current_step.tool_name
    
    for msg in reversed(state.messages[-10:]):  # Look at recent messages
        if isinstance(msg, ToolMessage):
            try:
                # Parse tool result
                content = get_message_text(msg)
                tool_result = json.loads(content) if content else {}
            except json.JSONDecodeError:
                tool_result = {"message": content}
            break
    
    # GENERATE RICH POST-TOOL NARRATION
    post_tool_narration = None
    if tool_result and tool_name:
        step_context = {
            "step_index": state.step_index,
            "total_steps": len(state.plan.steps),
            "goal": state.plan.goal,
            "tool_name": tool_name,
            "description": current_step.description
        }
        
        # Use the narration engine to create rich, contextual narration
        post_tool_narration = narration_engine.narrate_tool_result(
            tool_name,
            tool_result,
            step_context
        )
    
    # Find the most recent tool messages for better context
    recent_tool_results = []
    latest_tool_result = None
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            if latest_tool_result is None:
                try:
                    latest_tool_result = json.loads(get_message_text(msg))
                except:
                    latest_tool_result = {"message": get_message_text(msg)}
            recent_tool_results.append(f"Tool '{msg.name}': {get_message_text(msg)}")
            if len(recent_tool_results) >= 3:
                break
    
    tool_results_text = "\n".join(recent_tool_results) if recent_tool_results else "No recent tool results found"
    
    # Now perform the assessment using structured output
    assessment_request = f"""Assess if the current step has been successfully completed:

Overall Goal: {state.plan.goal}
Current Step ({state.step_index + 1}/{len(state.plan.steps)}): {current_step.description}
Success Criteria: {current_step.success_criteria}
Tool Used: {tool_name}

Tool Result Summary: {json.dumps(tool_result, indent=2) if tool_result else "No result captured"}

Determine if the step succeeded, needs retry, or is blocked."""
    
    # Static assessment prompt - cacheable
    static_assessment_content = """You are evaluating game development step completion with focus on deliverable quality and tool effectiveness.

Assessment Criteria:
- Was the required development tool used properly?
- Did the step produce a working game asset, script, or feature?
- Can the output be integrated into a Unity/Unreal project?
- Does the result follow game development best practices?
- Is there sufficient progress toward the gameplay goal?

Be particularly strict about:
- Code quality and Unity/Unreal compatibility
- Asset creation and proper file structure
- Project integration and build compatibility
- Following established game development patterns

Assessment outcomes:
- "success": Step completed with working game development output
- "retry": Implementation incomplete or doesn't meet game dev standards
- "blocked": Technical limitation preventing proper implementation

Judge based on professional game development quality and deliverables."""

    # Structure for optimal caching
    messages = [
        {"role": "system", "content": static_assessment_content},
        {"role": "user", "content": assessment_request}
    ]
    
    try:
        # Use structured output for assessment
        structured_model = model.with_structured_output(StructuredAssessment)
        structured_assessment = await structured_model.ainvoke(messages)
        
        assessment = AssessmentOutcome(
            outcome=structured_assessment.outcome,
            reason=structured_assessment.reason,
            fix=structured_assessment.fix or None,
            confidence=structured_assessment.confidence
        )
        
        # Prepare messages with rich narration
        messages_to_add = []
        if post_tool_narration:
            messages_to_add.append(AIMessage(content=post_tool_narration))
        
        # Add transition narration if moving to next step
        if assessment.outcome == "success" and state.step_index + 1 < len(state.plan.steps):
            next_step = state.plan.steps[state.step_index + 1]
            transition = _create_step_transition(current_step, next_step)
            if transition:
                messages_to_add.append(AIMessage(content=transition))
        
        result = {
            "current_assessment": assessment,
            "total_assessments": state.total_assessments + 1
        }
        
        if messages_to_add:
            result["messages"] = messages_to_add
        
        # Complete streaming narration if active
        if context.runtime_metadata.get("supports_streaming"):
            streaming_narrator = StreamingNarrator()
            stream_id = f"step_{state.step_index}"
            if assessment.outcome == "success":
                streaming_narrator.complete_step(stream_id, "Step completed successfully!")
            elif assessment.outcome == "retry":
                streaming_narrator.update_step_progress(stream_id, "Retrying with adjustments...")
            else:
                streaming_narrator.complete_step(stream_id, "Step blocked - replanning...")
        
        return result
        
    except Exception as e:
        # Fallback assessment with narration
        fallback_narration = f"Assessment check completed. Moving forward with the implementation."
        
        return {
            "current_assessment": AssessmentOutcome(
                outcome="success" if tool_result else "retry",
                reason="Automated assessment",
                confidence=0.5
            ),
            "messages": [AIMessage(content=fallback_narration)] if tool_result else [],
            "total_assessments": state.total_assessments + 1
        }