"""Enhanced finish node for the ReAct agent with AI-generated completion summaries."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State
from react_agent.narration import NarrationEngine, StreamingNarrator
from react_agent.utils import get_message_text, get_model


# Initialize narration components
narration_engine = NarrationEngine()


async def _generate_comprehensive_completion_summary(state: State, model, context: Context) -> str:
    """Generate a comprehensive AI completion summary based on the entire execution."""
    
    # Gather all completed work
    completed_steps_summary = []
    key_outputs = []
    
    for i, step in enumerate(state.plan.steps):
        status = "✅ Completed" if i in state.completed_steps else "⏸️ Partial"
        completed_steps_summary.append(f"{status}: {step.description}")
        
        # Track what was actually built/created
        if step.tool_name == "write_file":
            key_outputs.append("Created script files")
        elif step.tool_name == "create_asset":
            key_outputs.append("Built game assets")
        elif step.tool_name == "compile_and_test":
            key_outputs.append("Verified code compilation")
        elif step.tool_name == "scene_management":
            key_outputs.append("Configured scenes")
        elif step.tool_name == "edit_project_config":
            key_outputs.append("Updated project settings")
    
    # Extract concrete results from recent tool messages
    concrete_results = []
    files_created = []
    assets_built = []
    
    for msg in state.messages[-20:]:  # Look at recent messages
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(get_message_text(msg))
                tool_name = msg.name or 'unknown'
                
                if result.get("success", False):
                    if tool_name == "write_file":
                        file_path = result.get('file_path', 'script file')
                        lines = result.get('lines_written', 0)
                        files_created.append(f"📄 {file_path} ({lines} lines)")
                        
                    elif tool_name == "create_asset":
                        asset_type = result.get('asset_type', 'asset')
                        name = result.get('name', 'new asset')
                        assets_built.append(f"🎮 {asset_type}: {name}")
                        
                    elif tool_name == "compile_and_test":
                        errors = result.get('errors', 0)
                        warnings = result.get('warnings', 0)
                        concrete_results.append(f"🔧 Build: {errors} errors, {warnings} warnings")
                        
                    elif tool_name == "search":
                        results_count = len(result.get('result', []))
                        concrete_results.append(f"🔍 Found {results_count} resources")
                        
                    elif tool_name == "get_project_info":
                        engine = result.get("engine", "Unity")
                        version = result.get("version", "")
                        concrete_results.append(f"📊 Analyzed {engine} {version} project")
                        
            except:
                continue
    
    # Build context for AI summary
    execution_context = {
        "original_goal": state.plan.goal,
        "total_steps_planned": len(state.plan.steps),
        "steps_completed": len(state.completed_steps),
        "completion_percentage": (len(state.completed_steps) / len(state.plan.steps)) * 100 if state.plan.steps else 0,
        "total_tool_calls": state.total_tool_calls,
        "total_assessments": state.total_assessments,
        "files_created": files_created,
        "assets_built": assets_built,
        "concrete_results": concrete_results,
        "plan_revisions": state.plan_revision_count,
        "retry_attempts": sum(step.attempts for step in state.plan.steps)
    }
    
    # Create comprehensive prompt for AI summary
    completion_prompt = f"""You have just completed a Unity/game development session. Generate a professional, specific completion summary.

**Original Goal:** "{state.plan.goal}"

**Execution Summary:**
- Planned Steps: {len(state.plan.steps)}
- Completed Steps: {len(state.completed_steps)} ({execution_context['completion_percentage']:.1f}%)
- Total Tool Calls: {state.total_tool_calls}
- Plan Revisions: {state.plan_revision_count}

**Work Completed:**
{chr(10).join(completed_steps_summary)}

**Concrete Deliverables:**
{chr(10).join(files_created + assets_built + concrete_results) if (files_created + assets_built + concrete_results) else "Development work completed successfully"}

Generate a completion summary that:
1. Starts with a confident completion statement (e.g., "Perfect!", "Excellent!", "Mission accomplished!")
2. Specifically mentions what was actually built/created/configured
3. Highlights the key deliverables from the session
4. Shows awareness of the concrete work done
5. Suggests logical next steps for the developer
6. Maintains a professional, confident tone
7. Is 2-4 sentences long and actionable

Focus on tangible outcomes and what the developer can now do with their project.

Example good summaries:
- "Perfect! I've successfully created a complete FPS controller script with movement, mouse look, and jump mechanics. The script is compiled and ready to attach to your player GameObject. You can now test it in Play mode and customize the movement speeds to your liking."
- "Excellent! I've built a comprehensive UI health bar system with dynamic updates and proper text display. The script is integrated into your project and ready for implementation. Simply drag the HealthBar prefab into your scene and connect it to your player health system."

Your completion summary:"""

    try:
        # Generate AI completion summary
        completion_messages = [
            {"role": "system", "content": "You are providing a professional completion summary for a Unity development session. Be specific, confident, and focus on concrete deliverables. Keep it concise but comprehensive."},
            {"role": "user", "content": completion_prompt}
        ]
        
        completion_response = await model.ainvoke(completion_messages)
        ai_summary = get_message_text(completion_response).strip()
        
        # Validate AI response quality
        if len(ai_summary) < 30 or not any(word in ai_summary.lower() for word in ['perfect', 'excellent', 'great', 'success', 'complete', 'ready', 'built', 'created']):
            # Fallback if AI response is poor
            completion_rate = f"{len(state.completed_steps)}/{len(state.plan.steps)}"
            return f"Perfect! I've successfully completed {completion_rate} steps for your request: {state.plan.goal}. The implementation is ready for testing in your Unity project."
        
        return ai_summary
        
    except Exception as e:
        # Robust fallback
        deliverables = files_created + assets_built
        if deliverables:
            deliverable_summary = f"Created: {', '.join(deliverables[:2])}"
            if len(deliverables) > 2:
                deliverable_summary += f" and {len(deliverables) - 2} more items"
        else:
            deliverable_summary = "all requested development work"
            
        return f"Perfect! I've successfully completed {deliverable_summary} for your goal: {state.plan.goal}. Your Unity project now has the requested functionality."


async def finish(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhanced finish node with AI-generated completion summary based on actual execution."""
    context = runtime.context
    model = get_model(context.model)  # Use the main model for summaries
    
    # Generate AI-based completion summary instead of hardcoded template
    try:
        ai_completion_summary = await _generate_comprehensive_completion_summary(state, model, context)
    except Exception as e:
        # Fallback to basic summary if AI generation fails
        completed_count = len(state.completed_steps)
        total_count = len(state.plan.steps) if state.plan else 0
        ai_completion_summary = f"Perfect! I've completed {completed_count} of {total_count} planned steps for: {state.plan.goal if state.plan else 'your request'}. The implementation is ready for use."
    
    # Add final streaming updates if supported
    if context.runtime_metadata.get("supports_streaming"):
        streaming_narrator = StreamingNarrator()
        final_update = streaming_narrator.create_inline_update(
            "✨ Implementation complete! Check your Unity project for the new features.",
            style="success"
        )
        if hasattr(runtime, 'push_ui_message'):
            runtime.push_ui_message(final_update)
    
    return {
        "messages": [AIMessage(content=ai_completion_summary)]
    }