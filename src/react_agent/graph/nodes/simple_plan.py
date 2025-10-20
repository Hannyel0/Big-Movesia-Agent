"""Intelligent simple planning node that adapts to straightforward requests."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import (
    ExecutionPlan,
    PlanStep,
    State,
    StructuredExecutionPlan,
)
from react_agent.utils import get_message_text, get_model
from react_agent.memory import (
    inject_memory_into_prompt,
    extract_entities_from_request,
    extract_topics_from_request,
)


def _analyze_request_nature(user_request: str) -> Dict[str, Any]:
    """Analyze the nature of the request to determine optimal approach."""
    request_lower = user_request.lower()
    
    analysis = {
        "request_type": "unknown",
        "complexity_indicators": [],
        "suggested_approach": "adaptive",
        "user_intent": "unclear"
    }
    
    # Analyze user intent
    if any(word in request_lower for word in ["what", "how", "why", "explain", "tell me"]):
        analysis["user_intent"] = "information_seeking"
        analysis["request_type"] = "informational"
    elif any(word in request_lower for word in ["create", "make", "build", "generate"]):
        analysis["user_intent"] = "creation"
        analysis["request_type"] = "creation"
    elif any(word in request_lower for word in ["fix", "debug", "error", "problem", "broken"]):
        analysis["user_intent"] = "problem_solving"
        analysis["request_type"] = "troubleshooting"
    elif any(word in request_lower for word in ["setup", "configure", "settings", "install"]):
        analysis["user_intent"] = "configuration"
        analysis["request_type"] = "setup"
    
    # Analyze complexity indicators
    if any(word in request_lower for word in ["simple", "basic", "quick", "easy"]):
        analysis["complexity_indicators"].append("user_wants_simple")
    if any(word in request_lower for word in ["advanced", "complex", "sophisticated"]):
        analysis["complexity_indicators"].append("user_wants_advanced")
    if "step by step" in request_lower or "tutorial" in request_lower:
        analysis["complexity_indicators"].append("user_wants_guidance")
    
    # Determine optimal approach
    if analysis["user_intent"] == "information_seeking":
        analysis["suggested_approach"] = "research_first"
    elif analysis["request_type"] == "creation" and "user_wants_simple" in analysis["complexity_indicators"]:
        analysis["suggested_approach"] = "direct_implementation"
    elif analysis["request_type"] == "troubleshooting":
        analysis["suggested_approach"] = "diagnose_then_solve"
    else:
        analysis["suggested_approach"] = "context_then_implement"
    
    return analysis


async def simple_plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Create adaptive, intelligent simple plans based on request analysis."""
    context = runtime.context
    model = get_model(context.planning_model or context.model)
    
    # Extract user request
    user_request = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_request = get_message_text(msg)
            break
    
    if not user_request:
        return {"messages": [AIMessage(content="I need a clear request to create a plan.")]}
    
    # ✅ MEMORY: Update focus for this request
    if state.memory:
        entities = extract_entities_from_request(user_request)
        topics = extract_topics_from_request(user_request)
        if entities or topics:
            await state.memory.update_focus(entities, topics)
            print(f"🧠 [SimplePlan] Updated focus - Entities: {entities[:3]}, Topics: {topics[:3]}")
    
    # Analyze the request intelligently
    request_analysis = _analyze_request_nature(user_request)
    
    # Get complexity reasoning from previous classification
    complexity_reasoning = state.runtime_metadata.get(
        "complexity_reasoning", 
        "Straightforward task requiring focused approach"
    )
    
    # INTELLIGENT SIMPLE PLANNING PROMPT
    intelligent_simple_request = f"""Analyze this request and create an optimal simple plan: "{user_request}"

REQUEST ANALYSIS:
- Request Type: {request_analysis['request_type']}
- User Intent: {request_analysis['user_intent']}
- Complexity Indicators: {request_analysis['complexity_indicators']}
- Suggested Approach: {request_analysis['suggested_approach']}
- Classification Reasoning: {complexity_reasoning}

INTELLIGENT PLANNING STRATEGY:
Based on the analysis, choose the most effective approach:

1. **Research First** (for information seeking):
   - web_search → provide comprehensive answer (maybe + search_project for context)

2. **Direct Implementation** (for simple creation):
   - code_snippets → write_file (skip excessive research for basic requests)

3. **Diagnose Then Solve** (for troubleshooting):
   - search_project → code_snippets → (web_search if needed) → modify_file

4. **Context Then Implement** (for standard creation):
   - search_project → write_file → (search_project validation if needed)

EFFICIENCY PRINCIPLES:
- Don't over-research simple requests
- Don't under-research complex ones
- Consider what the user actually needs vs. what templates suggest
{{ ... }}
- 1-3 steps maximum, but make them count
- Each step should add real value

Create a smart, streamlined plan that efficiently solves this specific request."""

    # CORRECTED ADAPTIVE SYSTEM CONTENT for simple planning  
    base_adaptive_system_content = """You are creating efficient, intelligent plans for straightforward Unity/Unreal development tasks.

TOOL PURPOSE CLARIFICATION:
**search_project**: Query indexed Unity project database using natural language
**code_snippets**: Semantic search through C# scripts by functionality
**read_file**: Read file contents safely (no approval)
**write_file**: Create/write files (requires approval)
**modify_file**: Modify existing files (requires approval)
**delete_file**: Delete files (requires approval)
**move_file**: Move/rename files (requires approval)
**web_search**: Research external Unity documentation and tutorials

VALID TOOLS: search_project, code_snippets, read_file, write_file, modify_file, delete_file, move_file, web_search

INTELLIGENT EFFICIENCY:
- **Information requests**: web_search
- **Understanding existing code**: code_snippets → read_file
- **Modifying existing features**: code_snippets → modify_file
- **Creating new features**: web_search → write_file
- **Project inspection**: search_project
- **Asset discovery**: search_project
- **File cleanup**: delete_file
- **File reorganization**: move_file

DECISION FLOWCHART:
1. Need to find existing code by functionality? → code_snippets
2. Need to learn new Unity concepts? → web_search
3. Need project structure/asset info? → search_project
4. Need to read files? → read_file
5. Need to create files? → write_file
6. Need to modify files? → modify_file
7. Need to delete files? → delete_file
8. Need to move/rename files? → move_file

Create plans that use production tools for real Unity project integration."""

    # ✅ MEMORY: Inject memory context into planning prompt
    adaptive_system_content = await inject_memory_into_prompt(
        base_prompt=base_adaptive_system_content,
        state=state,
        include_patterns=True,
        include_episodes=True
    )

    # Structure messages for adaptive planning
    messages = [
        {"role": "system", "content": adaptive_system_content},
        {"role": "user", "content": intelligent_simple_request}
    ]
    
    try:
        # Use structured output for reliable planning
        structured_model = model.with_structured_output(StructuredExecutionPlan)
        structured_response = await structured_model.ainvoke(messages)
        
        # Enforce maximum step limit but with flexibility
        max_steps = 3
        limited_steps = structured_response.steps[:max_steps]
        
        # Convert to internal format
        steps = []
        for step_data in limited_steps:
            step = PlanStep(
                description=step_data.description,
                success_criteria=step_data.success_criteria,
                tool_name=step_data.tool_name,
                dependencies=step_data.dependencies
            )
            steps.append(step)
        
        plan = ExecutionPlan(
            goal=structured_response.goal,
            steps=steps,
            metadata={
                "planning_mode": "intelligent_simple", 
                "request_analysis": request_analysis,
                "original_step_count": len(structured_response.steps)
            }
        )
        
        # ✅ MEMORY: Store plan in memory
        if state.memory:
            state.memory.add_plan({
                "goal": plan.goal,
                "steps": [{"description": s.description, "tool": s.tool_name} for s in steps]
            })
            print(f"🧠 [SimplePlan] Plan stored in memory: {len(steps)} steps")
        
        # Create adaptive narration
        step_count = len(steps)
        approach_description = {
            "research_first": "research-focused approach",
            "direct_implementation": "direct implementation approach", 
            "diagnose_then_solve": "diagnostic approach",
            "context_then_implement": "context-aware approach"
        }.get(request_analysis["suggested_approach"], "focused approach")
        
        narration = f"I'll handle this with a {approach_description} in {step_count} step{'s' if step_count != 1 else ''}:"
        
        for i, step in enumerate(steps, 1):
            narration += f"\n{i}. {step.description}"
        
        return {
            "plan": plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=narration)]
        }
        
    except Exception as e:
        # Intelligent fallback based on request analysis
        fallback_steps = _create_intelligent_fallback_plan(user_request, request_analysis)
        fallback_plan = ExecutionPlan(
            goal=user_request,
            steps=fallback_steps,
            metadata={"planning_mode": "intelligent_simple_fallback"}
        )
        
        fallback_narration = f"I'll take an adaptive approach based on your {request_analysis['request_type']} request."
        
        return {
            "plan": fallback_plan,
            "step_index": 0,
            "retry_count": 0,
            "messages": [AIMessage(content=fallback_narration)]
        }


def _create_intelligent_fallback_plan(user_request: str, analysis: Dict[str, Any]) -> list[PlanStep]:
    """Create intelligent fallback plan based on request analysis."""
    
    request_type = analysis.get("request_type", "unknown")
    user_intent = analysis.get("user_intent", "unclear")
    
    if user_intent == "information_seeking":
        # Information requests - research focused
        return [
            PlanStep(
                description=f"Research comprehensive information about: {user_request}",
                tool_name="web_search",
                success_criteria="Found detailed, relevant information to answer the question"
            )
        ]
    
    elif request_type == "troubleshooting":
        # Problem solving - diagnose first
        return [
            PlanStep(
                description="Analyze current project state to identify issues",
                tool_name="search_project", 
                success_criteria="Retrieved project information and identified potential problems"
            ),
            PlanStep(
                description="Find existing code that might be causing problems",
                tool_name="code_snippets",
                success_criteria="Located relevant code implementations to analyze",
                dependencies=[0]
            )
        ]
    
    elif request_type == "creation":
        # Creation requests - efficient implementation
        if "simple" in analysis.get("complexity_indicators", []):
            # Direct approach for simple creation
            return [
                PlanStep(
                    description="Find existing code patterns for the request",
                    tool_name="code_snippets",
                    success_criteria="Retrieved suitable implementation examples"
                ),
                PlanStep(
                    description="Create the requested implementation",
                    tool_name="write_file",
                    success_criteria="Successfully created the requested file",
                    dependencies=[0]
                )
            ]
        else:
            # Context-aware approach for standard creation
            return [
                PlanStep(
                    description="Understand project context and requirements",
                    tool_name="search_project",
                    success_criteria="Retrieved relevant project information"
                ),
                PlanStep(
                    description="Create the requested implementation",
                    tool_name="write_file",
                    success_criteria="Successfully created the requested item",
                    dependencies=[0]
                )
            ]
    
    elif request_type == "setup":
        # Configuration requests
        return [
            PlanStep(
                description="Review current project configuration",
                tool_name="search_project",
                success_criteria="Understood current project setup and requirements"
            ),
            PlanStep(
                description="Apply requested configuration changes",
                tool_name="modify_file",
                success_criteria="Successfully updated project configuration",
                dependencies=[0]
            )
        ]
    
    else:
        # Generic intelligent fallback
        return [
            PlanStep(
                description="Research and understand the specific requirements",
                tool_name="web_search",
                success_criteria="Found relevant information and approach"
            ),
            PlanStep(
                description="Implement solution based on research",
                tool_name="code_snippets" if "code" in user_request.lower() else "search_project",
                success_criteria="Provided working solution for the request", 
                dependencies=[0]
            )
        ]