"""Direct action node for simple requests that don't require planning."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State
from react_agent.tools import TOOLS
from react_agent.utils import get_message_text, get_model


async def direct_act(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute simple requests directly without planning overhead."""
    context = runtime.context
    model = get_model(context.model)
    
    # Extract user request
    user_request = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_request = get_message_text(msg)
            break
    
    if not user_request:
        return {
            "messages": [AIMessage(content="I need a clear request to help you with your Unity development.")]
        }
    
    # Analyze request to determine the most appropriate single tool or response
    analysis_prompt = f"""Analyze this Unity/game development request and determine the best direct response approach: "{user_request}"

Available tools:
- search: For finding tutorials, documentation, best practices
- get_project_info: For project structure and configuration details  
- create_asset: For creating scripts, prefabs, materials, scenes
- write_file: For writing script files or configurations
- edit_project_config: For changing project settings
- get_script_snippets: For code examples and templates
- compile_and_test: For testing and compilation
- scene_management: For scene operations

Response types:
1. TOOL_CALL: Use a specific tool if the request clearly maps to one tool operation
2. INFORMATIONAL: Provide direct knowledge-based answer for "how to", "what is", "explain" questions
3. GUIDANCE: Offer step-by-step instructions for procedural questions

Choose the most efficient approach and specify:
- Response type
- Tool name (if TOOL_CALL)
- Brief reasoning

Focus on solving the user's immediate need efficiently."""

    # Static system content for caching
    static_system_content = """You are a Unity/Unreal Engine development assistant providing direct, efficient responses to simple requests.

For TOOL_CALL responses: Call the appropriate tool with suitable parameters
For INFORMATIONAL responses: Provide clear, actionable information from your knowledge
For GUIDANCE responses: Give step-by-step instructions

Be concise, practical, and focused on immediate value to the developer."""

    # Structure messages for analysis
    analysis_messages = [
        {"role": "system", "content": static_system_content},
        {"role": "user", "content": analysis_prompt}
    ]
    
    try:
        # Get analysis of best approach
        analysis_response = await model.ainvoke(analysis_messages)
        analysis_text = get_message_text(analysis_response).lower()
        
        # Determine if we should make a tool call or provide direct information
        should_use_tool = "tool_call" in analysis_text
        
        if should_use_tool:
            # Extract likely tool name from analysis or user request
            tool_name = _determine_tool_from_request(user_request, analysis_text)
            
            if tool_name:
                # Create tool-specific prompt
                tool_prompt = f"""Execute the {tool_name} tool to respond to: "{user_request}"

Use appropriate parameters based on the request. Focus on providing exactly what the user asked for."""

                # Bind specific tool and execute
                model_with_tools = model.bind_tools(TOOLS)
                
                try:
                    tool_response = await model_with_tools.ainvoke(
                        [
                            {"role": "system", "content": static_system_content},
                            {"role": "user", "content": tool_prompt}
                        ],
                        tool_choice={"type": "function", "function": {"name": tool_name}}
                    )
                    
                    return {
                        "messages": [tool_response],
                        "total_tool_calls": state.total_tool_calls + (
                            len(tool_response.tool_calls) if tool_response.tool_calls else 0
                        )
                    }
                    
                except Exception as tool_error:
                    # Fallback to informational response if tool fails
                    fallback_prompt = f"""The tool execution failed. Provide a helpful direct answer to: "{user_request}"

Give practical Unity/game development guidance based on your knowledge."""
                    
                    fallback_response = await model.ainvoke([
                        {"role": "system", "content": static_system_content},
                        {"role": "user", "content": fallback_prompt}
                    ])
                    
                    return {"messages": [fallback_response]}
        
        # Provide informational response
        info_prompt = f"""Provide a direct, helpful answer to this Unity/game development question: "{user_request}"

Give practical guidance, explanations, or instructions based on your knowledge. Be specific and actionable."""
        
        info_response = await model.ainvoke([
            {"role": "system", "content": static_system_content},
            {"role": "user", "content": info_prompt}
        ])
        
        return {"messages": [info_response]}
        
    except Exception as e:
        # Robust error handling
        error_response = f"I can help you with that. Let me provide some guidance on: {user_request}"
        
        return {
            "messages": [AIMessage(content=error_response)],
            "tool_errors": state.tool_errors + [{"error": str(e), "context": "direct_act"}]
        }


def _determine_tool_from_request(user_request: str, analysis_text: str) -> str:
    """Determine the most appropriate tool based on request content."""
    request_lower = user_request.lower()
    
    # Direct tool mapping based on request patterns
    tool_patterns = {
        "get_project_info": [
            "project info", "what's in my project", "project structure", 
            "project details", "current project", "project status"
        ],
        "search": [
            "search for", "find tutorials", "look up", "research", 
            "documentation", "best practices", "how to"
        ],
        "get_script_snippets": [
            "code example", "script template", "code snippet", "show me code",
            "example code", "sample script"
        ],
        "compile_and_test": [
            "compile", "test", "build", "check for errors", "run test"
        ],
        "create_asset": [
            "create script", "new script", "make script", "create prefab",
            "new prefab", "create material", "new material"
        ],
        "write_file": [
            "write script", "create file", "generate code", "write code"
        ],
        "scene_management": [
            "scene", "level", "create scene", "new scene", "manage scene"
        ],
        "edit_project_config": [
            "settings", "configure", "project settings", "build settings"
        ]
    }
    
    # Check analysis text first for explicit tool mention
    for tool_name in tool_patterns.keys():
        if tool_name in analysis_text:
            return tool_name
    
    # Fall back to pattern matching in request
    for tool_name, patterns in tool_patterns.items():
        if any(pattern in request_lower for pattern in patterns):
            return tool_name
    
    # Default to search for research-type requests
    if any(word in request_lower for word in ["how", "what", "why", "when", "where"]):
        return "search"
    
    return None