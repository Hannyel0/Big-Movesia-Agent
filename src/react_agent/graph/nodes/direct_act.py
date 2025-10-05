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


def _is_project_data_request(request: str) -> bool:
    """Detect if user wants data FROM their project vs. guidance ABOUT Unity."""
    request_lower = request.lower()
    
    # Strong indicators of project data requests
    project_indicators = [
        "my project", "my assets", "my scripts", "my gameobjects",
        "in my project", "what assets", "show me", "list all",
        "find all", "what scripts", "what gameobjects",
        "in the project", "current project"
    ]
    
    if any(indicator in request_lower for indicator in project_indicators):
        return True
    
    # Question words that indicate data retrieval (not conceptual questions)
    data_questions = [
        "what are all", "what are my", "what assets", "what scripts",
        "what gameobjects", "which assets", "how many assets"
    ]
    
    if any(pattern in request_lower for pattern in data_questions):
        return True
    
    # Conceptual questions that should NOT execute tools
    conceptual_patterns = [
        "what is a", "what is the", "how does a", "how do i",
        "what does", "how to", "explain", "teach me"
    ]
    
    if any(pattern in request_lower for pattern in conceptual_patterns):
        return False
    
    return False


async def direct_act(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute simple requests directly without planning overhead using production tools."""
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
    
    # Static system content for caching
    static_system_content = """You are a Unity/Unreal Engine development assistant that EXECUTES TOOLS to retrieve data from the user's project.

CRITICAL DISTINCTION:
- "What assets do I have?" → Use search_project tool to query their database
- "What is an asset?" → Provide knowledge-based explanation

- "Show me my GameObjects" → Use search_project tool to query hierarchy
- "How do I create GameObjects?" → Provide guidance

When the user asks for data FROM THEIR PROJECT, you MUST use tools to retrieve it.
When they ask conceptual questions ABOUT Unity, you provide informational responses.

KEY RULE: If they say "my", "in my project", "show me", "what are all" → USE TOOLS!"""
    
    # NEW: Check if this is a project data request
    if _is_project_data_request(user_request):
        # Force tool execution for project data requests
        tool_name = _determine_tool_from_request(user_request, "")
        
        if tool_name:
            tool_prompt = f"""Execute the {tool_name} tool to retrieve data for: "{user_request}"

The user is asking for data FROM their project. Use the tool to query their actual project database.
Do NOT provide guidance on how to do it themselves - they want YOU to execute the query and show them the results."""

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
                # Continue to normal flow if tool execution fails
                pass
    
    # Analyze request to determine the most appropriate single tool or response
    analysis_prompt = f"""Analyze this Unity/game development request and determine the best direct response approach: "{user_request}"

Available tools:
- search_project: For querying project data, assets, hierarchy, components
- code_snippets: For semantic search through C# scripts by functionality
- file_operation: For reading, writing, modifying project files safely
- web_search: For finding tutorials, documentation, best practices

Response types:
1. TOOL_CALL: Use a specific tool if the request clearly maps to one tool operation
2. INFORMATIONAL: Provide direct knowledge-based answer for "how to", "what is", "explain" questions
3. GUIDANCE: Offer step-by-step instructions for procedural questions

Choose the most efficient approach and specify:
- Response type
- Tool name (if TOOL_CALL)
- Brief reasoning

Focus on solving the user's immediate need efficiently."""

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
    """Determine the most appropriate production tool based on request content."""
    request_lower = user_request.lower()
    
    # PRIORITY 1: Project data patterns - check these FIRST
    project_data_patterns = {
        "search_project": [
            "what assets", "my assets", "list assets", "show assets",
            "what gameobjects", "my gameobjects", "list gameobjects",
            "show gameobjects", "what's in my project", "project info",
            "all the assets", "all assets", "all gameobjects",
            "in my project", "in the project", "my project",
            "show me my", "list my", "what are my", "what do i have",
            "current project", "project structure", "project details"
        ],
        "code_snippets": [
            "my scripts", "what scripts", "list scripts", "show scripts",
            "all scripts", "all the scripts", "scripts in my project"
        ]
    }
    
    # Check project-specific patterns FIRST (highest priority)
    for tool_name, patterns in project_data_patterns.items():
        if any(pattern in request_lower for pattern in patterns):
            return tool_name
    
    # PRIORITY 2: General tool patterns
    tool_patterns = {
        "search_project": [
            "find assets", "list components", "project hierarchy"
        ],
        "code_snippets": [
            "code example", "script template", "code snippet", "show me code",
            "example code", "sample script", "find code", "existing code",
            "how is this implemented", "code that does"
        ],
        "file_operation": [
            "create script", "write script", "create file", "generate code", 
            "write code", "modify file", "read file", "update script",
            "new script", "make script"
        ],
        "web_search": [
            "search for", "find tutorials", "look up", "research", 
            "documentation", "best practices", "how to", "learn about",
            "Unity documentation", "tutorials"
        ]
    }
    
    # Check analysis text for explicit tool mention
    for tool_name in tool_patterns.keys():
        if tool_name in analysis_text:
            return tool_name
    
    # Fall back to pattern matching in request
    for tool_name, patterns in tool_patterns.items():
        if any(pattern in request_lower for pattern in patterns):
            return tool_name
    
    # Default to web_search for research-type requests
    if any(word in request_lower for word in ["how", "what", "why", "when", "where"]):
        return "web_search"
    
    return None