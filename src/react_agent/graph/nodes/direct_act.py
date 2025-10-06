"""Enhanced direct action node with continuation request detection."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State
from react_agent.tools import TOOLS
from react_agent.utils import get_message_text, get_model


def _detect_continuation(user_request: str, recent_messages: List) -> Optional[Dict[str, Any]]:
    """Detect if user is continuing from a previous exchange."""
    request_lower = user_request.lower().strip()
    
    # Continuation indicators
    continuations = [
        "yes", "yeah", "yep", "sure", "ok", "okay", "please",
        "show me", "let me see", "i want to see", "go ahead",
        "yes please", "yeah please", "yes show", "sure show"
    ]
    
    # Check if this looks like a continuation
    is_continuation = (
        len(user_request.split()) <= 10 and  # Short requests
        any(request_lower.startswith(c) for c in continuations)
    )
    
    if not is_continuation:
        return None
    
    # Find what was offered and what tool was used
    last_ai_message = None
    last_tool_result = None
    
    for msg in reversed(recent_messages[-8:]):
        if isinstance(msg, AIMessage) and not last_ai_message:
            last_ai_message = get_message_text(msg)
        
        if isinstance(msg, ToolMessage) and not last_tool_result:
            try:
                last_tool_result = {
                    "tool_name": msg.name,
                    "result": json.loads(get_message_text(msg))
                }
            except:
                pass
    
    if last_ai_message and last_tool_result:
        return {
            "previous_message": last_ai_message,
            "tool_result": last_tool_result
        }
    
    return None


def _create_continuation_response(
    continuation_context: Dict[str, Any],
    user_request: str
) -> Optional[str]:
    """Generate response for continuation requests using stored tool data."""
    
    tool_name = continuation_context["tool_result"]["tool_name"]
    result = continuation_context["tool_result"]["result"]
    
    # Handle code_snippets continuation - show full code
    if tool_name == "code_snippets":
        snippets = result.get("snippets", [])
        if not snippets:
            return "I don't have any code snippets from the previous search."
        
        # Check what user wants
        request_lower = user_request.lower()
        
        if any(phrase in request_lower for phrase in ["full", "complete", "entire", "all"]):
            # Show full code from top result
            top_snippet = snippets[0]
            code = top_snippet.get("code", "")
            file_path = top_snippet.get("file_path", "unknown")
            
            if len(code) > 50:
                response = f"Here's the **complete source code** from `{file_path}`:\n\n```csharp\n{code}\n```"
                
                # Add metadata
                if top_snippet.get("class_name"):
                    response += f"\n\n**Class**: `{top_snippet['class_name']}`"
                if top_snippet.get("namespace"):
                    response += f"\n**Namespace**: `{top_snippet['namespace']}`"
                if top_snippet.get("line_count"):
                    response += f"\n**Lines**: {top_snippet['line_count']}"
                
                return response
            else:
                return f"The code snippet from `{file_path}` is quite short:\n\n```csharp\n{code}\n```"
        
        elif "more" in request_lower or "other" in request_lower:
            # Show other snippets
            if len(snippets) > 1:
                response = "Here are the other code snippets I found:\n\n"
                for i, snippet in enumerate(snippets[1:4], start=2):
                    file_path = snippet.get("file_path", "unknown")
                    score = snippet.get("relevance_score", 0)
                    response += f"{i}. **{file_path}** (relevance: {score:.2f})\n"
                    response += f"   {snippet.get('line_count', 0)} lines"
                    if snippet.get("class_name"):
                        response += f" - `{snippet['class_name']}`"
                    response += "\n\n"
                
                response += "Want me to show the full code for any of these?"
                return response
            else:
                return "That was the only code snippet I found."
    
    # Handle search_project continuation - show more details
    elif tool_name == "search_project":
        results = result.get("results", [])
        if not results:
            return "I don't have any project results from the previous search."
        
        request_lower = user_request.lower()
        
        if any(phrase in request_lower for phrase in ["more", "detail", "full", "all"]):
            # Show detailed results
            response = "Here are the full details from the project search:\n\n"
            for i, item in enumerate(results[:5], start=1):
                response += f"**{i}. {item.get('name', 'Unknown')}**\n"
                response += f"   Type: {item.get('type', 'Unknown')}\n"
                if item.get("path"):
                    response += f"   Path: `{item['path']}`\n"
                if item.get("components"):
                    response += f"   Components: {', '.join(item['components'][:5])}\n"
                response += "\n"
            
            return response
    
    # Handle file_operation continuation
    elif tool_name == "file_operation":
        operation = result.get("operation", "")
        
        if operation == "read":
            content = result.get("content", "")
            file_path = result.get("file_path", "unknown")
            
            if content:
                return f"Here's the **full content** from `{file_path}`:\n\n```csharp\n{content}\n```"
    
    return None


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
    """Enhanced direct action with continuation detection."""
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
    
    # CHECK FOR CONTINUATION FIRST
    continuation = _detect_continuation(user_request, state.messages)
    
    if continuation:
        # User is continuing from previous exchange
        continuation_response = _create_continuation_response(continuation, user_request)
        
        if continuation_response:
            return {
                "messages": [AIMessage(content=continuation_response)]
            }
        
        # If we can't handle continuation, add context and proceed normally
        # The tool result is still available in continuation["tool_result"]
    
    # Static system content for caching
    static_system_content = """You are a Unity/Unreal Engine development assistant that EXECUTES TOOLS to retrieve data from the user's project.

CRITICAL DISTINCTION:
- "What assets do I have?" → Use search_project tool to query their database
- "What is an asset?" → Provide knowledge-based explanation

- "Show me my GameObjects" → Use search_project tool to query hierarchy
- "How do I create GameObjects?" → Provide guidance

When the user asks for data FROM THEIR PROJECT, you MUST use tools to retrieve it.
When they ask conceptual questions ABOUT Unity, you provide informational responses.

KEY RULE: If they say "my", "in my project", "show me", "what are all" → USE TOOLS!

CONTINUATION AWARENESS:
If you previously showed someone a summary and they ask to "see more", "show full code", or "yes show me",
they're asking for more detail about what you just discussed. Check recent tool results and provide the full data."""
    
    # Check if this is a project data request
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
            except Exception:
                # Continue to normal flow if tool execution fails
                pass
    
    # Analyze request to determine approach
    analysis_prompt = f"""Analyze this Unity/game development request: "{user_request}"

Available tools:
- search_project: For querying project data, assets, hierarchy, components
- code_snippets: For semantic search through C# scripts by functionality
- file_operation: For reading, writing, modifying project files safely
- web_search: For finding tutorials, documentation, best practices

Response types:
1. TOOL_CALL: Use a specific tool if the request clearly maps to one tool operation
2. INFORMATIONAL: Provide direct knowledge-based answer
3. GUIDANCE: Offer step-by-step instructions

Choose the most efficient approach."""

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
            "how is this implemented", "code that does", "full code",
            "complete code", "source code"
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