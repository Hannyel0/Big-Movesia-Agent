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
from react_agent.memory import (
    extract_entities_from_request,
    extract_topics_from_request,
    inject_memory_into_prompt,
)


def _detect_continuation(
    user_request: str, 
    recent_messages: List,
    memory_manager: Optional[Any] = None  # ✅ ADD THIS
) -> Optional[Dict[str, Any]]:
    """Detect if user is continuing from a previous exchange."""
    request_lower = user_request.lower().strip()
    
    # Continuation indicators
    continuations = [
        "yes", "yeah", "yep", "sure", "ok", "okay", "please",
        "show me", "let me see", "i want to see", "go ahead",
        "yes please", "yeah please", "yes show", "sure show"
    ]
    
    # ✅ NEW: Add more context-aware continuation patterns
    contextual_continuations = [
        "which one", "which", "what about", "how about",
        "the biggest", "the largest", "the smallest",
        "tell me more", "more details", "more info",
        "the first", "the second", "the last"
    ]
    
    # Check if this looks like a continuation
    is_continuation = (
        (len(user_request.split()) <= 10 and  # Short requests
         any(request_lower.startswith(c) for c in continuations)) or
        any(pattern in request_lower for pattern in contextual_continuations)  # ✅ NEW
    )
    
    if not is_continuation:
        return None
    
    # ✅ NEW: FIRST check working memory for recent tool results
    if memory_manager and hasattr(memory_manager, 'working_memory'):
        recent_tools = memory_manager.working_memory.recent_tool_results
        if recent_tools:
            # Get the most recent tool result
            latest_tool = recent_tools[-1]
            
            # Also try to find the AI message that presented this result
            last_ai_message = None
            for msg in reversed(recent_messages[-5:]):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_ai_message = get_message_text(msg)
                    break
            
            # ✅ Extract structured data if available
            result_data = latest_tool["result"]
            structured = result_data.get("results_structured", []) if isinstance(result_data, dict) else []
            
            return {
                "previous_message": last_ai_message or "Previous results",
                "tool_result": {
                    "tool_name": latest_tool["tool_name"],
                    "result": latest_tool["result"],
                    "structured_data": structured  # ✅ ADD: For easier access
                }
            }
    
    # Fallback: Check state.messages (for same-turn continuations)
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
        # ✅ Extract structured data if available
        result_data = last_tool_result.get("result", {})
        structured = result_data.get("results_structured", []) if isinstance(result_data, dict) else []
        
        return {
            "previous_message": last_ai_message,
            "tool_result": {
                **last_tool_result,
                "structured_data": structured  # ✅ ADD: For easier access
            }
        }
    
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
    """Enhanced direct action with comprehensive logging."""
    
    print(f"\n{'='*60}")
    print(f"🎬 [DIRECT_ACT] ===== STARTING DIRECT ACTION =====")
    print(f"{'='*60}")
    
    context = runtime.context
    model = get_model(context.model)
    
    # Extract user request
    user_request = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_request = get_message_text(msg)
            break
    
    if not user_request:
        print(f"❌ [DirectAct] No user request found")
        return {
            "messages": [AIMessage(content="I need a clear request to help you with your Unity development.")]
        }
    
    print(f"\n💬 [DirectAct] User Request:")
    print(f"   \"{user_request}\"")
    print(f"   Length: {len(user_request)} chars, {len(user_request.split())} words")
    
    # ✅ FIXED: Detect continuation using memory manager
    continuation = None
    if user_request:
        continuation = _detect_continuation(user_request, state.messages, state.memory)
        
        if continuation:
            print(f"\n🔄 [DirectAct] Continuation detected!")
            print(f"   Previous: {continuation.get('previous_message', '')[:80]}...")
            if continuation.get('tool_result'):
                tool_name = continuation['tool_result'].get('tool_name', 'unknown')
                print(f"   Tool result available: {tool_name}")
    
    # ✅ NEW: Log memory state
    print(f"\n🧠 [DirectAct] Memory State:")
    if state.memory:
        if hasattr(state.memory, 'working_memory'):
            wm = state.memory.working_memory
            tool_count = len(wm.recent_tool_results)
            print(f"   Tool results in memory: {tool_count}")
            
            if tool_count > 0:
                print(f"   Recent tool results:")
                for i, tool in enumerate(wm.recent_tool_results[-3:], 1):
                    print(f"      {i}. {tool['tool_name']}: {tool['summary'][:60]}")
                    print(f"         Timestamp: {tool['timestamp']}")
            else:
                print(f"   No tool results stored")
            
            print(f"   Focus entities: {wm.focus_entities[:3] if wm.focus_entities else 'None'}")
            print(f"   Focus topics: {wm.focus_topics[:3] if wm.focus_topics else 'None'}")
            print(f"   User intent: {wm.user_intent[:60] if wm.user_intent else 'None'}...")
        else:
            print(f"   Working memory not available")
            
        # Record user message
        state.memory.add_message("user", user_request)
        
        entities = extract_entities_from_request(user_request)
        topics = extract_topics_from_request(user_request)
        if entities or topics:
            await state.memory.update_focus(entities, topics)
            print(f"   Updated focus - Entities: {entities[:3]}, Topics: {topics[:3]}")
    else:
        print(f"   ⚠️  No memory available")
    
    # Base system content
    base_system_content = """You are a Unity/Unreal Engine development assistant that EXECUTES TOOLS to retrieve data from the user's project.

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
    
    # ✅ FIXED: Add continuation context if detected
    if continuation:
        tool_result = continuation.get('tool_result', {})
        tool_name = tool_result.get('tool_name', 'unknown')
        result_data = tool_result.get('result', {})
        
        continuation_context = f"""

# CONTINUATION CONTEXT

The user is continuing from a previous interaction. Here's what they saw:

**Previous Message:** {continuation.get('previous_message', 'N/A')[:200]}

**Previous Tool Result ({tool_name}):**
{json.dumps(result_data, indent=2)[:1000]}

The user's current request ("{user_request}") is likely asking for more details, full code, or additional information about the above result.
Provide the complete information they're requesting based on this context."""
        
        base_system_content += continuation_context
        print(f"\n📋 [DirectAct] Added continuation context ({len(continuation_context)} chars)")
    
    # Inject memory context
    print(f"\n🔄 [DirectAct] Injecting memory context into prompt...")
    static_system_content = await inject_memory_into_prompt(
        base_prompt=base_system_content,
        state=state,
        include_patterns=True,
        include_episodes=True
    )
    
    base_length = len(base_system_content)
    enhanced_length = len(static_system_content)
    added_context = enhanced_length - base_length
    
    print(f"   Base prompt: {base_length} chars")
    print(f"   Enhanced prompt: {enhanced_length} chars")
    print(f"   Memory context added: {added_context} chars")
    
    if added_context > 0:
        print(f"\n📝 [DirectAct] Memory Context Preview:")
        memory_section = static_system_content[base_length:base_length+300]
        print(f"   {memory_section}...")
    
    # ✅ NEW: Check if this is a project data request
    is_project_query = _is_project_data_request(user_request)
    print(f"\n🔍 [DirectAct] Request Analysis:")
    print(f"   Is project data request: {is_project_query}")
    
    if is_project_query:
        tool_name = _determine_tool_from_request(user_request, "")
        print(f"   Inferred tool: {tool_name}")
        
        if tool_name:
            print(f"\n🔧 [DirectAct] Forcing tool execution for project data request")
            print(f"   Tool: {tool_name}")
            
            tool_prompt = f"""Execute the {tool_name} tool to retrieve data for: "{user_request}"

The user is asking for data FROM their project. Use the tool to query their actual project database.
Do NOT provide guidance on how to do it themselves - they want YOU to execute the query and show them the results."""
            
            print(f"   Tool prompt length: {len(tool_prompt)} chars")
            
            model_with_tools = model.bind_tools(TOOLS)
            
            try:
                print(f"   ⏳ Executing {tool_name}...")
                
                tool_response = await model_with_tools.ainvoke(
                    [
                        {"role": "system", "content": static_system_content},
                        {"role": "user", "content": tool_prompt}
                    ],
                    tool_choice={"type": "function", "function": {"name": tool_name}}
                )
                
                print(f"\n✅ [DirectAct] Tool execution requested:")
                if tool_response.tool_calls:
                    for tc in tool_response.tool_calls:
                        print(f"   Tool: {tc.get('name')}")
                        print(f"   Args: {str(tc.get('args', {}))[:100]}...")
                else:
                    print(f"   ⚠️  No tool calls generated")
                
                # Store expectation in memory
                if state.memory and tool_response.tool_calls:
                    for tc in tool_response.tool_calls:
                        state.memory.add_message(
                            "system",
                            f"Executing {tc.get('name')} with query: {tc.get('args', {}).get('query', 'N/A')}",
                            metadata={"tool_call": True, "tool_name": tc.get('name')}
                        )
                
                print(f"\n{'='*60}")
                print(f"🎬 [DIRECT_ACT] ===== DIRECT ACTION COMPLETE =====")
                print(f"{'='*60}\n")
                
                return {
                    "messages": [tool_response],
                    "total_tool_calls": state.total_tool_calls + (
                        len(tool_response.tool_calls) if tool_response.tool_calls else 0
                    )
                }
            except Exception as e:
                print(f"   ❌ Tool execution failed: {str(e)}")
                print(f"   Falling through to normal processing")
    
    # Normal LLM response path
    print(f"\n🤖 [DirectAct] Using LLM for direct response (no forced tool)")
    
    # Analyze request to determine approach
    analysis_prompt = f"""Analyze this Unity/game development request: "{user_request}"

Available tools:
- search_project: For querying project data, assets, hierarchy, components
- code_snippets: For semantic search through C# scripts by functionality
- read_file: For reading file contents
- write_file: For creating new files (requires approval)
- modify_file: For modifying existing files (requires approval)
- delete_file: For deleting files (requires approval)
- move_file: For moving/renaming files (requires approval)
- web_search: For finding tutorials, documentation, best practices

Response types:
1. TOOL_CALL: Use a specific tool if the request clearly maps to one tool operation
2. INFORMATIONAL: Provide direct knowledge-based answer
3. GUIDANCE: Offer step-by-step instructions

Choose the most efficient approach."""
    
    print(f"\n📤 [DirectAct] Sending analysis request to LLM...")
    print(f"   Analysis prompt: {len(analysis_prompt)} chars")
    
    analysis_messages = [
        {"role": "system", "content": static_system_content},
        {"role": "user", "content": analysis_prompt}
    ]
    
    try:
        print(f"   ⏳ Waiting for analysis...")
        
        analysis_response = await model.ainvoke(analysis_messages)
        analysis_text = get_message_text(analysis_response).lower()
        
        print(f"\n✅ [DirectAct] Analysis received:")
        print(f"   Response length: {len(analysis_text)} chars")
        print(f"   Preview: {analysis_text[:200]}...")
        
        should_use_tool = "tool_call" in analysis_text
        print(f"   Should use tool: {should_use_tool}")
        
        if should_use_tool:
            tool_name = _determine_tool_from_request(user_request, analysis_text)
            print(f"   Determined tool: {tool_name}")
            
            if tool_name:
                print(f"\n🔧 [DirectAct] Executing tool based on analysis")
                
                tool_prompt = f"""Execute the {tool_name} tool to respond to: "{user_request}"

Use appropriate parameters based on the request. Focus on providing exactly what the user asked for."""
                
                model_with_tools = model.bind_tools(TOOLS)
                
                try:
                    print(f"   ⏳ Executing {tool_name}...")
                    
                    tool_response = await model_with_tools.ainvoke(
                        [
                            {"role": "system", "content": static_system_content},
                            {"role": "user", "content": tool_prompt}
                        ],
                        tool_choice={"type": "function", "function": {"name": tool_name}}
                    )
                    
                    print(f"\n✅ [DirectAct] Tool execution requested:")
                    if tool_response.tool_calls:
                        for tc in tool_response.tool_calls:
                            print(f"   Tool: {tc.get('name')}")
                            print(f"   Args: {str(tc.get('args', {}))[:100]}...")
                    
                    # Store in memory
                    if state.memory and tool_response.tool_calls:
                        for tc in tool_response.tool_calls:
                            state.memory.add_message(
                                "system",
                                f"Executing {tc.get('name')} with query: {tc.get('args', {}).get('query', 'N/A')}",
                                metadata={"tool_call": True, "tool_name": tc.get('name')}
                            )
                    
                    print(f"\n{'='*60}")
                    print(f"🎬 [DIRECT_ACT] ===== DIRECT ACTION COMPLETE =====")
                    print(f"{'='*60}\n")
                    
                    return {
                        "messages": [tool_response],
                        "total_tool_calls": state.total_tool_calls + (
                            len(tool_response.tool_calls) if tool_response.tool_calls else 0
                        )
                    }
                    
                except Exception as tool_error:
                    print(f"   ❌ Tool execution failed: {str(tool_error)}")
                    print(f"   Falling back to informational response")
        
        # Provide informational response
        print(f"\n💬 [DirectAct] Generating informational response")
        
        info_prompt = f"""Provide a direct, helpful answer to this Unity/game development question: "{user_request}"

Give practical guidance, explanations, or instructions based on your knowledge. Be specific and actionable."""
        
        print(f"   Info prompt length: {len(info_prompt)} chars")
        print(f"   ⏳ Waiting for response...")
        
        info_response = await model.ainvoke([
            {"role": "system", "content": static_system_content},
            {"role": "user", "content": info_prompt}
        ])
        
        response_content = get_message_text(info_response)
        print(f"\n✅ [DirectAct] Response generated:")
        print(f"   Length: {len(response_content)} chars")
        print(f"   Preview: {response_content[:100]}...")
        
        # Record in memory
        if state.memory:
            state.memory.add_message("assistant", response_content)
            print(f"   ✅ Response recorded in memory")
        
        print(f"\n{'='*60}")
        print(f"🎬 [DIRECT_ACT] ===== DIRECT ACTION COMPLETE =====")
        print(f"{'='*60}\n")
        
        return {"messages": [info_response]}
        
    except Exception as e:
        print(f"\n❌ [DirectAct] ERROR during processing:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        error_response = f"I can help you with that. Let me provide some guidance on: {user_request}"
        
        if state.memory:
            state.memory.add_message("assistant", error_response)
        
        print(f"\n{'='*60}")
        print(f"🎬 [DIRECT_ACT] ===== DIRECT ACTION COMPLETE (ERROR) =====")
        print(f"{'='*60}\n")
        
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
        "read_file": [
            "read file", "show me", "display", "view file", "check file",
            "show file", "see file", "open file"
        ],
        "write_file": [
            "create script", "write script", "create file", "generate code",
            "write code", "new script", "make script", "generate file",
            "new file", "make file"
        ],
        "modify_file": [
            "modify file", "update script", "edit file", "change file",
            "fix file", "update file", "patch file", "edit script"
        ],
        "delete_file": [
            "delete file", "remove file", "delete script", "remove script"
        ],
        "move_file": [
            "move file", "rename file", "move script", "rename script"
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