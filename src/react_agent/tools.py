from __future__ import annotations
from typing import Literal
from react_agent.state import State

import asyncio
from typing import Any, Callable, List, Optional, Dict, cast
from functools import wraps
import time
import json
from datetime import datetime, UTC

from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langgraph.runtime import get_runtime

from react_agent.context import Context


# Tool execution wrapper for enhanced error handling and logging
def tool_wrapper(func: Callable) -> Callable:
    """Wrapper to add error handling, logging, and metadata to tools."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        tool_name = func.__name__
        
        try:
            # Execute the tool
            result = await func(*args, **kwargs)
            
            # Wrap result with metadata
            return {
                "success": True,
                "tool": tool_name,
                "result": result,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now(UTC).isoformat(),
                "error": None
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "tool": tool_name,
                "result": None,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now(UTC).isoformat(),
                "error": f"Tool {tool_name} timed out after {time.time() - start_time:.2f}s"
            }
            
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "result": None,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now(UTC).isoformat(),
                "error": f"Tool {tool_name} failed: {str(e)}"
            }
    
    # Preserve the original function's metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    
    return wrapper


@tool_wrapper
async def search(query: str) -> Optional[Dict[str, Any]]:
    """Search for general web results.
    
    This function performs a search using the Tavily search engine, which provides
    comprehensive, accurate, and trusted results. It's particularly useful for
    answering questions about current events and gathering factual information.
    
    Args:
        query: The search query string
        
    Returns:
        Dictionary containing search results with metadata
    """
    runtime = get_runtime(Context)
    
    # Add timeout handling
    timeout = runtime.context.tool_timeout_seconds
    
    try:
        wrapped = TavilySearch(max_results=runtime.context.max_search_results)
        
        # Execute with timeout
        result = await asyncio.wait_for(
            wrapped.ainvoke({"query": query}),
            timeout=timeout
        )
        
        return cast(Dict[str, Any], result)
        
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError(f"Search timed out after {timeout} seconds")


@tool_wrapper
async def calculate(expression: str) -> Dict[str, Any]:
    """Perform mathematical calculations.
    
    Evaluates mathematical expressions safely using a restricted evaluation environment.
    Supports basic arithmetic, powers, and common mathematical functions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")
        
    Returns:
        Dictionary containing the calculation result
    """
    import math
    import operator
    
    # Safe evaluation environment
    safe_dict = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'pi': math.pi,
        'e': math.e,
    }
    
    try:
        # Remove any potentially dangerous characters
        if any(char in expression for char in ['import', '__', 'exec', 'eval', 'compile']):
            raise ValueError("Expression contains forbidden operations")
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }
        
    except Exception as e:
        raise ValueError(f"Calculation failed: {str(e)}")


@tool_wrapper
async def get_current_time(timezone: Optional[str] = None) -> Dict[str, Any]:
    """Get the current date and time.
    
    Args:
        timezone: Optional timezone name (e.g., "America/New_York", "Europe/London").
                 If not provided, returns UTC time.
    
    Returns:
        Dictionary containing current time information
    """
    from zoneinfo import ZoneInfo
    
    try:
        if timezone:
            tz = ZoneInfo(timezone)
            current_time = datetime.now(tz)
        else:
            current_time = datetime.now(UTC)
        
        return {
            "iso_format": current_time.isoformat(),
            "formatted": current_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "timestamp": current_time.timestamp(),
            "timezone": timezone or "UTC",
            "date": current_time.date().isoformat(),
            "time": current_time.time().isoformat(),
            "weekday": current_time.strftime("%A"),
        }
        
    except Exception as e:
        raise ValueError(f"Failed to get time for timezone {timezone}: {str(e)}")


@tool_wrapper
async def store_memory(key: str, value: Any) -> Dict[str, Any]:
    """Store information in temporary memory for use across steps.
    
    This tool allows the agent to remember information between steps
    within the same execution session. Data is not persistent across sessions.
    
    Args:
        key: Unique identifier for the stored information
        value: The information to store (will be JSON-serialized)
    
    Returns:
        Confirmation of storage with metadata
    """
    runtime = get_runtime(Context)
    
    # Initialize memory storage if not exists
    if "agent_memory" not in runtime.context.runtime_metadata:
        runtime.context.runtime_metadata["agent_memory"] = {}
    
    # Store the value
    try:
        # Ensure value is JSON-serializable
        json_value = json.dumps(value)
        runtime.context.runtime_metadata["agent_memory"][key] = json.loads(json_value)
        
        return {
            "key": key,
            "stored": True,
            "size": len(json_value),
            "total_keys": len(runtime.context.runtime_metadata["agent_memory"])
        }
        
    except Exception as e:
        raise ValueError(f"Failed to store value: {str(e)}")


@tool_wrapper
async def retrieve_memory(key: str) -> Dict[str, Any]:
    """Retrieve information from temporary memory.
    
    Args:
        key: The identifier of the information to retrieve
    
    Returns:
        Dictionary containing the retrieved value and metadata
    """
    runtime = get_runtime(Context)
    
    # Check if memory exists
    if "agent_memory" not in runtime.context.runtime_metadata:
        runtime.context.runtime_metadata["agent_memory"] = {}
    
    memory = runtime.context.runtime_metadata["agent_memory"]
    
    if key in memory:
        return {
            "key": key,
            "value": memory[key],
            "found": True
        }
    else:
        return {
            "key": key,
            "value": None,
            "found": False,
            "available_keys": list(memory.keys())
        }


@tool_wrapper
async def validate_json(json_string: str) -> Dict[str, Any]:
    """Validate and parse JSON string.
    
    Useful for checking if data is valid JSON and extracting structured information.
    
    Args:
        json_string: String that should contain valid JSON
    
    Returns:
        Dictionary containing parsed JSON and validation result
    """
    try:
        parsed = json.loads(json_string)
        return {
            "valid": True,
            "parsed": parsed,
            "type": type(parsed).__name__,
            "size": len(json_string)
        }
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "error": str(e),
            "line": e.lineno,
            "column": e.colno
        }


@tool_wrapper
async def text_analysis(text: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """Analyze text for various properties.
    
    Args:
        text: The text to analyze
        analysis_type: Type of analysis - "summary", "sentiment", "keywords", "stats"
    
    Returns:
        Dictionary containing analysis results
    """
    import re
    from collections import Counter
    
    result = {
        "text_length": len(text),
        "analysis_type": analysis_type
    }
    
    if analysis_type == "stats":
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        result.update({
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "unique_words": len(set(words))
        })
        
    elif analysis_type == "keywords":
        # Simple keyword extraction (most common words)
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
        words = [w for w in words if w not in stop_words and len(w) > 3]
        word_freq = Counter(words)
        
        result["keywords"] = dict(word_freq.most_common(10))
        
    elif analysis_type == "summary":
        # Very basic summary (first and last sentences)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            result["first_sentence"] = sentences[0]
            result["last_sentence"] = sentences[-1] if len(sentences) > 1 else sentences[0]
            result["sentence_count"] = len(sentences)
            
    return result


# Compile all tools into a list
TOOLS: List[Callable[..., Any]] = [
    search,
    calculate,
    get_current_time,
    store_memory,
    retrieve_memory,
    validate_json,
    text_analysis,
]


# Tool metadata for better tool selection
TOOL_METADATA = {
    "search": {
        "category": "information_retrieval",
        "cost": "medium",
        "reliability": "high",
        "best_for": ["current events", "factual information", "web content"]
    },
    "calculate": {
        "category": "computation",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["math", "numerical analysis", "calculations"]
    },
    "get_current_time": {
        "category": "utility",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["time queries", "scheduling", "timezone conversion"]
    },
    "store_memory": {
        "category": "state_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["cross-step data", "temporary storage", "context preservation"]
    },
    "retrieve_memory": {
        "category": "state_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["accessing stored data", "context retrieval"]
    },
    "validate_json": {
        "category": "validation",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["data validation", "structure checking", "parsing"]
    },
    "text_analysis": {
        "category": "analysis",
        "cost": "low",
        "reliability": "high",
        "best_for": ["text statistics", "keyword extraction", "summarization"]
    }
}