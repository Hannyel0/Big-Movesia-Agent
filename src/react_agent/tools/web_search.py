"""Web search tool using Tavily."""

from __future__ import annotations
from typing import Dict, Any
import os
from datetime import datetime, UTC

from langchain_core.tools import tool
from langchain_tavily import TavilySearch


@tool
async def web_search(query: str) -> Dict[str, Any]:
    """Search the web for general information about game development, Unity, and related topics.
    
    Args:
        query: Search query string
        
    Returns:
        Search results with URLs, titles, and content
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured"
            }
        
        search = TavilySearch(api_key=tavily_api_key)
        results = await search.ainvoke({"query": query})
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "result_count": len(results),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Web search failed: {str(e)}",
            "query": query
        }
