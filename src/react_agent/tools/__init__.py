"""Production-ready tools for Unity agent with SQLite and Qdrant integration."""

from typing import List
from react_agent.tools.search_project import search_project, get_cache_stats, clear_query_cache
from react_agent.tools.code_snippets import code_snippets
from react_agent.tools.file_operation import file_operation
from react_agent.tools.web_search import web_search

# Export all tools
TOOLS = [search_project, code_snippets, file_operation, web_search]

# Tool metadata for planning and optimization
TOOL_METADATA = {
    "search_project": {
        "category": "data_query",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["finding assets", "querying hierarchy", "searching components", "checking dependencies"],
        "description": "Natural language queries against indexed Unity project data"
    },
    "code_snippets": {
        "category": "code_search",
        "cost": "medium",
        "reliability": "high",
        "best_for": ["finding code by functionality", "semantic code search", "discovering implementations"],
        "description": "Semantic search through C# scripts using vector embeddings"
    },
    "file_operation": {
        "category": "file_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["reading files", "writing scripts", "modifying code", "file manipulation"],
        "description": "Safe file I/O with validation and approval flow"
    },
    "web_search": {
        "category": "information_retrieval",
        "cost": "medium",
        "reliability": "high",
        "best_for": ["Unity documentation", "tutorials", "best practices", "troubleshooting"],
        "description": "Web search for Unity and game development information"
    }
}


def get_available_tool_names() -> List[str]:
    """Get list of available tool names for validation."""
    return [tool.name for tool in TOOLS]


__all__ = [
    "search_project",
    "code_snippets", 
    "file_operation",
    "web_search",
    "TOOLS",
    "TOOL_METADATA",
    "get_available_tool_names",
    "get_cache_stats",
    "clear_query_cache"
]
