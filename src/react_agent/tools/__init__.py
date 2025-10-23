"""Production-ready tools for Unity agent with SQLite and Qdrant integration."""

from typing import List
from react_agent.tools.search_project import search_project, get_cache_stats, clear_query_cache
from react_agent.tools.code_snippets import code_snippets
from react_agent.tools.unity_docs import unity_docs
from react_agent.tools.file_operation import read_file, write_file, modify_file, delete_file, move_file
from react_agent.tools.web_search import web_search

# Export all tools (now 9 tools total: 1 search + 1 code + 1 unity_docs + 5 file + 1 web)
TOOLS = [search_project, code_snippets, unity_docs, read_file, write_file, modify_file, delete_file, move_file, web_search]

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
    "unity_docs": {
        "category": "documentation_search",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["Unity API reference", "feature documentation", "scripting examples", "Unity concepts"],
        "description": "Semantic search through local Unity documentation with RAG"
    },
    "read_file": {
        "category": "file_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["reading file contents", "viewing scripts", "inspecting code"],
        "description": "Read file contents from Unity project with smart path resolution"
    },
    "write_file": {
        "category": "file_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["creating new files", "writing scripts", "generating code"],
        "description": "Write or create files in Unity project (requires approval)"
    },
    "modify_file": {
        "category": "file_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["editing existing files", "updating scripts", "surgical code changes"],
        "description": "Modify existing files in Unity project (requires approval)"
    },
    "delete_file": {
        "category": "file_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["removing files", "cleaning up scripts", "deleting assets"],
        "description": "Delete files from Unity project (requires approval)"
    },
    "move_file": {
        "category": "file_management",
        "cost": "low",
        "reliability": "very_high",
        "best_for": ["moving files", "renaming scripts", "reorganizing project"],
        "description": "Move or rename files in Unity project (requires approval)"
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
    "unity_docs",
    "read_file",
    "write_file",
    "modify_file",
    "delete_file",
    "move_file",
    "web_search",
    "TOOLS",
    "TOOL_METADATA",
    "get_available_tool_names",
    "get_cache_stats",
    "clear_query_cache"
]
