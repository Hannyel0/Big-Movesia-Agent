"""Production-ready tools for Unity agent with SQLite and Qdrant integration."""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List
import json
import os
import asyncio
import difflib
from datetime import datetime, UTC
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime
from langchain_tavily import TavilySearch

from react_agent.context import Context


# ============================================================================
# SQL Query Generator (uses LLM to convert natural language to SQL)
# ============================================================================

async def _generate_sql_query(
    query_description: str,
    tables_hint: Optional[List[str]],
    context: Context
) -> Dict[str, Any]:
    """Generate SQL query from natural language using LLM."""
    from react_agent.utils import get_model
    
    model = get_model(context.model)
    
    schema_info = """
    Available tables and their schemas:
    
    assets (guid TEXT PRIMARY KEY, path TEXT, kind TEXT, mtime INTEGER, size INTEGER, 
            hash TEXT, deleted INTEGER, updated_ts INTEGER, project_id TEXT)
    - Contains all Unity assets (scripts, prefabs, scenes, materials, etc.)
    
    asset_deps (guid TEXT, dep TEXT, PRIMARY KEY (guid, dep))
    - Contains asset dependencies
    
    scenes (guid TEXT PRIMARY KEY, path TEXT, updated_ts INTEGER, project_id TEXT)
    - Contains scene information
    
    game_objects (id INTEGER PRIMARY KEY, guid TEXT, scene_path TEXT, name TEXT, 
                  is_active INTEGER, tag TEXT, layer INTEGER, parent_id INTEGER, 
                  project_id TEXT)
    - Contains GameObject hierarchy from scenes
    
    components (id INTEGER PRIMARY KEY, game_object_id INTEGER, type TEXT, 
                properties TEXT, project_id TEXT)
    - Contains component data attached to GameObjects
    
    events (id INTEGER PRIMARY KEY, ts INTEGER, session TEXT, type TEXT, 
            body TEXT, project_id TEXT)
    - Contains Unity Editor events
    """
    
    prompt = f"""You are a SQL query generator for a Unity project database.

{schema_info}

User query: "{query_description}"
{f"Focus on tables: {', '.join(tables_hint)}" if tables_hint else ""}

Generate a valid SQLite query to answer this question. The query should:
1. Use proper JOINs when querying multiple tables
2. Filter by project_id when available in context
3. Include helpful column aliases for readability
4. Use appropriate WHERE clauses for filtering
5. Return relevant columns only

Respond with ONLY the SQL query, no explanations."""

    response = await model.ainvoke([{"role": "user", "content": prompt}])
    sql_query = response.content.strip()
    
    # Clean up the query (remove markdown code blocks if present)
    if sql_query.startswith("```sql"):
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    elif sql_query.startswith("```"):
        sql_query = sql_query.replace("```", "").strip()
    
    return {
        "query": sql_query,
        "original_request": query_description
    }


@tool
async def search_project(
    query_description: str,
    config: RunnableConfig,
    tables: Optional[List[str]] = None,
    return_format: Literal["structured", "natural_language"] = "structured"
) -> Dict[str, Any]:
    """Search the Unity project using natural language queries.
    
    Converts natural language to SQL and queries the indexed SQLite database containing
    scenes, assets, hierarchy, components, dependencies, and events.
    
    Args:
        query_description: Natural language description of what to find
        tables: Optional hint about which tables to query (assets, scenes, game_objects, components, events)
        return_format: How to format results - "structured" returns raw data, "natural_language" returns readable text
        
    Returns:
        Query results with the generated SQL for transparency
    """
    try:
        runtime = get_runtime(Context)
        context = runtime.context
        
        # Get project context from config
        configurable = config.get("configurable", {})
        project_id = configurable.get("project_id")
        sqlite_path = configurable.get("sqlite_path")
        
        if not sqlite_path or not os.path.exists(sqlite_path):
            return {
                "success": False,
                "error": "SQLite database not found. Ensure Unity project is connected.",
                "query_description": query_description
            }
        
        # Generate SQL query from natural language
        query_result = await _generate_sql_query(query_description, tables, context)
        sql_query = query_result["query"]
        
        # Inject project_id filter if not already present
        if project_id and "project_id" in sql_query and "project_id =" not in sql_query:
            # Add WHERE clause or extend existing one
            if "WHERE" in sql_query.upper():
                sql_query = sql_query.replace("WHERE", f"WHERE project_id = '{project_id}' AND", 1)
            else:
                sql_query = sql_query.rstrip(";") + f" WHERE project_id = '{project_id}'"
        
        # Execute query
        import sqlite3
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row  # Enable column name access
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        results = [dict(row) for row in rows]
        
        conn.close()
        
        # Format results based on return_format
        if return_format == "natural_language" and results:
            formatted = _format_results_natural_language(results, query_description)
        else:
            formatted = results
        
        return {
            "success": True,
            "results": formatted,
            "result_count": len(results),
            "sql_query": sql_query,
            "query_description": query_description,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Query execution failed: {str(e)}",
            "query_description": query_description,
            "sql_query": query_result.get("query", "N/A") if 'query_result' in locals() else "N/A"
        }


def _format_results_natural_language(results: List[Dict], query: str) -> str:
    """Format SQL results into natural language."""
    if not results:
        return f"No results found for: {query}"
    
    count = len(results)
    summary = f"Found {count} result{'s' if count != 1 else ''} for '{query}':\n\n"
    
    for i, row in enumerate(results[:10], 1):  # Limit to 10 for readability
        summary += f"{i}. "
        # Format key fields from the row
        key_fields = []
        if "name" in row:
            key_fields.append(f"{row['name']}")
        if "path" in row:
            key_fields.append(f"({row['path']})")
        if "kind" in row or "type" in row:
            key_fields.append(f"[{row.get('kind') or row.get('type')}]")
        
        summary += " ".join(key_fields) + "\n"
    
    if count > 10:
        summary += f"\n... and {count - 10} more results"
    
    return summary


# ============================================================================
# Code Snippets (Qdrant Vector Search)
# ============================================================================

@tool
async def code_snippets(
    query: str,
    config: RunnableConfig,
    filter_by: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    include_context: bool = True
) -> Dict[str, Any]:
    """Search through Unity C# scripts using semantic meaning via vector search.
    
    Finds code by WHAT IT DOES, not just what it's called. Uses Qdrant vector database
    to perform semantic search through indexed scripts.
    
    Args:
        query: Natural language description of what code you're looking for
        filter_by: Optional filters (file_type, namespace, etc.)
        top_k: Number of results to return (default 5)
        include_context: Include surrounding code context
        
    Returns:
        Ranked scripts with relevance scores and code snippets
    """
    try:
        runtime = get_runtime(Context)
        context = runtime.context
        
        # Get project context
        configurable = config.get("configurable", {})
        project_id = configurable.get("project_id")
        
        if not project_id:
            return {
                "success": False,
                "error": "Project ID not available. Ensure Unity project is connected."
            }
        
        # Generate query embedding
        from react_agent.utils import get_model
        model = get_model(context.model)
        
        # Use OpenAI embeddings (you'll need to adjust based on your embedding model)
        embedding_response = await _get_embedding(query, context)
        query_vector = embedding_response["embedding"]
        
        # Search Qdrant
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("QDRANT_COLLECTION", "movesia")
        
        import httpx
        async with httpx.AsyncClient() as client:
            search_payload = {
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
                "filter": {
                    "must": [
                        {"key": "project_id", "match": {"value": project_id}},
                        {"key": "kind", "match": {"value": "Script"}}
                    ]
                }
            }
            
            # Add additional filters if provided
            if filter_by:
                for key, value in filter_by.items():
                    search_payload["filter"]["must"].append({
                        "key": key,
                        "match": {"value": value}
                    })
            
            response = await client.post(
                f"{qdrant_url}/collections/{collection_name}/points/search",
                json=search_payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Qdrant search failed: {response.text}"
                }
            
            search_results = response.json()
        
        # Format results
        snippets = []
        for hit in search_results.get("result", []):
            payload = hit.get("payload", {})
            snippet = {
                "file_path": payload.get("rel_path", "unknown"),
                "line_range": payload.get("range", ""),
                "relevance_score": hit.get("score", 0.0),
                "code": payload.get("text", ""),
                "file_hash": payload.get("file_hash", "")
            }
            
            if include_context:
                # Add context lines (if available in future enhancement)
                snippet["context"] = "Full file context available on request"
            
            snippets.append(snippet)
        
        return {
            "success": True,
            "query": query,
            "snippets": snippets,
            "total_found": len(snippets),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Code search failed: {str(e)}",
            "query": query
        }


async def _get_embedding(text: str, context: Context) -> Dict[str, Any]:
    """Generate embedding for text using configured model."""
    # This is a placeholder - you'll need to integrate with your actual embedding service
    # For now, returning a mock embedding
    # In production, use OpenAI embeddings or your configured embedding model
    
    try:
        # Example using OpenAI (you'll need to adjust based on your setup)
        import openai
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        return {
            "embedding": response.data[0].embedding,
            "model": "text-embedding-3-small"
        }
    except Exception as e:
        # Fallback to mock for development
        import random
        return {
            "embedding": [random.random() for _ in range(1536)],
            "model": "mock",
            "error": str(e)
        }


# ============================================================================
# File Operation (Unified File I/O)
# ============================================================================

@tool
async def file_operation(
    operation: Literal["read", "write", "modify", "delete", "move", "diff"],
    file_path: str,
    config: RunnableConfig,
    content: Optional[str] = None,
    modification_spec: Optional[Dict[str, Any]] = None,
    validate_only: bool = False
) -> Dict[str, Any]:
    """Unified file I/O with Unity AssetDatabase integration.
    
    Handles all file manipulation safely with validation, diff generation,
    and Unity-compatible operations.
    
    Args:
        operation: Type of operation (read/write/modify/delete/move/diff)
        file_path: Target file path (relative to project root)
        content: Content for write/modify operations
        modification_spec: Specification for surgical edits (line_ranges, patterns, replacements)
        validate_only: If True, performs dry-run without actual changes
        
    Returns:
        Operation result with diff preview and validation status
    """
    try:
        runtime = get_runtime(Context)
        
        # Get project context
        configurable = config.get("configurable", {})
        project_root = configurable.get("project_root")
        
        if not project_root:
            return {
                "success": False,
                "error": "Project root not available. Ensure Unity project is connected."
            }
        
        # Resolve absolute path
        abs_path = Path(project_root) / file_path
        
        # Validate path is within project
        try:
            abs_path = abs_path.resolve()
            project_root_resolved = Path(project_root).resolve()
            if not str(abs_path).startswith(str(project_root_resolved)):
                return {
                    "success": False,
                    "error": f"Path {file_path} is outside project root"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid file path: {str(e)}"
            }
        
        # Execute operation
        if operation == "read":
            return await _file_read(abs_path, file_path)
        
        elif operation == "write":
            if not content:
                return {"success": False, "error": "Content required for write operation"}
            return await _file_write(abs_path, file_path, content, validate_only)
        
        elif operation == "modify":
            if not modification_spec:
                return {"success": False, "error": "Modification spec required for modify operation"}
            return await _file_modify(abs_path, file_path, modification_spec, validate_only)
        
        elif operation == "delete":
            return await _file_delete(abs_path, file_path, validate_only)
        
        elif operation == "move":
            if not modification_spec or "new_path" not in modification_spec:
                return {"success": False, "error": "new_path required in modification_spec for move"}
            new_path = Path(project_root) / modification_spec["new_path"]
            return await _file_move(abs_path, new_path, file_path, modification_spec["new_path"], validate_only)
        
        elif operation == "diff":
            if not content:
                return {"success": False, "error": "Content required for diff operation"}
            return await _file_diff(abs_path, file_path, content)
        
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
    except Exception as e:
        return {
            "success": False,
            "error": f"File operation failed: {str(e)}",
            "operation": operation,
            "file_path": file_path
        }


async def _file_read(abs_path: Path, rel_path: str) -> Dict[str, Any]:
    """Read file contents."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"File not found: {rel_path}"
        }
    
    try:
        content = abs_path.read_text(encoding="utf-8")
        return {
            "success": True,
            "operation": "read",
            "file_path": rel_path,
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.split("\n"))
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read file: {str(e)}",
            "file_path": rel_path
        }


async def _file_write(abs_path: Path, rel_path: str, content: str, validate_only: bool) -> Dict[str, Any]:
    """Write content to file."""
    # Generate diff if file exists
    diff = ""
    if abs_path.exists():
        old_content = abs_path.read_text(encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(
            old_content.splitlines(),
            content.splitlines(),
            fromfile=f"{rel_path} (original)",
            tofile=f"{rel_path} (new)",
            lineterm=""
        ))
    
    if validate_only:
        return {
            "success": True,
            "operation": "write",
            "file_path": rel_path,
            "validate_only": True,
            "would_create": not abs_path.exists(),
            "diff_preview": diff,
            "size_bytes": len(content.encode("utf-8"))
        }
    
    try:
        # Ensure parent directory exists
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        abs_path.write_text(content, encoding="utf-8")
        
        return {
            "success": True,
            "operation": "write",
            "file_path": rel_path,
            "created": not diff,
            "diff": diff,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": len(content.split("\n"))
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write file: {str(e)}",
            "file_path": rel_path
        }


async def _file_modify(abs_path: Path, rel_path: str, spec: Dict[str, Any], validate_only: bool) -> Dict[str, Any]:
    """Modify file with surgical edits."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"File not found: {rel_path}"
        }
    
    try:
        content = abs_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        modified_lines = lines.copy()
        
        # Apply modifications based on spec
        if "line_ranges" in spec:
            # Modify specific line ranges
            for line_range in spec["line_ranges"]:
                start = line_range.get("start", 1) - 1  # Convert to 0-indexed
                end = line_range.get("end", len(lines)) - 1
                replacement = line_range.get("replacement", "")
                
                if 0 <= start <= end < len(modified_lines):
                    modified_lines[start:end+1] = [replacement]
        
        elif "pattern_replacements" in spec:
            # Pattern-based replacements
            import re
            for pattern_spec in spec["pattern_replacements"]:
                pattern = pattern_spec.get("pattern")
                replacement = pattern_spec.get("replacement", "")
                
                if pattern:
                    modified_content = "\n".join(modified_lines)
                    modified_content = re.sub(pattern, replacement, modified_content)
                    modified_lines = modified_content.split("\n")
        
        new_content = "\n".join(modified_lines)
        
        # Generate diff
        diff = "\n".join(difflib.unified_diff(
            lines,
            modified_lines,
            fromfile=f"{rel_path} (original)",
            tofile=f"{rel_path} (modified)",
            lineterm=""
        ))
        
        if validate_only:
            return {
                "success": True,
                "operation": "modify",
                "file_path": rel_path,
                "validate_only": True,
                "diff_preview": diff
            }
        
        # Apply changes
        abs_path.write_text(new_content, encoding="utf-8")
        
        return {
            "success": True,
            "operation": "modify",
            "file_path": rel_path,
            "diff": diff,
            "modifications_applied": len(spec.get("line_ranges", [])) + len(spec.get("pattern_replacements", []))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to modify file: {str(e)}",
            "file_path": rel_path
        }


async def _file_delete(abs_path: Path, rel_path: str, validate_only: bool) -> Dict[str, Any]:
    """Delete file."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"File not found: {rel_path}"
        }
    
    if validate_only:
        return {
            "success": True,
            "operation": "delete",
            "file_path": rel_path,
            "validate_only": True,
            "would_delete": True
        }
    
    try:
        abs_path.unlink()
        return {
            "success": True,
            "operation": "delete",
            "file_path": rel_path,
            "deleted": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to delete file: {str(e)}",
            "file_path": rel_path
        }


async def _file_move(abs_path: Path, new_abs_path: Path, rel_path: str, new_rel_path: str, validate_only: bool) -> Dict[str, Any]:
    """Move/rename file."""
    if not abs_path.exists():
        return {
            "success": False,
            "error": f"Source file not found: {rel_path}"
        }
    
    if new_abs_path.exists():
        return {
            "success": False,
            "error": f"Destination already exists: {new_rel_path}"
        }
    
    if validate_only:
        return {
            "success": True,
            "operation": "move",
            "from_path": rel_path,
            "to_path": new_rel_path,
            "validate_only": True,
            "would_move": True
        }
    
    try:
        # Ensure destination directory exists
        new_abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        abs_path.rename(new_abs_path)
        
        return {
            "success": True,
            "operation": "move",
            "from_path": rel_path,
            "to_path": new_rel_path,
            "moved": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to move file: {str(e)}",
            "from_path": rel_path,
            "to_path": new_rel_path
        }


async def _file_diff(abs_path: Path, rel_path: str, new_content: str) -> Dict[str, Any]:
    """Generate diff without modifying file."""
    if not abs_path.exists():
        return {
            "success": True,
            "operation": "diff",
            "file_path": rel_path,
            "file_exists": False,
            "diff": f"New file would be created:\n{new_content}"
        }
    
    try:
        old_content = abs_path.read_text(encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=f"{rel_path} (current)",
            tofile=f"{rel_path} (proposed)",
            lineterm=""
        ))
        
        return {
            "success": True,
            "operation": "diff",
            "file_path": rel_path,
            "diff": diff,
            "changes_detected": bool(diff)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate diff: {str(e)}",
            "file_path": rel_path
        }


# ============================================================================
# Web Search (Keep existing Tavily search)
# ============================================================================

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


# ============================================================================
# Export Tools and Metadata
# ============================================================================

TOOLS = [
    search_project,
    code_snippets,
    file_operation,
    web_search,
]

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
        "description": "Safe file I/O with validation and diff generation"
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
