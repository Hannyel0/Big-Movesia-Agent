"""Enhanced search_project tool with caching, debugging, and optimizations."""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List
import asyncio
import json
import os
import hashlib
import logging
import sqlite3
from datetime import datetime, UTC
from functools import lru_cache

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

from react_agent.context import Context

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - [SEARCH_PROJECT] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Query cache: stores (query_hash) -> (sql_query, timestamp)
_query_cache: Dict[str, Dict[str, Any]] = {}
_cache_hits = 0
_cache_misses = 0
CACHE_MAX_AGE = 3600  # 1 hour in seconds


def _get_query_hash(query_description: str, tables_hint: Optional[List[str]]) -> str:
    """Generate a hash for caching query translations."""
    cache_key = f"{query_description.lower().strip()}:{','.join(sorted(tables_hint or []))}"
    return hashlib.md5(cache_key.encode()).hexdigest()


def _get_cached_query(query_hash: str) -> Optional[str]:
    """Get cached SQL query if available and not expired."""
    global _cache_hits, _cache_misses
    
    if query_hash in _query_cache:
        cached = _query_cache[query_hash]
        age = datetime.now(UTC).timestamp() - cached['timestamp']
        
        if age < CACHE_MAX_AGE:
            _cache_hits += 1
            logger.debug(f"✅ Cache HIT for query hash {query_hash[:8]}... (age: {age:.1f}s)")
            logger.debug(f"📊 Cache stats: {_cache_hits} hits, {_cache_misses} misses, {len(_query_cache)} entries")
            return cached['sql']
        else:
            logger.debug(f"⏰ Cache EXPIRED for query hash {query_hash[:8]}... (age: {age:.1f}s)")
            del _query_cache[query_hash]
    
    _cache_misses += 1
    logger.debug(f"❌ Cache MISS for query hash {query_hash[:8]}...")
    logger.debug(f"📊 Cache stats: {_cache_hits} hits, {_cache_misses} misses, {len(_query_cache)} entries")
    return None


def _cache_query(query_hash: str, sql_query: str) -> None:
    """Cache a generated SQL query."""
    _query_cache[query_hash] = {
        'sql': sql_query,
        'timestamp': datetime.now(UTC).timestamp()
    }
    logger.debug(f"💾 Cached query {query_hash[:8]}... (total cached: {len(_query_cache)})")
    
    # Limit cache size
    if len(_query_cache) > 100:
        oldest_key = min(_query_cache.keys(), key=lambda k: _query_cache[k]['timestamp'])
        del _query_cache[oldest_key]
        logger.debug(f"🗑️ Evicted oldest cache entry (keeping cache at 100 entries)")


def _validate_sql_query(sql_query: str) -> Dict[str, Any]:
    """Validate SQL query for safety and correctness."""
    sql_lower = sql_query.lower().strip()
    
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check for dangerous operations
    dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'create', 'truncate']
    for keyword in dangerous_keywords:
        if f' {keyword} ' in f' {sql_lower} ':
            validation['is_valid'] = False
            validation['errors'].append(f"Dangerous SQL operation detected: {keyword.upper()}")
            logger.error(f"🚫 SECURITY: Blocked dangerous SQL operation: {keyword.upper()}")
    
    # Check for SELECT statement
    if not sql_lower.startswith('select'):
        validation['is_valid'] = False
        validation['errors'].append("Only SELECT queries are allowed")
        logger.error(f"🚫 SECURITY: Non-SELECT query blocked")
    
    # Warn about missing LIMIT
    if 'limit' not in sql_lower:
        validation['warnings'].append("Query has no LIMIT clause - may return many rows")
        logger.warning("⚠️ Query has no LIMIT - consider adding one for performance")
    
    # Check for valid table names
    valid_tables = ['assets', 'asset_deps', 'scenes', 'hierarchy_scenes', 
                'hierarchy_gameobjects', 'hierarchy_components', 'events']
    found_tables = []
    for table in valid_tables:
        if table in sql_lower:
            found_tables.append(table)
    
    if not found_tables:
        validation['warnings'].append("No recognized table names found in query")
        logger.warning(f"⚠️ No recognized tables found. Valid tables: {valid_tables}")
    else:
        logger.debug(f"✅ Query references tables: {found_tables}")
    
    if validation['is_valid']:
        logger.debug(f"✅ SQL validation passed{' with warnings' if validation['warnings'] else ''}")
    else:
        logger.error(f"❌ SQL validation failed: {validation['errors']}")
    
    return validation


async def _generate_sql_query(
    query_description: str,
    tables_hint: Optional[List[str]],
    context: Context
) -> Dict[str, Any]:
    """Generate SQL query from natural language using LLM with enhanced schema info."""
    from react_agent.utils import get_model
    
    logger.info(f"🔄 Generating SQL for: '{query_description}'")
    if tables_hint:
        logger.debug(f"📋 Table hints provided: {tables_hint}")
    
    model = get_model(context.model)
    
    # Enhanced schema information with examples
    schema_info = """
    DATABASE SCHEMA (SQLite):
    
    📦 assets - All Unity/Unreal assets (scripts, prefabs, scenes, materials, etc.)
    Columns:
      - guid TEXT PRIMARY KEY (unique asset identifier)
      - path TEXT (relative path like "Assets/Scripts/Player.cs")
      - kind TEXT (asset type: "MonoScript", "Prefab", "Scene", "Material", etc.)
      - mtime INTEGER (last modified time, Unix timestamp)
      - size INTEGER (file size in bytes)
      - hash TEXT (SHA256 hash of file content)
      - deleted INTEGER (0=active, 1=deleted)
      - updated_ts INTEGER (last database update timestamp)
      - project_id TEXT (project identifier)
    
    🔗 asset_deps - Asset dependency relationships
    Columns:
      - guid TEXT (asset GUID)
      - dep TEXT (dependency GUID)
      PRIMARY KEY (guid, dep)
    
    🎬 scenes - Scene files
    Columns:
      - guid TEXT PRIMARY KEY
      - path TEXT (scene file path)
      - updated_ts INTEGER
      - project_id TEXT
    
    🎬 hierarchy_scenes - Scene metadata with snapshot tracking
    Columns:
      - id INTEGER PRIMARY KEY
      - project_id TEXT
      - scene_path TEXT
      - scene_guid TEXT
      - last_updated INTEGER
      - snapshot_hash TEXT
    
    🎮 hierarchy_gameobjects - GameObject hierarchy from parsed scenes
    Columns:
      - id INTEGER PRIMARY KEY
      - project_id TEXT
      - scene_path TEXT
      - instance_id INTEGER (Unity instance ID)
      - name TEXT (GameObject name)
      - hierarchy_path TEXT (full path in hierarchy)
      - parent_path TEXT (parent's hierarchy path, NULL for root)
      - tag TEXT (Unity tag like "Player", "Enemy")
      - layer INTEGER (Unity layer number)
      - active_self INTEGER (0 or 1)
      - active_in_hierarchy INTEGER (0 or 1)
      - is_static INTEGER (0 or 1)
      - pos_x, pos_y, pos_z REAL (position)
      - rot_x, rot_y, rot_z, rot_w REAL (rotation quaternion)
      - scale_x, scale_y, scale_z REAL (scale)
      - sibling_index INTEGER
      - last_updated INTEGER
      - is_deleted INTEGER (0=active, 1=deleted)
    
    🧩 hierarchy_components - Components attached to GameObjects
    Columns:
      - id INTEGER PRIMARY KEY
      - project_id TEXT
      - scene_path TEXT
      - gameobject_instance_id INTEGER (references hierarchy_gameobjects.instance_id)
      - type_name TEXT (short name like "Transform", "Rigidbody")
      - full_type_name TEXT (fully qualified type name)
      - assembly_name TEXT (assembly containing the component)
      - enabled INTEGER (0 or 1)
      - properties_json TEXT (JSON string of component properties)
      - last_updated INTEGER
      - is_deleted INTEGER (0=active, 1=deleted)
    
    📝 events - Unity Editor events log
    Columns:
      - id INTEGER PRIMARY KEY
      - ts INTEGER (timestamp)
      - session TEXT (session ID)
      - type TEXT (event type)
      - body TEXT (JSON event data)
      - project_id TEXT
    
    QUERY EXAMPLES:
    
    1. "Find all player scripts"
       SELECT path, kind, size FROM assets 
       WHERE kind = 'MonoScript' AND path LIKE '%Player%' 
       AND deleted = 0 AND project_id = ?
       LIMIT 50;
    
    2. "Show me GameObjects tagged as Player"
       SELECT name, tag, active_self, scene_path FROM hierarchy_gameobjects 
       WHERE tag = 'Player' AND is_deleted = 0 AND project_id = ?
       LIMIT 50;
    
    3. "What assets depend on CharacterController?"
       SELECT DISTINCT a.path, a.kind 
       FROM assets a
       JOIN asset_deps ad ON a.guid = ad.guid
       JOIN assets dep ON ad.dep = dep.guid
       WHERE dep.path LIKE '%CharacterController%' 
       AND a.deleted = 0 AND a.project_id = ?
       LIMIT 50;
    
    4. "Find all prefabs in the Player folder"
       SELECT path, size, mtime FROM assets 
       WHERE kind = 'Prefab' AND path LIKE '%/Player/%' 
       AND deleted = 0 AND project_id = ?
       ORDER BY mtime DESC
       LIMIT 50;
    
    5. "Show me all components on GameObjects named 'Player'"
       SELECT hgo.name, hgo.scene_path, hc.type_name, hc.enabled 
       FROM hierarchy_gameobjects hgo
       JOIN hierarchy_components hc ON hgo.instance_id = hc.gameobject_instance_id 
           AND hgo.project_id = hc.project_id AND hgo.scene_path = hc.scene_path
       WHERE hgo.name = 'Player' AND hgo.is_deleted = 0 AND hc.is_deleted = 0
       AND hgo.project_id = ?
       LIMIT 50;
    
    6. "Find GameObjects with Rigidbody components"
       SELECT DISTINCT hgo.name, hgo.hierarchy_path, hgo.scene_path
       FROM hierarchy_gameobjects hgo
       JOIN hierarchy_components hc ON hgo.instance_id = hc.gameobject_instance_id
           AND hgo.project_id = hc.project_id AND hgo.scene_path = hc.scene_path
       WHERE hc.type_name = 'Rigidbody' AND hgo.is_deleted = 0 AND hc.is_deleted = 0
       AND hgo.project_id = ?
       LIMIT 50;
    
    7. "List all component types used in the project"
       SELECT type_name, COUNT(*) as count
       FROM hierarchy_components
       WHERE is_deleted = 0 AND project_id = ?
       GROUP BY type_name
       ORDER BY count DESC
       LIMIT 50;
    
    QUERY BEST PRACTICES:
    - Always filter is_deleted = 0 for hierarchy tables
    - Always filter deleted = 0 for assets table
    - Use LIKE with % wildcards for partial matches
    - Include project_id filter when available (use ? placeholder)
    - Add LIMIT to prevent excessive results (default: 50)
    - When joining hierarchy_gameobjects and hierarchy_components, match on:
      instance_id = gameobject_instance_id AND project_id AND scene_path
    - Use ORDER BY for sorted results (DESC for newest first)
    - Use DISTINCT to avoid duplicates when joining
    """
    
    prompt = f"""You are a SQL query generator for a Unity/Unreal project database.

{schema_info}

User query: "{query_description}"
{f"Focus on these tables: {', '.join(tables_hint)}" if tables_hint else ""}

Generate a valid SQLite SELECT query to answer this question. Requirements:
1. Use proper JOINs when querying multiple tables
2. Include project_id filter using ? placeholder (e.g., WHERE project_id = ?)
3. Filter deleted = 0 for assets table
4. Use helpful column aliases for readability
5. Add LIMIT clause (default 50, adjust based on query type)
6. Use ORDER BY for sorted results when appropriate
7. Return only relevant columns, not SELECT *

Respond with ONLY the SQL query, no explanations or markdown."""

    logger.debug(f"📤 Sending query generation request to LLM")
    
    try:
        response = await model.ainvoke([{"role": "user", "content": prompt}])
        sql_query = response.content.strip()
        
        logger.debug(f"📥 Received LLM response ({len(sql_query)} chars)")
        
        # Clean up the query (remove markdown code blocks if present)
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            logger.debug("🧹 Cleaned SQL markdown formatting")
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
            logger.debug("🧹 Cleaned generic markdown formatting")
        
        logger.info(f"✅ Generated SQL query ({len(sql_query)} chars)")
        logger.debug(f"📝 SQL: {sql_query}")
        
        return {
            "query": sql_query,
            "original_request": query_description
        }
    
    except Exception as e:
        logger.error(f"❌ SQL generation failed: {str(e)}", exc_info=True)
        raise


def _execute_query_sync(sqlite_path: str, sql_query: str) -> tuple[list[dict], float]:
    """Execute SQLite query synchronously (used in thread pool).
    
    This function is intentionally synchronous and will be called via asyncio.to_thread().
    Returns (results, execution_time).
    """
    import time
    start = time.time()
    
    try:
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        results = [dict(row) for row in rows]
        
        conn.close()
        
        duration = time.time() - start
        return (results, duration)
        
    except Exception as e:
        raise  # Re-raise to be caught by async wrapper


@tool
async def search_project(
    query_description: str,
    config: RunnableConfig,
    tables: Optional[List[str]] = None,
    return_format: Literal["structured", "natural_language"] = "structured"
) -> Dict[str, Any]:
    """Search the Unity/Unreal project using natural language queries.
    
    Converts natural language to SQL and queries the indexed SQLite database containing
    scenes, assets, hierarchy, components, dependencies, and events.
    
    Args:
        query_description: Natural language description of what to find
        tables: Optional hint about which tables to query (assets, scenes, hierarchy_gameobjects, hierarchy_components, events)
        return_format: How to format results - "structured" returns raw data, "natural_language" returns readable text
        
    Returns:
        Query results with the generated SQL for transparency
    """
    start_time = datetime.now(UTC)
    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 SEARCH_PROJECT TOOL INVOKED")
    logger.info(f"📝 Query: '{query_description}'")
    logger.info(f"📊 Format: {return_format}")
    logger.info(f"{'='*70}")
    
    try:
        runtime = get_runtime(Context)
        context = runtime.context
        
        # Get project context from config
        configurable = config.get("configurable", {})
        project_id = configurable.get("project_id")
        sqlite_path = configurable.get("sqlite_path")
        
        logger.debug(f"🔧 Config - project_id: {project_id}")
        logger.debug(f"🔧 Config - sqlite_path: {sqlite_path}")
        
        if not sqlite_path:
            logger.error("❌ SQLite path not provided in config")
            return {
                "success": False,
                "error": "SQLite database path not configured",
                "query_description": query_description
            }
        
        if not os.path.exists(sqlite_path):
            logger.error(f"❌ SQLite database not found at: {sqlite_path}")
            return {
                "success": False,
                "error": f"SQLite database not found at path: {sqlite_path}",
                "query_description": query_description
            }
        
        logger.info(f"✅ Database file exists: {sqlite_path}")
        
        # Check cache first
        query_hash = _get_query_hash(query_description, tables)
        cached_sql = _get_cached_query(query_hash)
        
        if cached_sql:
            sql_query = cached_sql
            logger.info("⚡ Using CACHED SQL query")
        else:
            # Generate SQL query from natural language
            logger.info("🤖 Generating NEW SQL query via LLM")
            query_result = await _generate_sql_query(query_description, tables, context)
            sql_query = query_result["query"]
            
            # Validate the generated query
            validation = _validate_sql_query(sql_query)
            
            if not validation['is_valid']:
                logger.error(f"❌ Generated query failed validation")
                return {
                    "success": False,
                    "error": f"Invalid SQL query generated: {'; '.join(validation['errors'])}",
                    "query_description": query_description,
                    "sql_query": sql_query
                }
            
            # Cache the validated query
            _cache_query(query_hash, sql_query)
        
        # Inject project_id filter if project_id is available
        if project_id:
            # ✅ FIXED: Simply replace ? placeholders if they exist
            if "?" in sql_query:
                placeholder_count = sql_query.count("?")
                sql_query = sql_query.replace("?", f"'{project_id}'")
                logger.debug(f"✏️ Replaced {placeholder_count} placeholder(s) with project_id")
            # If no placeholders but query mentions project_id, try to add filter
            elif "project_id" in sql_query.lower() and "project_id =" not in sql_query.lower():
                # Add WHERE clause or extend existing one
                if "WHERE" in sql_query.upper():
                    # Find the WHERE clause and add project_id filter
                    where_pos = sql_query.upper().find("WHERE")
                    before_where = sql_query[:where_pos + 5]
                    after_where = sql_query[where_pos + 5:]
                    sql_query = f"{before_where} project_id = '{project_id}' AND {after_where}"
                    logger.debug(f"✏️ Added project_id to existing WHERE clause")
                else:
                    # Add new WHERE clause before ORDER BY or LIMIT
                    if "ORDER BY" in sql_query.upper():
                        order_pos = sql_query.upper().find("ORDER BY")
                        sql_query = f"{sql_query[:order_pos]} WHERE project_id = '{project_id}' {sql_query[order_pos:]}"
                    elif "LIMIT" in sql_query.upper():
                        limit_pos = sql_query.upper().find("LIMIT")
                        sql_query = f"{sql_query[:limit_pos]} WHERE project_id = '{project_id}' {sql_query[limit_pos:]}"
                    else:
                        sql_query = sql_query.rstrip(";") + f" WHERE project_id = '{project_id}'"
                    logger.debug(f"✏️ Added new WHERE clause with project_id")
        
        logger.info(f"🎯 Final SQL to execute:")
        logger.info(f"   {sql_query}")
        
        # ✅ FIX: Execute query in thread pool to avoid blocking
        logger.debug(f"⚡ Executing query in thread pool...")
        
        try:
            # Run blocking SQLite operations in a thread pool
            results, query_duration = await asyncio.to_thread(
                _execute_query_sync,
                sqlite_path,
                sql_query
            )
            
            logger.info(f"✅ Query executed in {query_duration:.3f}s")
            logger.info(f"📊 Retrieved {len(results)} rows")
            
            # Log sample of results for debugging
            if results:
                logger.debug(f"📋 First result sample: {json.dumps(results[0], indent=2, default=str)}")
            else:
                logger.debug(f"📋 No results returned")
            
            logger.debug("🔒 Database connection closed")
            
        except sqlite3.Error as sql_err:
            logger.error(f"❌ SQLite error during execution: {str(sql_err)}", exc_info=True)
            return {
                "success": False,
                "error": f"SQL execution failed: {str(sql_err)}",
                "query_description": query_description,
                "sql_query": sql_query
            }
        
        # Format results based on return_format
        if return_format == "natural_language" and results:
            logger.debug("📝 Formatting results as natural language")
            formatted = _format_results_natural_language(results, query_description)
        else:
            formatted = results
        
        total_duration = (datetime.now(UTC) - start_time).total_seconds()
        
        result = {
            "success": True,
            "results": formatted,
            "result_count": len(results),
            "sql_query": sql_query,
            "query_description": query_description,
            "timestamp": datetime.now(UTC).isoformat(),
            "execution_time_seconds": query_duration,
            "total_time_seconds": total_duration,
            "cache_hit": cached_sql is not None
        }
        
        logger.info(f"✅ SEARCH_PROJECT COMPLETED SUCCESSFULLY")
        logger.info(f"   Results: {len(results)} rows")
        logger.info(f"   Query time: {query_duration:.3f}s")
        logger.info(f"   Total time: {total_duration:.3f}s")
        logger.info(f"   Cache: {'HIT' if result['cache_hit'] else 'MISS'}")
        logger.info(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        error_duration = (datetime.now(UTC) - start_time).total_seconds()
        logger.error(f"❌ SEARCH_PROJECT FAILED")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Duration: {error_duration:.3f}s")
        logger.error(f"{'='*70}\n", exc_info=True)
        
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "query_description": query_description,
            "sql_query": locals().get('sql_query', 'N/A'),
            "execution_time_seconds": error_duration
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
        
        # Smart detection: Find ANY column with 'name' in it
        name_value = None
        for key, value in row.items():
            if 'name' in key.lower() and value:
                name_value = value
                break
        if name_value:
            key_fields.append(f"{name_value}")
        
        # Smart detection: Find ANY column with 'path' in it
        path_value = None
        for key, value in row.items():
            if 'path' in key.lower() and value:
                path_value = value
                break
        if path_value:
            key_fields.append(f"({path_value})")
        
        # Smart detection: Find ANY column with 'type' or 'kind' in it
        type_value = None
        for key, value in row.items():
            if ('type' in key.lower() or 'kind' in key.lower()) and value:
                type_value = value
                break
        if type_value:
            key_fields.append(f"[{type_value}]")
        
        # If no standard fields found, show first few columns
        if not key_fields:
            for key, value in list(row.items())[:3]:
                if value is not None:
                    key_fields.append(f"{key}: {value}")
        
        summary += " ".join(key_fields) + "\n"
    
    if count > 10:
        summary += f"\n... and {count - 10} more results"
    
    return summary


def get_cache_stats() -> Dict[str, Any]:
    """Get current cache statistics."""
    return {
        "cache_size": len(_query_cache),
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate": _cache_hits / (_cache_hits + _cache_misses) if (_cache_hits + _cache_misses) > 0 else 0.0
    }


def clear_query_cache() -> None:
    """Clear the query cache (useful for testing)."""
    global _query_cache, _cache_hits, _cache_misses
    _query_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("🗑️ Query cache cleared")
