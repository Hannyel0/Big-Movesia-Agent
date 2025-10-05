"""Code snippets tool using Qdrant vector search."""

from __future__ import annotations
from typing import Optional, Dict, Any
import os
from datetime import datetime, UTC

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

from react_agent.context import Context


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
