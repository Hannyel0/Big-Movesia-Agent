"""
Smart Unity Documentation RAG Tool using direct Qdrant HTTP API.
Follows the same pattern as code_snippets.py for consistency.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from functools import lru_cache
from datetime import datetime, UTC

import httpx
import trafilatura
from trafilatura.settings import use_config
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
UNITY_DOCS_COLLECTION = os.getenv("UNITY_DOCS_COLLECTION", "unity_docs-fast")
EMBED_SERVER_URL = os.getenv("EMBED_SERVER_URL", "http://127.0.0.1:8766")


class UnityDocsInput(BaseModel):
    """Input schema for Unity docs search."""
    query: str = Field(description="Natural language query about Unity functionality, API, or concepts")
    version: Optional[str] = Field(default=None, description="Unity version filter (e.g., '6000.2')")
    category: Optional[str] = Field(default=None, description="Category filter: 'ScriptReference', 'Manual', etc.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return (1-20)")
    fetch_full_content: bool = Field(default=True, description="Whether to fetch full page content from URLs")
    score_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum relevance threshold")


# Simple in-memory cache for fetched URLs
_url_cache: Dict[str, str] = {}
_cache_max_size = 100


async def _fetch_url_content(url: str) -> Optional[str]:
    """
    Fetch and extract clean content from Unity documentation URL.
    Uses trafilatura for robust content extraction with fallback strategies.
    """
    global _url_cache

    logger.info(f"🔍 [UnityDocs._fetch_url_content] Starting content fetch for: {url}")

    # Check cache first
    if url in _url_cache:
        logger.info(f"📋 [UnityDocs._fetch_url_content] Cache hit for {url}")
        logger.debug(f"📊 [UnityDocs._fetch_url_content] Cache stats - Size: {len(_url_cache)}/{_cache_max_size}")
        return _url_cache[url]

    logger.debug(f"📊 [UnityDocs._fetch_url_content] Cache miss - Current cache size: {len(_url_cache)}/{_cache_max_size}")

    try:
        logger.debug(f"🌐 [UnityDocs._fetch_url_content] Creating HTTP client with 30s timeout")
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        ) as client:
            logger.info(f"📡 [UnityDocs._fetch_url_content] Fetching URL: {url}")
            response = await client.get(url)
            response.raise_for_status()
            logger.info(f"✅ [UnityDocs._fetch_url_content] HTTP {response.status_code} - Downloaded {len(response.text)} chars")

            # Configure trafilatura for optimal extraction
            logger.debug(f"⚙️ [UnityDocs._fetch_url_content] Configuring trafilatura for extraction")
            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")  # No timeout for async

            # Primary extraction with trafilatura
            logger.info(f"🔧 [UnityDocs._fetch_url_content] Starting primary content extraction with trafilatura")
            extracted = trafilatura.extract(
                response.text,
                output_format='txt',
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                no_fallback=False,  # Enable fallback extraction
                favor_precision=False,  # Favor recall for documentation
                favor_recall=True,
                config=config
            )

            if extracted:
                logger.info(f"✅ [UnityDocs._fetch_url_content] Primary extraction successful - {len(extracted.strip())} chars")
            else:
                logger.warning(f"⚠️ [UnityDocs._fetch_url_content] Primary extraction returned None")

            if not extracted or len(extracted.strip()) < 100:
                # Fallback: Try with different settings
                logger.warning(f"⚠️ [UnityDocs._fetch_url_content] Primary extraction yielded little content ({len(extracted.strip()) if extracted else 0} chars), trying fallback mode")
                extracted = trafilatura.extract(
                    response.text,
                    output_format='txt',
                    include_comments=False,
                    include_tables=True,
                    no_fallback=True,  # Force fallback mode
                    favor_precision=False,
                    favor_recall=True,
                    config=config
                )

                if extracted:
                    logger.info(f"✅ [UnityDocs._fetch_url_content] Fallback extraction successful - {len(extracted.strip())} chars")
                else:
                    logger.error(f"❌ [UnityDocs._fetch_url_content] Fallback extraction also returned None")

            if not extracted:
                logger.error(f"❌ [UnityDocs._fetch_url_content] Failed to extract any content from {url}")
                return None

            # Clean up the extracted text
            logger.debug(f"🧹 [UnityDocs._fetch_url_content] Starting text cleanup")
            clean_text = extracted.strip()
            original_length = len(clean_text)
            logger.debug(f"📏 [UnityDocs._fetch_url_content] Text after strip: {original_length} chars")

            # Remove excessive blank lines (more than 2 consecutive)
            logger.debug(f"🧹 [UnityDocs._fetch_url_content] Removing excessive blank lines")
            lines = clean_text.split('\n')
            cleaned_lines = []
            blank_count = 0
            for line in lines:
                if line.strip():
                    cleaned_lines.append(line)
                    blank_count = 0
                else:
                    blank_count += 1
                    if blank_count <= 2:
                        cleaned_lines.append(line)

            clean_text = '\n'.join(cleaned_lines)
            logger.debug(f"📏 [UnityDocs._fetch_url_content] Text after blank line removal: {len(clean_text)} chars (removed {original_length - len(clean_text)} chars)")

            # Smart truncation that respects paragraphs
            max_chars = 8000
            if len(clean_text) > max_chars:
                logger.info(f"✂️ [UnityDocs._fetch_url_content] Content too long ({len(clean_text)} chars), truncating to {max_chars}")

                # Find the last paragraph break before the limit
                truncate_point = clean_text.rfind('\n\n', 0, max_chars)
                if truncate_point > max_chars * 0.8:  # If we found a good break point
                    logger.debug(f"✂️ [UnityDocs._fetch_url_content] Truncating at paragraph break (position {truncate_point})")
                    clean_text = clean_text[:truncate_point] + "\n\n[Content truncated at natural paragraph break...]"
                else:
                    # Fall back to character limit with sentence awareness
                    truncate_point = clean_text.rfind('. ', 0, max_chars)
                    if truncate_point > max_chars * 0.9:
                        logger.debug(f"✂️ [UnityDocs._fetch_url_content] Truncating at sentence break (position {truncate_point})")
                        clean_text = clean_text[:truncate_point + 1] + "\n\n[Content truncated...]"
                    else:
                        logger.debug(f"✂️ [UnityDocs._fetch_url_content] Hard truncation at {max_chars} chars")
                        clean_text = clean_text[:max_chars] + "\n\n[Content truncated...]"
            else:
                logger.debug(f"✅ [UnityDocs._fetch_url_content] Content within limit ({len(clean_text)}/{max_chars} chars), no truncation needed")

            # Cache the result (simple FIFO eviction)
            if len(_url_cache) >= _cache_max_size:
                first_key = next(iter(_url_cache))
                logger.debug(f"🗑️ [UnityDocs._fetch_url_content] Cache full, evicting oldest entry: {first_key}")
                del _url_cache[first_key]

            _url_cache[url] = clean_text
            logger.info(f"✅ [UnityDocs._fetch_url_content] Successfully fetched and cached - URL: {url}, Size: {len(clean_text)} chars, Cache: {len(_url_cache)}/{_cache_max_size}")

            return clean_text

    except httpx.HTTPError as e:
        logger.error(f"❌ [UnityDocs._fetch_url_content] HTTP error fetching {url}: {type(e).__name__} - {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ [UnityDocs._fetch_url_content] Unexpected error fetching {url}: {type(e).__name__} - {e}", exc_info=True)
        return None


async def _get_local_embedding(text: str) -> Dict[str, Any]:
    """Generate embedding using local embedding server."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{EMBED_SERVER_URL}/embed",
            json={"input": text, "type": "query"}
        )

        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.status_code}")

        result = response.json()
        return {
            "embedding": result["embedding"],
            "model": "Xenova/bge-small-en-v1.5",
            "dimension": len(result["embedding"])
        }


async def _search_unity_docs_semantic(
    query: str,
    version: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5,
    score_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Perform semantic search against Unity docs collection.
    Uses Qdrant's query API with text input (auto-embedding).
    """
    logger.info(f"🔍 [UnityDocs._search_unity_docs_semantic] Starting search - Query: '{query}', Limit: {limit}, Threshold: {score_threshold}")
    logger.debug(f"📋 [UnityDocs._search_unity_docs_semantic] Filters - Version: {version}, Category: {category}")

    try:
        # Build filter conditions
        filter_conditions = []
        if version:
            filter_conditions.append({
                "key": "version",
                "match": {"value": version}
            })
            logger.debug(f"🔧 [UnityDocs._search_unity_docs_semantic] Added version filter: {version}")
        if category:
            filter_conditions.append({
                "key": "category",
                "match": {"value": category}
            })
            logger.debug(f"🔧 [UnityDocs._search_unity_docs_semantic] Added category filter: {category}")

        logger.debug(f"📊 [UnityDocs._search_unity_docs_semantic] Total filter conditions: {len(filter_conditions)}")

        # Generate embedding first (like code_snippets does)
        logger.info(f"🔧 [UnityDocs._search_unity_docs_semantic] Generating embedding for query")
        embedding_result = await _get_local_embedding(query)
        query_vector = embedding_result["embedding"]
        logger.debug(f"✅ [UnityDocs._search_unity_docs_semantic] Embedding generated: dimension={len(query_vector)}")

        # Construct search payload with VECTOR (not text)
        search_payload = {
            "vector": query_vector,  # ✅ FIX: Use vector, not text query
            "limit": limit * 2,  # Fetch 2x for filtering
            "with_payload": True,
            "score_threshold": score_threshold
        }

        if filter_conditions:
            search_payload["filter"] = {"must": filter_conditions}
            logger.debug(f"✅ [UnityDocs._search_unity_docs_semantic] Applied {len(filter_conditions)} filter(s) to search")

        # ✅ FIX: Use /points/search endpoint (not /points/query)
        qdrant_endpoint = f"{QDRANT_URL}/collections/{UNITY_DOCS_COLLECTION}/points/search"
        logger.info(f"📡 [UnityDocs._search_unity_docs_semantic] Sending query to Qdrant: {qdrant_endpoint}")
        logger.debug(f"📦 [UnityDocs._search_unity_docs_semantic] Payload keys: {list(search_payload.keys())}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                qdrant_endpoint,
                json=search_payload,
                timeout=30.0
            )

            logger.info(f"📨 [UnityDocs._search_unity_docs_semantic] Qdrant response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"❌ [UnityDocs._search_unity_docs_semantic] Qdrant search failed: {response.status_code} - {response.text}")
                return []

            search_results = response.json()
            logger.debug(f"📊 [UnityDocs._search_unity_docs_semantic] Raw response keys: {list(search_results.keys())}")

            # Extract results
            raw_hits = search_results.get("result", [])
            logger.info(f"📊 [UnityDocs._search_unity_docs_semantic] Qdrant returned {len(raw_hits)} results")

            results = []
            for idx, hit in enumerate(raw_hits):
                payload = hit.get("payload", {})
                score = hit.get("score", 0.0)
                title = payload.get("title", "Unknown")

                logger.debug(f"📄 [UnityDocs._search_unity_docs_semantic] Result {idx+1}: {title} (score: {score:.3f})")

                # Additional relevance boost for title matches
                query_lower = query.lower()
                title_lower = title.lower()
                if query_lower in title_lower:
                    original_score = score
                    score *= 1.2  # 20% boost
                    logger.debug(f"⭐ [UnityDocs._search_unity_docs_semantic] Title match boost: {original_score:.3f} -> {score:.3f} for '{title}'")

                results.append({
                    "payload": payload,
                    "score": score
                })

            logger.info(f"✅ [UnityDocs._search_unity_docs_semantic] Processed {len(results)} results")

            # Re-sort after boosting
            results.sort(key=lambda x: x["score"], reverse=True)
            logger.debug(f"🔀 [UnityDocs._search_unity_docs_semantic] Re-sorted results after title boosting")

            # Apply final limit
            final_results = results[:limit]
            logger.info(f"✂️ [UnityDocs._search_unity_docs_semantic] Returning top {len(final_results)} results (limit: {limit})")

            if final_results:
                top_titles = [r["payload"].get("title", "Unknown")[:50] for r in final_results[:3]]
                logger.info(f"🏆 [UnityDocs._search_unity_docs_semantic] Top results: {', '.join(top_titles)}")

            return final_results

    except Exception as e:
        logger.error(f"❌ [UnityDocs._search_unity_docs_semantic] Search failed: {type(e).__name__} - {e}", exc_info=True)
        return []


@tool(args_schema=UnityDocsInput)
async def unity_docs(
    query: str,
    version: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5,
    fetch_full_content: bool = True,
    score_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Search locally indexed Unity documentation using semantic search.

    This tool provides fast, semantic search over Unity's documentation with:
    - Semantic similarity matching (auto-embedded by Qdrant)
    - Version and category filtering
    - Automatic full content fetching from URLs
    - Relevance scoring and ranking

    Best for: Unity API references, scripting examples, feature documentation.

    Args:
        query: Natural language description of what you're looking for
        version: Optional Unity version filter (e.g., "6000.2")
        category: Optional category filter ("ScriptReference", "Manual", etc.)
        top_k: Number of results to return (1-20)
        fetch_full_content: Whether to fetch full page content from URLs
        score_threshold: Minimum relevance threshold (0.0-1.0)

    Returns:
        Dictionary with search results including titles, URLs, descriptions,
        relevance scores, and optionally full page content.

    Examples:
        unity_docs(query="how to detect 2D collisions")
        unity_docs(query="Collider2D.CreateMesh", category="ScriptReference")
        unity_docs(query="particle system", version="6000.2", top_k=3)
    """
    logger.info(f"{'='*80}")
    logger.info(f"🚀 [UnityDocs] TOOL INVOKED")
    logger.info(f"📝 [UnityDocs] Query: '{query}'")
    logger.info(f"🔧 [UnityDocs] Parameters: top_k={top_k}, fetch_content={fetch_full_content}, threshold={score_threshold}")
    logger.info(f"🔧 [UnityDocs] Filters: version={version}, category={category}")
    logger.info(f"🌐 [UnityDocs] Qdrant URL: {QDRANT_URL}")
    logger.info(f"📚 [UnityDocs] Collection: {UNITY_DOCS_COLLECTION}")
    logger.info(f"{'='*80}")

    try:
        # Perform semantic search
        logger.info(f"🔍 [UnityDocs] Step 1: Performing semantic search")
        search_results = await _search_unity_docs_semantic(
            query=query,
            version=version,
            category=category,
            limit=top_k,
            score_threshold=score_threshold
        )
        logger.info(f"📊 [UnityDocs] Semantic search returned {len(search_results)} results")

        if not search_results:
            logger.warning(f"⚠️ [UnityDocs] No results found matching query with score >= {score_threshold}")
            response = {
                "success": True,
                "results": [],
                "result_count": 0,
                "message": f"No Unity docs found matching query with score >= {score_threshold}",
                "query": query,
                "filters_applied": {
                    "version": version,
                    "category": category,
                    "min_score": score_threshold
                }
            }
            logger.info(f"📤 [UnityDocs] Returning empty result response")
            return response

        # Structure results
        logger.info(f"🔨 [UnityDocs] Step 2: Structuring {len(search_results)} results")
        structured_results = []
        for idx, result in enumerate(search_results):
            payload = result["payload"]
            title = payload.get("title", "Unknown")
            url = payload.get("url", "")
            score = result["score"]

            logger.debug(f"📄 [UnityDocs] Processing result {idx+1}/{len(search_results)}: {title} (score: {score:.3f})")

            doc_entry = {
                "title": title,
                "url": url,
                "version": payload.get("version", ""),
                "relevance_score": round(score, 3)
            }

            # Optionally fetch full content
            if fetch_full_content and doc_entry["url"]:
                logger.info(f"📥 [UnityDocs] Step 3.{idx+1}: Fetching full content for '{title}'")
                full_content = await _fetch_url_content(doc_entry["url"])
                if full_content:
                    doc_entry["full_content"] = full_content
                    doc_entry["content_fetched"] = True
                    logger.info(f"✅ [UnityDocs] Successfully fetched content ({len(full_content)} chars)")
                else:
                    doc_entry["content_fetched"] = False
                    logger.warning(f"⚠️ [UnityDocs] Failed to fetch content for '{title}'")
            else:
                logger.debug(f"⏭️ [UnityDocs] Skipping content fetch (fetch_full_content={fetch_full_content})")

            structured_results.append(doc_entry)
            logger.debug(f"✅ [UnityDocs] Added result {idx+1} to structured results")

        logger.info(f"✅ [UnityDocs] Successfully processed all {len(structured_results)} results")

        response = {
            "success": True,
            "results": structured_results,
            "result_count": len(structured_results),
            "message": f"Found {len(structured_results)} relevant Unity docs",
            "query": query,
            "filters_applied": {
                "version": version,
                "category": category,
                "min_score": score_threshold
            },
            "timestamp": datetime.now(UTC).isoformat()
        }

        logger.info(f"{'='*80}")
        logger.info(f"✅ [UnityDocs] TOOL COMPLETED SUCCESSFULLY")
        logger.info(f"📊 [UnityDocs] Results: {len(structured_results)} docs")
        logger.info(f"📝 [UnityDocs] Top result: {structured_results[0]['title'] if structured_results else 'N/A'}")
        logger.info(f"⏱️ [UnityDocs] Timestamp: {response['timestamp']}")
        logger.info(f"{'='*80}")

        return response

    except Exception as e:
        logger.error(f"{'='*80}")
        logger.error(f"❌ [UnityDocs] TOOL FAILED")
        logger.error(f"❌ [UnityDocs] Error type: {type(e).__name__}")
        logger.error(f"❌ [UnityDocs] Error message: {str(e)}")
        logger.error(f"{'='*80}", exc_info=True)

        error_response = {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "results": [],
            "result_count": 0,
            "query": query,
            "error_type": type(e).__name__
        }

        return error_response
