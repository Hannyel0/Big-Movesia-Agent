"""
Smart entity extraction for Unity/Unreal projects.

Extracts and normalizes file paths, GameObjects, components, and other entities
from tool results and text with intelligent matching.
"""

from typing import List, Set, Dict, Any, Optional, Tuple
import logging
import re
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Entity types we care about
class EntityType:
    FILE = "file"           # .cs, .unity, .prefab, etc.
    GAMEOBJECT = "gameobject"  # Scene objects
    COMPONENT = "component"    # Unity components
    ASSET = "asset"            # Textures, materials, etc.
    SCENE = "scene"            # Scene files
    UNKNOWN = "unknown"

# Unity file extensions by category
FILE_EXTENSIONS = {
    EntityType.FILE: {".cs", ".shader", ".cginc"},
    EntityType.SCENE: {".unity"},
    EntityType.ASSET: {".prefab", ".asset", ".mat", ".anim", ".controller", 
                      ".png", ".jpg", ".fbx", ".wav", ".mp3"}
}

# Patterns for Unity class names
UNITY_CLASS_PATTERNS = [
    r'\b\w+Controller\b',
    r'\b\w+Manager\b', 
    r'\b\w+System\b',
    r'\b\w+Handler\b',
    r'\b\w+Component\b',
    r'\b\w+Behaviour\b',
    r'\b\w+Script\b'
]


def normalize_entity(entity: str, entity_type: str = EntityType.UNKNOWN) -> str:
    """
    Normalize an entity to a canonical form.
    
    Rules:
    - Convert to lowercase for consistent storage
    - For file paths: extract just the filename (keep extension)
    - Remove special characters except . / _
    - Trim whitespace
    """
    if not entity:
        return ""
    
    # Basic cleanup
    entity = entity.strip().lower()
    
    # If it's a file path, extract filename
    if "/" in entity or "\\" in entity:
        entity = entity.replace("\\", "/")
        entity = entity.split("/")[-1]  # Get filename only
    
    # Remove quotes, parentheses, brackets
    entity = entity.strip("\"'()[]{}.,;:")
    
    return entity


def classify_entity(entity: str) -> str:
    """
    Classify what type of entity this is.
    
    Returns one of: file, gameobject, component, asset, scene, unknown
    """
    entity_lower = entity.lower()
    
    # Check file extensions
    for entity_type, extensions in FILE_EXTENSIONS.items():
        if any(entity_lower.endswith(ext) for ext in extensions):
            return entity_type
    
    # Check for Unity class patterns
    for pattern in UNITY_CLASS_PATTERNS:
        if re.search(pattern, entity, re.IGNORECASE):
            return EntityType.COMPONENT
    
    return EntityType.UNKNOWN


def extract_entities_from_tool_result(
    tool_name: str,
    result: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Extract entities from tool results (PRIMARY source).
    
    Returns list of (entity, entity_type) tuples.
    """
    logger.info(f"🔍 [EntityExtractor] Extracting from {tool_name} tool result")
    
    entities = []
    
    if not isinstance(result, dict) or not result.get("success", True):
        logger.warning(f"  ⚠️ Tool result not valid or unsuccessful")
        return entities
    
    if tool_name == "search_project":
        logger.debug(f"  Checking search_project result structure...")
        
        # Priority 1: Use results_structured (has full data)
        structured = result.get("results_structured", [])
        if isinstance(structured, list):
            logger.info(f"  Found results_structured with {len(structured)} items")
            
            for idx, item in enumerate(structured[:20], 1):  # Top 20 results
                if isinstance(item, dict):
                    entity = item.get("path") or item.get("name")
                    if entity:
                        normalized = normalize_entity(entity)
                        if normalized and not _is_template_pattern(normalized):
                            entity_type = classify_entity(normalized)
                            entities.append((normalized, entity_type))
                            logger.debug(f"    {idx}. ✅ Extracted: {normalized} (type: {entity_type})")
                        else:
                            logger.debug(f"    {idx}. ❌ Skipped template: {entity}")
        
        # Fallback: Use regular results
        if not entities:
            logger.debug(f"  No results_structured, trying regular results...")
            results = result.get("results", [])
            if isinstance(results, list):
                logger.info(f"  Found regular results with {len(results)} items")
                
                for idx, item in enumerate(results[:20], 1):
                    if isinstance(item, dict):
                        entity = item.get("path") or item.get("name") or item.get("guid")
                        if entity:
                            normalized = normalize_entity(entity)
                            if normalized and not _is_template_pattern(normalized):
                                entity_type = classify_entity(normalized)
                                entities.append((normalized, entity_type))
                                logger.debug(f"    {idx}. ✅ Extracted: {normalized} (type: {entity_type})")
                            else:
                                logger.debug(f"    {idx}. ❌ Skipped template: {entity}")
    
    elif tool_name == "code_snippets":
        logger.debug(f"  Checking code_snippets result...")
        snippets = result.get("snippets", [])
        if isinstance(snippets, list):
            logger.info(f"  Found {len(snippets)} snippets")
            
            for idx, snippet in enumerate(snippets[:10], 1):
                if isinstance(snippet, dict):
                    file_path = snippet.get("file_path")
                    if file_path:
                        normalized = normalize_entity(file_path)
                        if normalized and not _is_template_pattern(normalized):
                            entities.append((normalized, EntityType.FILE))
                            logger.debug(f"    {idx}. ✅ Extracted: {normalized}")
    
    elif tool_name in ["read_file", "write_file", "modify_file", "delete_file", "move_file"]:
        logger.debug(f"  Checking {tool_name} result...")
        # Check both locations for file path
        file_path = result.get("file_path") or result.get("pending_operation", {}).get("rel_path")
        if file_path:
            normalized = normalize_entity(file_path)
            if normalized and not _is_template_pattern(normalized):
                entities.append((normalized, EntityType.FILE))
                logger.debug(f"    ✅ Extracted: {normalized}")
        
        # For move_file, also extract destination path
        if tool_name == "move_file":
            to_path = result.get("to_path") or result.get("pending_operation", {}).get("to_path")
            if to_path:
                normalized = normalize_entity(to_path)
                if normalized and not _is_template_pattern(normalized):
                    entities.append((normalized, EntityType.FILE))
                    logger.debug(f"    ✅ Extracted destination: {normalized}")
        
        if not file_path:
            logger.warning(f"    ⚠️ No file_path found in result")
    
    logger.info(f"✅ [EntityExtractor] Extracted {len(entities)} entities from {tool_name}")
    return entities


def extract_entities_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract entities from text (SECONDARY source).
    
    Used for extracting from queries, messages, etc.
    Returns list of (entity, entity_type) tuples.
    """
    logger.info(f"🔍 [EntityExtractor] Extracting from text ({len(text)} chars)")
    logger.debug(f"  Text preview: '{text[:100]}'")
    
    entities = []
    text_lower = text.lower()
    
    # Strategy 1: Find file paths/names with extensions
    logger.debug(f"  Strategy 1: Looking for file extensions...")
    all_extensions = set()
    for exts in FILE_EXTENSIONS.values():
        all_extensions.update(exts)
    
    words = text.split()
    found_by_extension = 0
    for word in words:
        word_clean = word.strip(".,;:\"'()[]{}" ).lower()
        
        # Check for file extensions
        if any(word_clean.endswith(ext) for ext in all_extensions):
            normalized = normalize_entity(word_clean)
            if normalized and not _is_template_pattern(normalized):
                entity_type = classify_entity(normalized)
                entities.append((normalized, entity_type))
                logger.debug(f"    ✅ Found by extension: {normalized} (type: {entity_type})")
                found_by_extension += 1
    
    logger.info(f"  Strategy 1 found: {found_by_extension} entities")
    
    # Strategy 2: Find Unity class patterns
    logger.debug(f"  Strategy 2: Looking for Unity class patterns...")
    found_by_pattern = 0
    for pattern in UNITY_CLASS_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity = match.group(0)
            normalized = normalize_entity(entity)
            if normalized and normalized not in [e[0] for e in entities]:
                entities.append((normalized, EntityType.COMPONENT))
                logger.debug(f"    ✅ Found by pattern: {normalized}")
                found_by_pattern += 1
    
    logger.info(f"  Strategy 2 found: {found_by_pattern} entities")
    logger.info(f"✅ [EntityExtractor] Extracted {len(entities)} entities from text")
    
    return entities


def _is_template_pattern(entity: str) -> bool:
    """Check if entity looks like a template pattern (test123.cs, file[id].cs)."""
    # Match patterns like: test123, test[123], file_456, temp_abc123
    template_patterns = [
        r'^test\d+',           # test123.cs
        r'^temp_?\d+',         # temp123.cs, temp_123.cs
        r'^file_?\d+',         # file123.cs
        r'\[\d+\]',            # file[123].cs
        r'^new_?file',         # newfile.cs
        r'^untitled'           # untitled.cs
    ]
    
    entity_lower = entity.lower()
    return any(re.search(pattern, entity_lower) for pattern in template_patterns)


def extract_entities_simple(
    text: str = "",
    tool_name: Optional[str] = None,
    tool_result: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Main entry point for entity extraction.
    
    Args:
        text: Text to extract from (queries, messages)
        tool_name: Name of tool if extracting from result
        tool_result: Tool result dict
    
    Returns:
        List of normalized entity names (up to 10)
    """
    logger.info(f"🔍 [EntityExtractor] ===== STARTING EXTRACTION =====")
    logger.info(f"  Text length: {len(text)} chars")
    logger.info(f"  Tool name: {tool_name or 'None'}")
    logger.info(f"  Has tool result: {tool_result is not None}")
    
    all_entities = {}  # entity -> entity_type
    
    # Priority 1: Extract from tool result (if available)
    if tool_name and tool_result:
        logger.info(f"📊 [EntityExtractor] Priority 1: Extracting from tool result...")
        tool_entities = extract_entities_from_tool_result(tool_name, tool_result)
        for entity, entity_type in tool_entities:
            if entity not in all_entities:
                all_entities[entity] = entity_type
        
        logger.info(f"  Added {len(tool_entities)} entities from tool result")
    
    # Priority 2: Extract from text
    if text:
        logger.info(f"📊 [EntityExtractor] Priority 2: Extracting from text...")
        text_entities = extract_entities_from_text(text)
        new_from_text = 0
        for entity, entity_type in text_entities:
            if entity not in all_entities:
                all_entities[entity] = entity_type
                new_from_text += 1
        
        logger.info(f"  Added {new_from_text} new entities from text")
    
    # Return up to 10 most relevant
    result = list(all_entities.keys())[:10]
    
    logger.info(f"✅ [EntityExtractor] ===== EXTRACTION COMPLETE =====")
    logger.info(f"  Total unique entities: {len(all_entities)}")
    logger.info(f"  Returning top {len(result)} entities")
    
    if result:
        logger.info(f"  Entities: {result}")
        # Log entity types breakdown
        type_counts = {}
        for entity, entity_type in all_entities.items():
            if entity in result:
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        logger.info(f"  Type breakdown: {dict(type_counts)}")
    else:
        logger.warning(f"  ⚠️ No entities extracted!")
    
    return result


def get_extraction_stats() -> Dict[str, Any]:
    """Get statistics about entity extraction configuration."""
    total_extensions = sum(len(exts) for exts in FILE_EXTENSIONS.values())
    
    return {
        "file_extensions_count": total_extensions,
        "entity_types": list(FILE_EXTENSIONS.keys()),
        "unity_class_patterns": len(UNITY_CLASS_PATTERNS),
        "supported_extensions": {
            entity_type: list(exts) 
            for entity_type, exts in FILE_EXTENSIONS.items()
        }
    }


def debug_extraction(
    text: str = "",
    tool_name: Optional[str] = None,
    tool_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Debug version that returns full details about extraction.
    
    Useful for troubleshooting entity extraction issues.
    
    Args:
        text: Text to extract from (queries, messages)
        tool_name: Name of tool if extracting from result
        tool_result: Tool result dict
    
    Returns:
        Dictionary with detailed extraction information
    """
    logger.info(f"🐛 [EntityExtractor] ===== DEBUG EXTRACTION =====")
    
    debug_info = {
        "text_length": len(text),
        "tool_name": tool_name,
        "has_tool_result": tool_result is not None,
        "entities_by_source": {},
        "entities_by_type": {},
        "skipped_templates": [],
        "all_words_checked": []
    }
    
    # Check what's in the tool result
    if tool_result:
        debug_info["tool_result_keys"] = list(tool_result.keys())
        debug_info["tool_result_success"] = tool_result.get("success", "N/A")
        
        if tool_name == "search_project":
            debug_info["has_results_structured"] = "results_structured" in tool_result
            debug_info["has_results"] = "results" in tool_result
            
            structured = tool_result.get("results_structured", [])
            if structured:
                debug_info["structured_count"] = len(structured)
                debug_info["structured_sample"] = structured[0] if structured else None
    
    # Try extraction
    entities = extract_entities_simple(text, tool_name, tool_result)
    debug_info["extracted_entities"] = entities
    debug_info["extraction_count"] = len(entities)
    
    logger.info(f"🐛 [EntityExtractor] Debug info: {json.dumps(debug_info, indent=2, default=str)}")
    
    return debug_info
