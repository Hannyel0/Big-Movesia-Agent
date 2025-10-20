"""
Lightweight keyword-based topic extraction for Unity/game development.

Uses comprehensive seed topics + smart matching for word variations.
"""

from typing import List, Set
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# COMPREHENSIVE UNITY/UNREAL SEED TOPICS
# ============================================================================

# Core topics (canonical forms)
SEED_TOPICS = [
    # Input & Control
    "input", "keyboard", "mouse", "gamepad", "touch", "controller",
    
    # Physics
    "physics", "rigidbody", "collider", "collision", "trigger", "force", 
    "velocity", "gravity", "constraint",
    
    # Animation
    "animation", "animator", "blend", "state machine", "motion",
    
    # Audio
    "audio", "sound", "music", "sfx", "mixer", "listener",
    
    # UI
    "ui", "canvas", "button", "text", "image", "slider", "panel", 
    "scroll", "menu", "hud",
    
    # Rendering
    "camera", "material", "shader", "texture", "lighting", "shadow",
    "render", "mesh", "sprite",
    
    # Scripting
    "script", "code", "component", "monobehaviour", "coroutine",
    
    # Scene & Objects
    "scene", "level", "gameobject", "prefab", "hierarchy", "transform",
    "position", "rotation", "scale",
    
    # Particle & VFX
    "particle", "effect", "vfx", "trail", "emission",
    
    # AI & Navigation
    "ai", "navmesh", "pathfinding", "navigation", "agent", "behavior",
    "enemy", "npc",
    
    # Player & Character
    "player", "character", "movement", "controller", "walk", "run", 
    "jump", "sprint",
    
    # Combat & Damage
    "combat", "damage", "health", "weapon", "attack", "shoot", "hit",
    
    # Inventory & Items
    "inventory", "item", "pickup", "collect", "equip",
    
    # Networking
    "network", "multiplayer", "sync", "rpc", "server", "client",
    
    # Save & Data
    "save", "load", "persistence", "data", "playerprefs", "serialization",
    
    # Performance
    "performance", "optimization", "profiling", "fps", "memory", "garbage",
    
    # Debugging & Errors
    "error", "debug", "bug", "crash", "exception", "null", "warning",
    "compilation", "build",
    
    # Terrain & Environment
    "terrain", "landscape", "foliage", "vegetation", "heightmap",
    
    # Post Processing
    "postprocess", "bloom", "colorgrade", "dof", "motion blur",
    
    # Timeline & Cinematics
    "timeline", "cutscene", "cinematic", "sequence",
    
    # Unreal Specific
    "blueprint", "actor", "pawn", "gamemode", "niagara",
    
    # Asset Management
    "asset", "import", "export", "resource", "addressable",
    
    # Testing
    "test", "unit test", "playmode",
]


# ============================================================================
# TOPIC SYNONYMS & VARIATIONS
# ============================================================================

TOPIC_SYNONYMS: dict[str, Set[str]] = {
    # Input variations
    "input": {"inputs", "inputsystem", "input system"},
    
    # Physics variations
    "physics": {"physic", "physical", "rigidbody2d", "rigidbody3d"},
    "collider": {"colliders", "collider2d", "collision", "collisions"},
    "rigidbody": {"rb", "rigid body", "rigidbodies"},
    
    # Animation variations
    "animation": {"anim", "animate", "animated", "animating", "animator", 
                  "animations", "animclip", "animationclip"},
    
    # Scripting variations
    "script": {"scripts", "scripting", "scriptable", "cs"},
    "code": {"coding", "coded", "csharp", "c#"},
    "component": {"components", "getcomponent", "addcomponent"},
    
    # UI variations
    "ui": {"gui", "interface", "uielements", "ugui"},
    "button": {"buttons", "btn"},
    "canvas": {"canvases"},
    
    # Rendering variations
    "camera": {"cameras", "cam", "orthographic", "perspective"},
    "shader": {"shaders", "shadergraph", "shaderlab"},
    "material": {"materials", "mat", "materialinstance"},
    "texture": {"textures", "tex", "sprite", "sprites"},
    "lighting": {"light", "lights", "illumination"},
    "render": {"rendering", "renderer", "renders"},
    
    # Scene & Objects
    "scene": {"scenes", "scenemanagement", "level", "levels"},
    "gameobject": {"gameobjects", "go", "object", "objects"},
    "prefab": {"prefabs", "instantiate", "instantiation"},
    "transform": {"transforms", "position", "rotation", "scale"},
    
    # Particle variations
    "particle": {"particles", "particlesystem", "particleemitter"},
    "effect": {"effects", "fx", "vfx", "visualeffect"},
    
    # AI variations
    "ai": {"artificial intelligence", "agent", "agents"},
    "pathfinding": {"pathfind", "navigation", "navmesh", "navmeshagent"},
    "enemy": {"enemies", "hostile", "mob", "mobs"},
    "npc": {"npcs", "non-player character"},
    
    # Player variations
    "player": {"players", "playercontroller", "playercharacter"},
    "character": {"characters", "charactercontroller"},
    "movement": {"move", "moving", "moved", "locomotion"},
    
    # Combat variations
    "combat": {"fight", "fighting", "battle"},
    "damage": {"hurt", "dmg", "harmed"},
    "health": {"hp", "hitpoints", "healthbar"},
    "weapon": {"weapons", "gun", "guns", "sword"},
    
    # Inventory variations
    "inventory": {"inventories", "bag", "storage"},
    "item": {"items", "loot", "drop"},
    
    # Networking variations
    "network": {"networking", "multiplayer", "online"},
    "sync": {"synchronize", "synchronization", "synced"},
    
    # Error/Debug variations
    "error": {"errors", "exception", "exceptions", "fail", "failure"},
    "debug": {"debugging", "debugger", "log", "logging"},
    "bug": {"bugs", "issue", "issues", "problem", "problems"},
    
    # Performance variations
    "performance": {"optimize", "optimization", "optimizing", "efficient"},
    "profiling": {"profiler", "profile"},
    
    # Unreal variations
    "blueprint": {"blueprints", "bp", "visual scripting"},
    "actor": {"actors", "pawn", "pawns"},
}


# ============================================================================
# SMART MATCHING FUNCTIONS
# ============================================================================

def normalize_word(word: str) -> str:
    """
    Normalize a word for matching by removing common suffixes.
    
    Examples:
        "scripting" -> "script"
        "animations" -> "animation"
        "colliders" -> "collider"
    """
    word = word.lower().strip()
    
    # Remove trailing 's' for plurals (but not for words like "class", "grass")
    if len(word) > 4 and word.endswith('s') and not word.endswith('ss'):
        word_singular = word[:-1]
        # Check if singular form is a known topic
        if word_singular in SEED_TOPICS:
            return word_singular
    
    # Remove "ing" suffix
    if len(word) > 5 and word.endswith('ing'):
        word_base = word[:-3]
        # Check if base form is known
        if word_base in SEED_TOPICS:
            return word_base
        # Try adding 'e' back (like "move" -> "moving")
        if word_base + 'e' in SEED_TOPICS:
            return word_base + 'e'
    
    return word


def find_matching_topics(text: str) -> List[str]:
    """
    Find all matching topics in text using smart matching.
    
    Strategy:
    1. Check exact matches in SEED_TOPICS
    2. Check normalized word matches
    3. Check synonym matches
    4. Remove duplicates and return
    
    Args:
        text: Text to search for topics (query + result summary)
        
    Returns:
        List of matched topic names (up to 3)
    """
    text_lower = text.lower()
    matched_topics = set()
    
    # Strategy 1: Check for exact topic matches
    for topic in SEED_TOPICS:
        if topic in text_lower:
            matched_topics.add(topic)
    
    # Strategy 2: Check normalized words
    words = text_lower.split()
    for word in words:
        normalized = normalize_word(word)
        if normalized in SEED_TOPICS and normalized not in matched_topics:
            matched_topics.add(normalized)
    
    # Strategy 3: Check synonyms
    for canonical_topic, synonyms in TOPIC_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in text_lower:
                matched_topics.add(canonical_topic)
                break  # Don't keep checking synonyms once we found one
    
    result = list(matched_topics)[:3]  # Limit to 3 topics
    return result


# ============================================================================
# PUBLIC API
# ============================================================================

def extract_topics_simple(
    query: str,
    tool_name: str,
    result_summary: str = ""
) -> List[str]:
    """
    Extract topics using keyword matching + smart synonyms.
    
    This is the ONLY topic extraction method now (no LLM).
    
    Args:
        query: The user query or tool invocation
        tool_name: Name of the tool used
        result_summary: Brief summary of result (optional)
    
    Returns:
        List of 1-3 relevant topics
    """
    # Combine query and summary for matching
    text_to_match = f"{query} {result_summary}".lower()
    
    # Find matching topics using smart matching
    matched_topics = find_matching_topics(text_to_match)
    
    return matched_topics


# ============================================================================
# STATISTICS & DIAGNOSTICS
# ============================================================================

def get_extraction_stats() -> dict:
    """Get statistics about topic extraction configuration."""
    total_synonyms = sum(len(syns) for syns in TOPIC_SYNONYMS.values())
    
    stats = {
        "seed_topics_count": len(SEED_TOPICS),
        "topics_with_synonyms": len(TOPIC_SYNONYMS),
        "total_synonym_variations": total_synonyms,
        "average_synonyms_per_topic": round(total_synonyms / max(len(TOPIC_SYNONYMS), 1), 1),
        "seed_topics": SEED_TOPICS[:10],  # First 10 for preview
        "example_synonyms": {
            "animation": list(TOPIC_SYNONYMS.get("animation", set()))[:5],
            "script": list(TOPIC_SYNONYMS.get("script", set()))[:5],
            "physics": list(TOPIC_SYNONYMS.get("physics", set()))[:5],
        }
    }
    
    logger.info(f"📊 [TopicExtractor] Statistics:")
    logger.info(f"  Seed topics: {stats['seed_topics_count']}")
    logger.info(f"  Topics with synonyms: {stats['topics_with_synonyms']}")
    logger.info(f"  Total variations: {stats['total_synonym_variations']}")
    
    return stats
