"""Verification script to test local embedding server integration."""

import asyncio
import httpx


async def verify_embedding_server():
    """Test the local embedding server is working correctly."""
    
    EMBED_SERVER_URL = "http://127.0.0.1:8766"
    
    print("=" * 60)
    print("Testing Local Embedding Server Integration")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n[Test 1] Checking server health...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{EMBED_SERVER_URL}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Server is ready: {health}")
                print(f"   Model: {health.get('model', 'unknown')}")
                print(f"   Dimension: {health.get('dim', 'unknown')}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nMake sure to start the embed server:")
        print("   npm run embed-server")
        return
    
    # Test 2: Single embedding
    print("\n[Test 2] Testing single text embedding...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{EMBED_SERVER_URL}/embed",
                json={
                    "input": "player movement controller",
                    "type": "query"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                print(f"✅ Generated embedding with {len(embedding)} dimensions")
                print(f"   First 5 values: {embedding[:5]}")
                
                # Verify dimension
                if len(embedding) == 384:
                    print("✅ Correct dimension (384) for bge-small-en-v1.5")
                else:
                    print(f"⚠️  Unexpected dimension: {len(embedding)}")
            else:
                print(f"❌ Embedding failed: {response.status_code}")
                print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Embedding request failed: {e}")
        return
    
    # Test 3: Batch embeddings
    print("\n[Test 3] Testing batch embeddings...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            test_queries = [
                "character movement script",
                "inventory system",
                "UI button handler"
            ]
            
            response = await client.post(
                f"{EMBED_SERVER_URL}/embed",
                json={
                    "input": test_queries,
                    "type": "query"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result.get("embeddings", [])
                print(f"✅ Generated {len(embeddings)} embeddings")
                for i, emb in enumerate(embeddings):
                    print(f"   Query {i+1}: {len(emb)} dimensions")
            else:
                print(f"❌ Batch embedding failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Batch embedding request failed: {e}")
    
    # Test 4: Similarity comparison
    print("\n[Test 4] Testing semantic similarity...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Embed two similar queries
            similar_queries = [
                "player movement controller",
                "character walking system"
            ]
            
            response = await client.post(
                f"{EMBED_SERVER_URL}/embed",
                json={
                    "input": similar_queries,
                    "type": "query"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result.get("embeddings", [])
                
                if len(embeddings) == 2:
                    # Calculate cosine similarity
                    import math
                    emb1, emb2 = embeddings[0], embeddings[1]
                    
                    dot_product = sum(a * b for a, b in zip(emb1, emb2))
                    mag1 = math.sqrt(sum(a * a for a in emb1))
                    mag2 = math.sqrt(sum(b * b for b in emb2))
                    
                    similarity = dot_product / (mag1 * mag2)
                    print(f"✅ Cosine similarity between similar queries: {similarity:.4f}")
                    
                    if similarity > 0.7:
                        print("✅ High similarity detected (good!)")
                    else:
                        print("⚠️  Lower similarity than expected")
    except Exception as e:
        print(f"❌ Similarity test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Embedding server verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_embedding_server())