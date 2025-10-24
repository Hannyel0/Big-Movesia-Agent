"""Test script to verify token usage tracking is working correctly."""

import os
from dotenv import load_dotenv
from src.react_agent.utils import load_chat_model

# Load environment variables
load_dotenv()

def test_openai_token_tracking():
    """Test that OpenAI models return token usage metadata."""
    print("Testing OpenAI token tracking...")

    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY not set. Skipping OpenAI test.")
        return

    try:
        # Load OpenAI model
        model = load_chat_model("openai/gpt-4.1-mini-2025-04-14")

        # Make a simple invoke call
        response = model.invoke("Hello! Say hi back in 5 words.")

        # Check for usage metadata
        print(f"\n[SUCCESS] Response: {response.content}")
        print(f"\nUsage Metadata:")

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            print(f"   - Input tokens: {response.usage_metadata.get('input_tokens', 'N/A')}")
            print(f"   - Output tokens: {response.usage_metadata.get('output_tokens', 'N/A')}")
            print(f"   - Total tokens: {response.usage_metadata.get('total_tokens', 'N/A')}")
            print("\n[SUCCESS] OpenAI token tracking is WORKING!")
        elif hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            token_usage = response.response_metadata['token_usage']
            print(f"   - Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
            print(f"   - Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")
            print(f"   - Total tokens: {token_usage.get('total_tokens', 'N/A')}")
            print("\n[SUCCESS] OpenAI token tracking is WORKING!")
        else:
            print("\n[ERROR] No token usage metadata found!")
            print(f"Response metadata: {response.response_metadata if hasattr(response, 'response_metadata') else 'N/A'}")

    except Exception as e:
        print(f"\n[ERROR] Error testing OpenAI: {e}")


def test_anthropic_token_tracking():
    """Test that Anthropic models return token usage metadata."""
    print("\n\nTesting Anthropic token tracking...")

    # Check if ANTHROPIC_API_KEY is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[WARNING] ANTHROPIC_API_KEY not set. Skipping Anthropic test.")
        return

    try:
        # Load Anthropic model
        model = load_chat_model("anthropic/claude-3-5-sonnet-20240620")

        # Make a simple invoke call
        response = model.invoke("Hello! Say hi back in 5 words.")

        # Check for usage metadata
        print(f"\n[SUCCESS] Response: {response.content}")
        print(f"\nUsage Metadata:")

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            print(f"   - Input tokens: {response.usage_metadata.get('input_tokens', 'N/A')}")
            print(f"   - Output tokens: {response.usage_metadata.get('output_tokens', 'N/A')}")
            print(f"   - Total tokens: {response.usage_metadata.get('total_tokens', 'N/A')}")
            print("\n[SUCCESS] Anthropic token tracking is WORKING!")
        elif hasattr(response, 'response_metadata') and 'usage' in response.response_metadata:
            usage = response.response_metadata['usage']
            print(f"   - Input tokens: {usage.get('input_tokens', 'N/A')}")
            print(f"   - Output tokens: {usage.get('output_tokens', 'N/A')}")
            print(f"   - Total tokens: {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)}")
            print("\n[SUCCESS] Anthropic token tracking is WORKING!")
        else:
            print("\n[ERROR] No token usage metadata found!")
            print(f"Response metadata: {response.response_metadata if hasattr(response, 'response_metadata') else 'N/A'}")

    except Exception as e:
        print(f"\n[ERROR] Error testing Anthropic: {e}")


def test_streaming_token_tracking():
    """Test that streaming responses include token usage."""
    print("\n\nTesting Streaming token tracking...")

    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY not set. Skipping streaming test.")
        return

    try:
        # Load OpenAI model
        model = load_chat_model("openai/gpt-4.1-mini-2025-04-14")

        # Stream a response
        print("\n[INFO] Streaming response...")
        aggregate = None
        chunk_count = 0
        for chunk in model.stream("Count from 1 to 3, one number per message."):
            chunk_count += 1
            print(f"   Chunk {chunk_count}: content='{chunk.content}' has_usage={hasattr(chunk, 'usage_metadata') and bool(chunk.usage_metadata)}")
            aggregate = chunk if aggregate is None else aggregate + chunk

        # Check final aggregate for usage metadata
        print(f"\nAggregated Usage Metadata (after {chunk_count} chunks):")

        if hasattr(aggregate, 'usage_metadata') and aggregate.usage_metadata:
            print(f"   - Input tokens: {aggregate.usage_metadata.get('input_tokens', 'N/A')}")
            print(f"   - Output tokens: {aggregate.usage_metadata.get('output_tokens', 'N/A')}")
            print(f"   - Total tokens: {aggregate.usage_metadata.get('total_tokens', 'N/A')}")
            print("\n[SUCCESS] Streaming token tracking is WORKING!")
        elif hasattr(aggregate, 'response_metadata') and 'token_usage' in aggregate.response_metadata:
            token_usage = aggregate.response_metadata['token_usage']
            print(f"   - Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
            print(f"   - Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")
            print(f"   - Total tokens: {token_usage.get('total_tokens', 'N/A')}")
            print("\n[SUCCESS] Streaming token tracking is WORKING!")
        else:
            print("\n[ERROR] No token usage metadata found in streaming!")
            print(f"Response metadata: {aggregate.response_metadata if hasattr(aggregate, 'response_metadata') else 'N/A'}")

    except Exception as e:
        print(f"\n[ERROR] Error testing streaming: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("Token Usage Tracking Test Suite")
    print("=" * 80)

    test_openai_token_tracking()
    test_anthropic_token_tracking()
    test_streaming_token_tracking()

    print("\n" + "=" * 80)
    print("Test suite completed!")
    print("=" * 80)
