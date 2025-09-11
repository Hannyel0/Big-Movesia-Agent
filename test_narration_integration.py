"""Test script to verify the narration integration changes."""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from react_agent.graph import graph
from react_agent.context import Context

# Load environment variables
load_dotenv()

async def test_narration_integration():
    """Test the integration of narration with tool calls."""
    print("Testing Narration Integration...")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "Search for Unity best practices for player movement",
        "Create a simple player controller script for Unity",
        "How do I set up a first person controller in Unity?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        
        try:
            # Create context with verbose narration
            context = Context(
                model="gpt-4o-mini",  # Using a lightweight model for testing
                verbose_narration=True
            )
            
            # Invoke the graph
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                {"context": context}
            )
            
            # Analyze the messages to verify integration
            print(f"✅ Query processed successfully")
            
            # Check for AIMessages with both content and tool_calls
            ai_messages_with_tools = 0
            for msg in result.get("messages", []):
                if hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage':
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        if msg.content and msg.content.strip():
                            ai_messages_with_tools += 1
                            print(f"  Found AIMessage with both narration and tool call:")
                            print(f"    Narration: {msg.content[:100]}...")
                            print(f"    Tool calls: {len(msg.tool_calls)} tool(s)")
            
            if ai_messages_with_tools > 0:
                print(f"✅ Found {ai_messages_with_tools} message(s) with integrated narration and tool calls")
            else:
                print(f"⚠️  No integrated messages found (might be due to query type)")
            
            # Small delay between tests
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Testing Complete!")

async def test_message_structure():
    """Specifically test the message structure in act() function."""
    print("\nTesting Message Structure in act()...")
    print("=" * 50)
    
    try:
        context = Context(model="gpt-4o-mini")
        
        # Simple query that will trigger tool usage
        query = "Get information about my Unity project"
        
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            {"context": context}
        )
        
        # Examine the structure
        print("Message Analysis:")
        for i, msg in enumerate(result.get("messages", [])):
            msg_type = msg.__class__.__name__ if hasattr(msg, '__class__') else type(msg).__name__
            print(f"\nMessage {i+1} ({msg_type}):")
            
            if hasattr(msg, 'content'):
                content = msg.content or ""
                if content:
                    print(f"  Content: {content[:150]}...")
            
            if hasattr(msg, 'tool_calls'):
                if msg.tool_calls:
                    print(f"  Tool Calls: {len(msg.tool_calls)}")
                    for tool_call in msg.tool_calls:
                        if hasattr(tool_call, 'name'):
                            print(f"    - {tool_call.name}")
                        elif isinstance(tool_call, dict) and 'name' in tool_call:
                            print(f"    - {tool_call['name']}")
        
        print("\n✅ Message structure test complete")
        
    except Exception as e:
        print(f"❌ Error in message structure test: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests."""
    print("\n🚀 Starting Integration Tests\n")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Run tests
    await test_narration_integration()
    await test_message_structure()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
