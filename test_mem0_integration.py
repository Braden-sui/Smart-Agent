"""Integration test script for Mem0 client.

This script tests the basic functionality of the Mem0 client, including:
- Creating memories
- Retrieving memories
- Error handling
"""

from dotenv import load_dotenv
from simulated_mind.memory.mem0_client import Mem0Client

def test_mem0_connection():
    """Test the Mem0 client connection and basic operations."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize client
        print("Initializing Mem0 client...")
        client = Mem0Client()
        
        # Test creating a memory
        test_content = "Test memory from integration test"
        memory_id = "test_memory_1"
        print(f"Creating test memory with ID: {memory_id}")
        
        client.create_memory(
            memory_id=memory_id,
            content=test_content,
            tags=["test", "integration"],
            metadata={"source": "integration_test"}
        )
        print("✓ Successfully created test memory")
        
        # Test retrieving the memory
        print(f"Retrieving test memory with ID: {memory_id}")
        memory = client.get_memory(memory_id=memory_id)
        
        if memory and memory.get("content") == test_content:
            print("✓ Successfully retrieved test memory")
            print(f"Memory content: {memory.get('content')}")
            print(f"Memory metadata: {memory.get('metadata', {})}")
        else:
            print("✗ Failed to retrieve test memory or content mismatch")
        
        print("\nMem0 integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during Mem0 integration test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mem0_connection()
