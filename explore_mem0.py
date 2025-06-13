"""Explore the mem0 SDK interactively."""
import os
from dotenv import load_dotenv
from mem0 import MemoryClient

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("MEM0_API_KEY")
    print(f"API Key found: {'Yes' if api_key else 'No'}")

    # Initialize the client with API key
    try:
        client = MemoryClient(api_key=api_key)
        print("Client initialized successfully!")
        
        # List available methods
        methods = [m for m in dir(client) if not m.startswith('_') and callable(getattr(client, m))]
        print("\nAvailable methods:")
        for method in sorted(methods):
            print(f"- {method}()")
        
        # Try to add a memory with the correct format
        try:
            print("\nAttempting to add a memory...")
            messages = [
                {"role": "user", "content": "Test memory content"},
                {"role": "assistant", "content": "I've noted this test memory."}
            ]
            result = client.add(messages, user_id="test_user")
            print(f"Add result: {result}")
            
            # Try to search for the memory
            print("\nAttempting to search memories...")
            search_results = client.search("test memory", user_id="test_user")
            print(f"Search results: {search_results}")
            
            # Get all memories
            print("\nGetting all memories...")
            all_memories = client.get_all(user_id="test_user")
            print(f"Found {len(all_memories)} memories")
            
        except Exception as e:
            print(f"Error during memory operations: {e}")

    except Exception as e:
        print(f"Error initializing client: {e}")

if __name__ == "__main__":
    main()
