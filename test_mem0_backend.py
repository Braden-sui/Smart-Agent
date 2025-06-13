"""Test script for Mem0 backend integration.

This script tests the Mem0 client with the actual backend API.
It verifies:
1. SDK connection and authentication
2. Memory CRUD operations
3. Error handling and cleanup
"""

import os
import sys
import uuid
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import pytest

# Try to import the mem0 SDK
try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    print("❌ mem0 SDK not installed. Please install it with: pip install mem0ai")
    sys.exit(1)

def test_mem0_backend():
    """Test Mem0 backend integration with the actual API."""
    print("\n=== Testing Mem0 Backend Integration ===")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    api_key = os.getenv("MEM0_API_KEY")
    org_id = os.getenv("MEM0_ORG_ID", "default_org")
    project_id = os.getenv("MEM0_PROJECT_ID", "default_project")
    
    if not api_key:
        print("❌ Error: MEM0_API_KEY not found in environment variables")
        print("Please add your Mem0 API key to the .env file")
        pytest.fail("❌ Error: MEM0_API_KEY not found in environment variables")
    
    print(f"\n🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    print(f"🏢 Organization ID: {org_id}")
    print(f"📂 Project ID: {project_id}")
    
    # Initialize the Mem0 client
    print("\n🔄 Initializing Mem0 client...")
    try:
        client = MemoryClient(
            api_key=api_key,
            org_id=org_id,
            project_id=project_id
        )
        print("✅ Successfully initialized Mem0 client")
    except Exception as e:
        pytest.fail(f"❌ Failed to initialize Mem0 client: {e}")
    
    # Test memory creation
    test_id = f"test_{str(uuid.uuid4())[:8]}"
    test_content = f"Test memory content for {test_id}"
    test_metadata = {"test_id": test_id, "purpose": "integration_test"}
    
    print(f"\n📝 Creating test memory with ID: {test_id}")
    try:
        # Create memory using the SDK's method
        # Note: The actual method might be different - check mem0 SDK docs
        memory = client.add(
            messages=[{"role": "system", "content": test_content}],
            user_id="integration_test_user",
            metadata=test_metadata,
            tags=["integration_test", "pytest"],
            version="v2",
        )
        
        if not memory or not hasattr(memory, 'id'):
            pytest.fail("❌ Failed to create test memory: Invalid response from API")
            
        memory_id = memory.id
        print(f"✅ Successfully created test memory with ID: {memory_id}")
        
        # Retrieve memory
        print(f"\n🔍 Retrieving test memory with ID: {memory_id}")
        retrieved = client.get(memory_id)
        
        if not retrieved:
            pytest.fail("❌ Failed to retrieve test memory")
            
        print("✅ Successfully retrieved test memory:")
        print(f"   ID: {getattr(retrieved, 'id', 'N/A')}")
        print(f"   Content: {getattr(retrieved, 'content', 'N/A')}")
        print(f"   Metadata: {getattr(retrieved, 'metadata', {})}")
        print(f"   Created at: {getattr(retrieved, 'created_at', 'N/A')}")
        
        # Verify content
        if getattr(retrieved, 'content', '') != test_content:
            pytest.fail("❌ Retrieved content does not match original")
            
        print("✅ Content verification passed")
        
    except Exception as e:
        from mem0.client.main import APIError
        if isinstance(e, APIError) and "Project with ID" in str(e):
            pytest.skip("Skipping backend test: MEM0_PROJECT_ID not configured for this environment.")
            return
        pytest.fail(str(e))
    finally:
        # Cleanup - delete test memory if it was created
        if 'memory_id' in locals():
            print(f"\n🧹 Cleaning up test memory: {memory_id}")
            try:
                if 'client' in locals():
                    client.delete(memory_id)
                    print("✅ Cleanup complete")
            except Exception as e:
                print(f"⚠️ Warning: Failed to clean up test memory: {e}")

if __name__ == "__main__":
    print("Starting Mem0 backend integration test...")
    test_mem0_backend()
    print("\n✅ Test completed successfully!")
    sys.exit(0)
