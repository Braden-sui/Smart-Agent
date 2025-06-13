from typing import Dict, Any, List
from enum import Enum

class MemoryType(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    META = "meta"

class RWKV7CognitiveMemoryDAO:
    """A hierarchical memory system for the RWKV-7 agent."""

    def __init__(self, mem0_client, usage_tracker):
        self.mem0_client = mem0_client
        self.usage_tracker = usage_tracker
        self._memories: Dict[MemoryType, List[Dict[str, Any]]] = {mem_type: [] for mem_type in MemoryType}

    def add_memory(self, memory_type: MemoryType, content: str, metadata: Dict[str, Any] = None):
        """Add a memory to a specific hierarchical level."""
        if metadata is None:
            metadata = {}

        if memory_type == MemoryType.WORKING:
            memory_item = {
                "content": content,
                "metadata": metadata
            }
            self._memories[MemoryType.WORKING].append(memory_item)
        else:
            # For other memory types, delegate to mem0_client
            metadata['type'] = memory_type.value
            self.mem0_client.add(content, metadata=metadata)
            self.usage_tracker.track_add(memory_type=memory_type, size=len(content))

    def get_memories(self, memory_type: MemoryType, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories from a specific hierarchical level."""
        if memory_type == MemoryType.WORKING:
            # Simple text matching for working memory, not a real search
            results = [m for m in self._memories[MemoryType.WORKING] if query.lower() in m['content'].lower()]
            return results[:limit]
        else:
            # For other memory types, delegate to mem0_client
            results = self.mem0_client.search(query, limit=limit, metadata={'type': memory_type.value})
            self.usage_tracker.track_search(memory_type=memory_type, query_length=len(query), results_count=len(results))
            return results

    def get_working_memory(self) -> List[Dict[str, Any]]:
        """Get all memories currently in the working memory."""
        return self._memories[MemoryType.WORKING]

    def clear_working_memory(self):
        """Clear the working memory."""
        self._memories[MemoryType.WORKING] = []

    def consolidate_memories(self):
        """A background process to move memories between levels.

        This simple implementation moves all memories from WORKING to EPISODIC.
        """
        working_memories = self.get_working_memory().copy()
        if not working_memories:
            return

        for memory in working_memories:
            self.add_memory(
                memory_type=MemoryType.EPISODIC,
                content=memory['content'],
                metadata=memory['metadata']
            )
        
        self.clear_working_memory()
