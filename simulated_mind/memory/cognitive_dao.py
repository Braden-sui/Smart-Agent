from typing import Dict, Any, List, Optional

from .types import MemoryType

class RWKV7CognitiveMemoryDAO:
    """A hierarchical memory system for the RWKV-7 agent."""

    def __init__(self, llm_client, mem0_client, usage_tracker):
        self.llm_client = llm_client
        self.mem0_client = mem0_client
        self.usage_tracker = usage_tracker
        # Per-user working memory: {user_id: {memory_type: [mems]}}
        self._memories: Dict[str, Dict[MemoryType, List[Dict[str, Any]]]] = {}

    def _ensure_user_memory(self, user_id: str):
        """Ensure the memory structure for a user exists."""
        if user_id not in self._memories:
            self._memories[user_id] = {mem_type: [] for mem_type in MemoryType}

    def add_memory(self, user_id: str, memory_type: MemoryType, content: str, metadata: Dict[str, Any] = None):
        """Add a memory to a specific hierarchical level for a given user."""
        self._ensure_user_memory(user_id)
        if metadata is None:
            metadata = {}

        if memory_type == MemoryType.WORKING:
            memory_item = {"content": content, "metadata": metadata}
            self._memories[user_id][MemoryType.WORKING].append(memory_item)
        else:
            metadata['type'] = memory_type.value
            # Assuming mem0_client methods need user_id. This might need adjustment.
            self.mem0_client.add(content, user_id=user_id, metadata=metadata)
            self.usage_tracker.track_add(memory_type=memory_type, size=len(content))

    def get_memories(self, user_id: str, memory_type: MemoryType, query: str, limit: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve memories from a specific hierarchical level for a given user."""
        self._ensure_user_memory(user_id)
        if memory_type == MemoryType.WORKING:
            results = [m for m in self._memories[user_id][MemoryType.WORKING] if query.lower() in m['content'].lower()]
            if metadata_filter:
                results = [r for r in results if all(r.get('metadata', {}).get(k) == v for k, v in metadata_filter.items())]
            return results[:limit]
        else:
            filters = {'type': memory_type.value}
            if metadata_filter:
                filters.update(metadata_filter)
            results = self.mem0_client.search(query, user_id=user_id, limit=limit, filters=filters)
            self.usage_tracker.track_search(memory_type=memory_type, query_length=len(query), results_count=len(results))
            return results

    def get_working_memory(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories currently in the working memory for a given user."""
        self._ensure_user_memory(user_id)
        return self._memories[user_id][MemoryType.WORKING]

    def clear_working_memory(self, user_id: str):
        """Clear the working memory for a given user."""
        self._ensure_user_memory(user_id)
        self._memories[user_id][MemoryType.WORKING] = []

    def consolidate_memories(self, user_id: str):
        """A background process to summarize working memory into episodic memory for a user.

        This process uses the LLM to generate a concise summary of the current
        working memory, which is then stored as a single episodic memory.
        This prevents clutter and creates higher-level abstractions of events.
        """
        working_memories = self.get_working_memory(user_id)
        if not working_memories:
            return

        # Combine content for summarization
        combined_content = "\n".join(m['content'] for m in working_memories)
        if not combined_content.strip():
            return

        # Use LLM to summarize
        prompt = f"Summarize the following observations into a concise memory:\n\n{combined_content}"
        summary = self.llm_client.complete_text(prompt, max_tokens=150)

        # Add the consolidated memory to episodic store
        self.add_memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content=summary,
            metadata={'source': 'consolidation'}
        )

        self.clear_working_memory(user_id)

    def distill_semantic_knowledge(self, user_id: str, time_window: str = "24h"):
        """A background process to distill episodic memories into semantic knowledge for a user.

        This process queries recent episodic memories, uses the LLM to extract
        key facts, concepts, or relationships, and stores them as structured
        semantic memories.
        """
        # 1. Get recent episodic memories (using a broad query as a stand-in for time-based search)
        episodic_memories = self.get_memories(user_id, MemoryType.EPISODIC, query="*", limit=10)
        if not episodic_memories:
            return

        # 2. Combine for distillation
        combined_content = "\n".join(m.get('content', '') for m in episodic_memories)
        if not combined_content.strip():
            return

        # 3. Use LLM to distill facts into a structured format
        prompt = (
            "Extract key facts, entities, and relationships from the following memories. "
            "Present the output as a JSON list of objects, where each object has 'subject', 'predicate', and 'object' keys.\n\n"
            f"Memories:\n{combined_content}"
        )
        distilled_json = self.llm_client.complete_text(prompt, max_tokens=500)

        # 4. Parse and store as semantic memories
        try:
            import json
            facts = json.loads(distilled_json)
            if not isinstance(facts, list):
                return  # Failed to get a list

            for fact in facts:
                if isinstance(fact, dict) and all(k in fact for k in ['subject', 'predicate', 'object']):
                    content = f"{fact['subject']} {fact['predicate']} {fact['object']}"
                    self.add_memory(
                        user_id=user_id,
                        memory_type=MemoryType.SEMANTIC,
                        content=content,
                        metadata={'source': 'distillation', 'fact': fact}
                    )
        except (json.JSONDecodeError, TypeError):
            # LLM output was not valid JSON, could log this failure.
            pass

    def find_memories_by_tags(self, user_id: str, tags: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Find memories by a list of tags."""
        return self.mem0_client.search_memories_by_tags(user_id=user_id, tags=tags, limit=limit)
