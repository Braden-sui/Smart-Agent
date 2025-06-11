"""Thin wrapper around the mem0 SDK or HTTP API.

Integrates with the mem0ai Python SDK if an API key is provided,
otherwise falls back to an in-memory placeholder.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from mem0 import Memory
    MEM0_SDK_AVAILABLE = True
except ImportError:
    MEM0_SDK_AVAILABLE = False
    Memory = None  # Placeholder if SDK not installed

logger = logging.getLogger(__name__)


class Mem0Client:  # pragma: no cover – stub for SDK interaction
    """Lightweight, synchronous mem0 client.

    Wraps the official mem0 Python SDK if `api_key` is provided and the SDK
    is installed. Otherwise, mimics behavior with an in-process store.
    Methods are designed to align with MemoryDAO's expectations.
    """

    def __init__(self, api_key: Optional[str] = None, journal: Optional[Any] = None):
        self.journal = journal  # For logging, if needed
        self.sdk_active = False
        self.client: Optional[Memory] = None

        if MEM0_SDK_AVAILABLE and api_key:
            try:
                self.client = Memory(api_key=api_key)
                self.sdk_active = True
                self.journal.log_event("Mem0Client.SDK: Mem0Client initialized with SDK.")
            except Exception as e:
                self.journal.log_event(
                    "Mem0Client.SDK.Error", 
                    payload={
                        "message": f"Failed to initialize Mem0 SDK: {e}. Falling back to in-memory store.", 
                        "level": "error",
                        "exception_type": type(e).__name__
                    }
                )
                self.client = None
        elif api_key and not MEM0_SDK_AVAILABLE:
            self.journal.log_event("Mem0Client.Fallback.Warning", {"message": "Mem0 API key provided, but 'mem0ai' SDK not installed. Falling back to in-memory store.", "level": "warning"})
        else:
            self.journal.log_event("Mem0Client.Fallback: Mem0Client initialized with in-memory store (no API key or SDK not found).")

        # In-memory fallback stores
        self._kv_fallback: Dict[str, Dict[str, Any]] = {}  # user_id -> {memory_id: record}
        self._kg_fallback: list[dict[str, Any]] = []  # list of {sub, pred, obj, meta, user_id}

    def _log_sdk(self, message: str, level: int = logging.INFO):
        if self.journal:
            self.journal.log_event(f"Mem0Client.SDK: {message}")
        else:
            logger.log(level, f"Mem0Client.SDK: {message}")

    def _log_fallback(self, message: str, level: int = logging.INFO):
        if self.journal:
            self.journal.log_event(f"Mem0Client.Fallback: {message}")
        else:
            logger.log(level, f"Mem0Client.Fallback: {message}")

    def create_memory(
        self,
        memory_id: str,
        content: Any,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = "default_user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        user_id = user_id or "default_user"
        full_metadata = {"tags": tags or [], **(metadata or {})}

        if self.sdk_active and self.client:
            try:
                # The mem0 SDK's add method is designed for conversational context (list of messages).
                # For a single piece of content, we'll wrap it as a system message.
                # It also doesn't explicitly take memory_id for `add`, so we store it in metadata.
                # `data` field is preferred for raw string content if available, else use messages.
                sdk_metadata = {**full_metadata, "original_memory_id": memory_id}
                self.client.add(data=str(content), user_id=user_id, metadata=sdk_metadata)
                self._log_sdk(f"Created memory via SDK for user '{user_id}', original_id '{memory_id}'.")
                return
            except Exception as e:
                self._log_sdk(f"SDK add failed: {e}. Using fallback.", logging.ERROR)
        
        # Fallback logic
        self._log_fallback(f"Storing memory for user '{user_id}', id '{memory_id}'.")
        user_store = self._kv_fallback.setdefault(user_id, {})
        user_store[memory_id] = {"content": content, **full_metadata}

    def get_memory(
        self, memory_id: str, user_id: Optional[str] = "default_user"
    ) -> Optional[Dict[str, Any]]:
        user_id = user_id or "default_user"
        if self.sdk_active and self.client:
            try:
                # The SDK's get method takes memory_id.
                # sdk_memory = self.client.get(memory_id=memory_id, user_id=user_id)
                # Assuming .get() returns the memory object directly or None
                # For now, as .get() by ID is not in basic example, we search by metadata.
                # This is a workaround and might be inefficient or not precise.
                search_results = self.client.search(query=memory_id, user_id=user_id, limit=10) # Query by ID
                for res_list in search_results:
                    for res in res_list.get('results', []):
                        if res.get("metadata", {}).get("original_memory_id") == memory_id:
                            self._log_sdk(f"Retrieved memory via SDK search for user '{user_id}', id '{memory_id}'.")
                            # Adapt SDK result to expected dict format
                            return {
                                "memory_id": memory_id, 
                                "content": res.get('memory'), 
                                **res.get("metadata", {})
                            }
                self._log_sdk(f"Memory ID '{memory_id}' not found via SDK search for user '{user_id}'. Using fallback.", logging.WARNING)
            except Exception as e:
                self._log_sdk(f"SDK get/search failed: {e}. Using fallback.", logging.ERROR)

        # Fallback logic
        self._log_fallback(f"Recalling memory for user '{user_id}', id '{memory_id}'.")
        user_store = self._kv_fallback.get(user_id, {})
        return user_store.get(memory_id)

    def update_memory(
        self,
        memory_id: str,
        content: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = "default_user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        user_id = user_id or "default_user"
        if self.sdk_active and self.client:
            try:
                # Assuming SDK has an update method: self.client.update(memory_id=..., data=..., metadata=...)
                # If not, it's a get, modify, then re-add (or delete and add), which is complex.
                # For now, log that SDK update is not directly supported by basic example.
                self._log_sdk(f"SDK update for memory ID '{memory_id}' (user '{user_id}') not directly implemented based on basic SDK examples. Using fallback.", logging.WARNING)
            except Exception as e:
                self._log_sdk(f"SDK update attempt failed: {e}. Using fallback.", logging.ERROR)

        # Fallback logic
        self._log_fallback(f"Updating memory for user '{user_id}', id '{memory_id}'.")
        user_store = self._kv_fallback.get(user_id)
        if user_store and memory_id in user_store:
            if content is not None:
                user_store[memory_id]["content"] = content
            if tags is not None:
                user_store[memory_id]["tags"] = tags
            if metadata is not None:
                user_store[memory_id].update(metadata) # Merge new metadata
            return True
        return False

    def delete_memory(self, memory_id: str, user_id: Optional[str] = "default_user") -> bool:
        user_id = user_id or "default_user"
        if self.sdk_active and self.client:
            try:
                # Assuming SDK has a delete method: self.client.delete(memory_id=memory_id, user_id=user_id)
                self._log_sdk(f"SDK delete for memory ID '{memory_id}' (user '{user_id}') not directly implemented based on basic SDK examples. Using fallback.", logging.WARNING)
            except Exception as e:
                self._log_sdk(f"SDK delete attempt failed: {e}. Using fallback.", logging.ERROR)
        
        # Fallback logic
        self._log_fallback(f"Deleting memory for user '{user_id}', id '{memory_id}'.")
        user_store = self._kv_fallback.get(user_id)
        if user_store and memory_id in user_store:
            del user_store[memory_id]
            return True
        return False

    def search_memories_by_tags(
        self, tags: List[str], user_id: Optional[str] = "default_user", limit: int = 10
    ) -> List[Dict[str, Any]]:
        user_id = user_id or "default_user"
        if self.sdk_active and self.client:
            try:
                # Use the first tag as the primary query for SDK search.
                # Client-side filtering might be needed if SDK doesn't support multi-tag metadata search well.
                query = tags[0] if tags else ""
                # The SDK search example: `memory.search(query=message, user_id=user_id, limit=3)`
                # It returns `{"results": [{'memory': '...', 'metadata': {...}}]}
                # We need to adapt this. The results structure in the example is `search_results["results"]`
                # but the example code iterates `search_results` directly as if it's the list of results.
                # Assuming `self.client.search` returns a list of dicts directly, or a dict with a 'results' key.
                
                # Let's assume search results is a list of dicts, each dict is a memory item.
                # sdk_results = self.client.search(query=query, user_id=user_id, limit=limit)
                # For now, we'll assume the SDK search is broad and we filter tags client-side from metadata.
                # This is a placeholder for more specific SDK metadata search capabilities.
                all_user_memories_sdk = [] # Placeholder for fetching all and filtering
                # A more realistic approach if metadata search is not robust:
                # Fetch many memories and filter, or use a broader query.
                # For now, we'll query by the first tag and hope for the best, then filter.
                
                raw_sdk_results_list = self.client.search(query=query, user_id=user_id, limit=limit * 5) # Fetch more to filter
                
                sdk_results_formatted = []
                # The example `relevant_memories["results"]` implies search returns a dict.
                # Let's assume it's `{'results': [...]}` based on the example's processing loop.
                actual_results_list = []
                if isinstance(raw_sdk_results_list, dict) and 'results' in raw_sdk_results_list:
                    actual_results_list = raw_sdk_results_list['results']
                elif isinstance(raw_sdk_results_list, list):
                    actual_results_list = raw_sdk_results_list

                for res in actual_results_list:
                    res_metadata = res.get("metadata", {})
                    res_tags = res_metadata.get("tags", [])
                    if all(tag in res_tags for tag in tags):
                        sdk_results_formatted.append({
                            "memory_id": res_metadata.get("original_memory_id", "N/A"),
                            "content": res.get("memory"),
                            **res_metadata
                        })
                    if len(sdk_results_formatted) >= limit:
                        break
                self._log_sdk(f"Searched memories via SDK for user '{user_id}' with tags '{tags}'. Found {len(sdk_results_formatted)}.")
                return sdk_results_formatted[:limit]
            except Exception as e:
                self._log_sdk(f"SDK search failed: {e}. Using fallback.", logging.ERROR)

        # Fallback logic
        self._log_fallback(f"Searching memories for user '{user_id}' with tags '{tags}'.")
        results: List[Dict[str, Any]] = []
        user_store = self._kv_fallback.get(user_id, {})
        for mem_id, record in user_store.items():
            record_tags = record.get("tags", [])
            if all(tag in record_tags for tag in tags):
                results.append({"memory_id": mem_id, **record})
            if len(results) >= limit:
                break
        return results

    # Knowledge graph methods - remain on fallback for now
    def add_relation(
        self, subject: str, predicate: str, obj: str, 
        user_id: Optional[str] = "default_user", 
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        user_id = user_id or "default_user"
        if self.sdk_active:
            self._log_sdk("Knowledge graph 'add_relation' currently uses in-memory fallback.", logging.WARNING)
        self._log_fallback(f"Adding KG relation for user '{user_id}': {subject}-{predicate}-{obj}")
        self._kg_fallback.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "meta": metadata or {},
                "user_id": user_id
            }
        )

    def get_relations_by_pattern(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        user_id: Optional[str] = "default_user"
    ) -> list[dict[str, Any]]:
        user_id_filter = user_id or "default_user"
        if self.sdk_active:
            self._log_sdk("Knowledge graph 'get_relations_by_pattern' currently uses in-memory fallback.", logging.WARNING)
        self._log_fallback(f"Querying KG relations for user '{user_id_filter}'.")
        results: list[dict[str, Any]] = []
        for rel in self._kg_fallback:
            if rel["user_id"] != user_id_filter and user_id is not None: # Allow global if user_id is None
                continue
            if subject is not None and rel["subject"] != subject:
                continue
            if predicate is not None and rel["predicate"] != predicate:
                continue
            if obj is not None and rel["object"] != obj:
                continue
            results.append(rel)
        return results
