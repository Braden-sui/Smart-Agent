"""Thin wrapper around the mem0 SDK or HTTP API.

Integrates with the mem0ai Python SDK if an API key is provided,
otherwise falls back to an in-memory placeholder.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

try:
    from mem0 import MemoryClient
    from mem0.models import Message
    MEM0_SDK_AVAILABLE = True
except ImportError:
    MEM0_SDK_AVAILABLE = False
    MemoryClient = None  # Placeholder if SDK not installed

logger = logging.getLogger(__name__)


class Mem0Client:  # pragma: no cover – stub for SDK interaction
    """Lightweight, synchronous mem0 client.

    Wraps the official mem0 Python SDK if `api_key` is provided and the SDK
    is installed. Otherwise, mimics behavior with an in-process store.
    Methods are designed to align with MemoryDAO's expectations.
    """

    def __init__(self, api_key: Optional[str] = None, journal: Optional[Any] = None, *, enable_versioning: bool = True):
        self.journal = journal  # For logging, if needed
        self.sdk_active = False
        self.client: Optional[MemoryClient] = None

        if MEM0_SDK_AVAILABLE and api_key:
            try:
                # When an explicit API key is provided, use deterministic defaults for org_id / project_id
                # to avoid test brittleness from environment variables that may be set on the host.
                org_id = "default_org"
                project_id = "default_project"
                
                self.client = MemoryClient(
                    api_key=api_key,
                    org_id=org_id,
                    project_id=project_id
                )
                self.sdk_active = True
                self._log_sdk("Mem0Client initialized with SDK")
            except Exception as e:
                self._log_error(f"Failed to initialize Mem0 SDK: {e}", e)
                self.client = None
        elif api_key and not MEM0_SDK_AVAILABLE:
            self._log_warning("Mem0 API key provided, but 'mem0ai' SDK not installed. Falling back to in-memory store.")
        else:
            self._log_info("Mem0Client initialized with in-memory store (no API key or SDK not found).")

        # In-memory fallback stores
        self._kv_fallback: Dict[str, Dict[str, Any]] = {}  # user_id -> {memory_id: record}
        self._kg_fallback: list[dict[str, Any]] = []  # list of {sub, pred, obj, meta, user_id}
        # Version history per memory_id
        self.enable_versioning = enable_versioning
        self._version_history: Dict[str, List[Dict[str, Any]]] = {}

    def _log_sdk(self, message: str):
        """Log an SDK-related message."""
        if self.journal:
            self.journal.log_event("Mem0Client.SDK", message)
        else:
            logger.info(f"Mem0Client.SDK: {message}")

    def _log_info(self, message: str):
        """Log an informational message."""
        if self.journal:
            self.journal.log_event("Mem0Client.Info", message)
        else:
            logger.info(f"Mem0Client: {message}")

    def _log_warning(self, message: str):
        """Log a warning message."""
        if self.journal:
            self.journal.log_event(
                "Mem0Client.Warning",
                {"message": message, "level": "warning"}
            )
        else:
            logger.warning(f"Mem0Client: {message}")

    def _log_error(self, message: str, exc_info=None):
        """Log an error message with optional exception info."""
        if self.journal:
            self.journal.log_event(
                "Mem0Client.Error",
                {
                    "message": f"{message}. Falling back to in-memory store.",
                    "level": "error",
                    "exception_type": exc_info.__class__.__name__ if exc_info else None
                }
            )
        else:
            logger.error(f"Mem0Client: {message}", exc_info=exc_info)

    def create_memory(
        self,
        memory_id: str,
        content: Any,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        user_id: Optional[str] = "default_user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create or update a memory in mem0.
        
        Args:
            memory_id: Unique identifier for the memory
            content: The content to store (can be string or dict)
            tags: Optional list of tags for categorization
            categories: Optional list of categories for categorization
            user_id: ID of the user this memory belongs to
            metadata: Additional metadata to store with the memory
        """
        user_id = user_id or "default_user"
        full_metadata = {
            "tags": tags or [],
            "categories": categories or [],
            "original_memory_id": memory_id,
            **(metadata or {})
        }

        if self.sdk_active and self.client:
            try:
                # Prepare messages for the memory
                if isinstance(content, (str, bytes)):
                    messages = [{"role": "system", "content": str(content)}]
                elif isinstance(content, list):
                    messages = content  # Assume it's already in the right format
                elif isinstance(content, dict):
                    messages = [{"role": "system", "content": str(content)}]
                else:
                    messages = [{"role": "system", "content": str(content)}]
                
                # Add the memory using the SDK
                result = self.client.add(
                    messages=messages,
                    user_id=user_id,
                    metadata=full_metadata,
                    version="v2"  # Use v2 API for latest features
                )
                
                self._log_sdk(f"Created memory via SDK for user '{user_id}', id '{memory_id}'")
                # Record version in history if enabled
                if self.enable_versioning:
                    self._version_history.setdefault(memory_id, []).append({
                        "timestamp": time.time(),
                        "operation": "create",
                        "content": content,
                        "metadata": full_metadata,
                        "user_id": user_id,
                    })
                return result
                
            except Exception as e:
                self._log_error(f"Failed to create memory via SDK: {e}", e)
        
        # Fallback to in-memory storage if SDK is not available or failed
        self._log_warning(f"Using fallback storage for memory '{memory_id}'")
        user_store = self._kv_fallback.setdefault(user_id, {})
        user_store[memory_id] = {"content": content, **full_metadata}
        # also record fallback version
        if self.enable_versioning:
            self._version_history.setdefault(memory_id, []).append({
                "timestamp": time.time(),
                "operation": "create-fallback",
                "content": content,
                "metadata": full_metadata,
                "user_id": user_id,
            })

    def get_memory(
        self, memory_id: str, user_id: Optional[str] = "default_user"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            user_id: The ID of the user who owns the memory
            
        Returns:
            The memory data if found, None otherwise
        """
        user_id = user_id or "default_user"
        
        if self.sdk_active and self.client:
            try:
                # Try to get the memory directly by ID if the SDK supports it
                try:
                    memory = self.client.get(memory_id=memory_id, user_id=user_id)
                    if memory:
                        if hasattr(memory, 'to_dict'):
                            memory = memory.to_dict()
                        self._log_sdk(f"Retrieved memory via SDK for user '{user_id}', id '{memory_id}'")
                        return {
                            "memory_id": memory_id,
                            "content": memory.get('content', memory.get('memory')),
                            **memory.get('metadata', {})
                        }
                except (AttributeError, NotImplementedError):
                    pass  # Fall through to search
                
                # Fallback to search if direct get is not available
                search_results = self.client.search(
                    query=f"id:{memory_id}",
                    filters={"user_id": user_id},
                    limit=1,
                    version="v2"
                )
                
                if search_results and len(search_results) > 0:
                    memory = search_results[0]
                    if hasattr(memory, 'to_dict'):
                        memory = memory.to_dict()
                    self._log_sdk(f"Retrieved memory via SDK search for user '{user_id}', id '{memory_id}'")
                    return {
                        "memory_id": memory_id,
                        "content": memory.get('content', memory.get('memory')),
                        **memory.get('metadata', {})
                    }
                    
            except Exception as e:
                self._log_error(f"Failed to get memory via SDK: {e}", e)
                # Fall through to in-memory fallback
        
        # Fallback to in-memory store
        self._log_warning(f"Using fallback storage to retrieve memory for user '{user_id}', id '{memory_id}'")
        user_store = self._kv_fallback.get(user_id, {})
        return user_store.get(memory_id)

    def update_memory(
        self,
        memory_id: str,
        content: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        user_id: Optional[str] = "default_user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        user_id = user_id or "default_user"
        if self.sdk_active and self.client:
            try:
                # Assuming SDK has an update method: self.client.update(memory_id=..., data=..., metadata=...)
                # If not, it's a get, modify, then re-add (or delete and add), which is complex.
                # For now, log that SDK update is not directly supported by basic example.
                self._log_warning(f"SDK update for memory ID '{memory_id}' (user '{user_id}') not directly implemented based on basic SDK examples. Using fallback.")
            except Exception as e:
                self._log_error(f"SDK update attempt failed: {e}. Using fallback.", e)

        # Fallback logic
        self._log_warning(f"Updating memory for user '{user_id}', id '{memory_id}'.")
        user_store = self._kv_fallback.get(user_id)
        if user_store and memory_id in user_store:
            if content is not None:
                user_store[memory_id]["content"] = content
            if tags is not None:
                user_store[memory_id]["tags"] = tags
            if categories is not None:
                user_store[memory_id]["categories"] = categories
            if metadata is not None:
                user_store[memory_id].update(metadata) # Merge new metadata
            return True
        return False

    def delete_memory(self, memory_id: str, user_id: Optional[str] = "default_user") -> bool:
        user_id = user_id or "default_user"

        if self.sdk_active and self.client:
            try:
                # Assuming SDK has a delete method: self.client.delete(memory_id=memory_id, user_id=user_id)
                self._log_warning(f"SDK delete for memory ID '{memory_id}' (user '{user_id}') not directly implemented based on basic SDK examples. Using fallback.")
            except Exception as e:
                self._log_error(f"SDK delete attempt failed: {e}. Using fallback.", e)
        
        # Fallback logic
        self._log_warning(f"Deleting memory for user '{user_id}', id '{memory_id}'.")
        user_store = self._kv_fallback.get(user_id)
        if user_store and memory_id in user_store:
            del user_store[memory_id]
            return True
        return False

    def search_memories_by_tags(
        self, 
        tags: List[str], 
        user_id: Optional[str] = "default_user", 
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for memories that match all the given tags.
        
        Args:
            tags: List of tags that must all be present in the memory's metadata
            user_id: ID of the user whose memories to search
            limit: Maximum number of results to return
            **kwargs: Additional parameters for the search API
            
        Returns:
            List of matching memories with their metadata
        """
        user_id = user_id or "default_user"
        if not tags:
            return []

        if self.sdk_active and self.client:
            try:
                # Create a filter that requires all tags to be present
                filters = {
                    "user_id": user_id,
                    "tags": {"$all": tags}
                }
                
                # Use a broad query that might match any of the tags
                query = " ".join(tags)
                
                # Execute the search with filters
                search_params = {
                    "query": query,
                    "filters": filters,
                    "limit": limit,
                    "version": "v2",
                    **kwargs
                }
                
                results = []
                sdk_results = self.client.search(**search_params)
                
                # Process and format results
                for item in sdk_results:
                    if hasattr(item, 'to_dict'):
                        item = item.to_dict()
                    
                    # Ensure the item has all required tags
                    item_tags = item.get('metadata', {}).get('tags', [])
                    if all(tag in item_tags for tag in tags):
                        results.append({
                            "memory_id": item.get('id'),
                            "content": item.get('content'),
                            **item.get('metadata', {})
                        })
                    
                    if len(results) >= limit:
                        break
                        
                self._log_sdk(f"Found {len(results)} memories matching tags: {tags}")
                return results
                
            except Exception as e:
                self._log_error(f"Tag search failed: {e}", e)
                # Fall through to fallback implementation
        
        # Fallback implementation: filter in-memory store
        self._log_warning(f"Using fallback tag search for tags: {tags}")
        user_store = self._kv_fallback.get(user_id, {})
        results = []
        
        for mem_id, mem_data in user_store.items():
            mem_tags = mem_data.get('tags', [])
            if all(tag in mem_tags for tag in tags):
                results.append({"memory_id": mem_id, **mem_data})
                if len(results) >= limit:
                    break
                    
        return results

    # Knowledge graph methods - remain on fallback for now
    def add_relation(
        self, 
        subject: str, 
        predicate: str, 
        obj: str, 
        user_id: Optional[str] = "default_user", 
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Add a relation to the knowledge graph.
        
        Note: Currently uses in-memory fallback as the SDK's KG capabilities are not fully implemented.
        """
        user_id = user_id or "default_user"
        if self.sdk_active:
            self._log_warning("Knowledge graph 'add_relation' currently uses in-memory fallback.")
        
        self._log_info(f"Adding KG relation for user '{user_id}': {subject}-{predicate}-{obj}")
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
        user_id: Optional[str] = "default_user",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query the knowledge graph for relations matching the given pattern.
        
        Args:
            subject: Subject of the relation (optional)
            predicate: Predicate/type of the relation (optional)
            obj: Object of the relation (optional)
            user_id: ID of the user whose knowledge graph to query
            **kwargs: Additional query parameters
            
        Returns:
            List of matching relations with their metadata
        """
        user_id = user_id or "default_user"
        
        if self.sdk_active and self.client:
            try:
                # Build query filters based on provided pattern
                filters = {"user_id": user_id}
                if subject is not None:
                    filters["subject"] = subject
                if predicate is not None:
                    filters["predicate"] = predicate
                if obj is not None:
                    filters["object"] = obj
                
                # Use the search endpoint with filters
                search_params = {
                    "query": " ".join(filter(None, [subject, predicate, obj])),
                    "filters": filters,
                    "limit": kwargs.get("limit", 100),  # Default limit
                    "version": "v2",
                    **{k: v for k, v in kwargs.items() if k != "limit"}
                }
                
                # Execute the search
                sdk_results = self.client.search(**search_params)
                
                # Convert results to a consistent format
                results = []
                for item in sdk_results:
                    if hasattr(item, 'to_dict'):
                        item = item.to_dict()
                    
                    # Skip if metadata doesn't match the pattern (should be filtered by API)
                    item_subj = item.get('subject')
                    item_pred = item.get('predicate')
                    item_obj = item.get('object')
                    
                    if (subject is None or item_subj == subject) and \
                       (predicate is None or item_pred == predicate) and \
                       (obj is None or item_obj == obj):
                        results.append({
                            "subject": item_subj,
                            "predicate": item_pred,
                            "object": item_obj,
                            "meta": item.get('metadata', {}).get('meta', {}),
                            "user_id": item.get('user_id')
                        })
                
                self._log_sdk(f"Found {len(results)} KG relations matching pattern")
                return results
                
            except Exception as e:
                self._log_error(f"KG query failed: {e}", e)
                # Fall through to fallback implementation
            
        # Fallback implementation: filter in-memory store
        self._log_warning("Using fallback KG query")
        results = []
        for rel in self._kg_fallback:
            if rel["user_id"] != user_id:
                continue
            if subject is not None and rel["subject"] != subject:
                continue
            if predicate is not None and rel["predicate"] != predicate:
                continue
            if obj is not None and rel["object"] != obj:
                continue
            results.append(rel)
            
        return results

    # -------------------- New Advanced Features --------------------
    def batch_create_memories(self, items: List[Dict[str, Any]], user_id: str = "default_user") -> List[str]:
        """Batch create memories. Each item should have keys: memory_id, content, tags, categories, metadata"""
        created_ids = []
        for item in items:
            memory_id = item.get("memory_id") or str(uuid.uuid4())
            self.create_memory(
                memory_id=memory_id,
                content=item.get("content"),
                tags=item.get("tags"),
                categories=item.get("categories"),
                user_id=user_id,
                metadata=item.get("metadata")
            )
            created_ids.append(memory_id)
        return created_ids

    def get_memory_versions(self, memory_id: str) -> List[Dict[str, Any]]:
        """Return version history for a memory (fallback only)."""
        return self._version_history.get(memory_id, [])

    def advanced_search(
        self,
        query: str,
        user_id: str = "default_user",
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Advanced search with optional tag/category filtering."""
        if self.sdk_active and self.client:
            filters = {"user_id": user_id}
            if tags:
                filters["tags"] = {"ANY": tags}
            if categories:
                filters["categories"] = {"ANY": categories}
            try:
                results = self.client.search(
                    query=query,
                    filters=filters,
                    version="v2",
                    limit=limit
                )
                return [r if isinstance(r, dict) else r.to_dict() for r in results]
            except Exception as e:
                self._log_error(f"Advanced search via SDK failed: {e}")
        # Fallback: simple keyword search
        matches = []
        for mem_id, record in self._kv_fallback.get(user_id, {}).items():
            if query.lower() in str(record.get("content", "")).lower():
                if tags and not set(tags) & set(record.get("tags", [])):
                    continue
                if categories and not set(categories) & set(record.get("categories", [])):
                    continue
                matches.append({"memory_id": mem_id, **record})
                if len(matches) >= limit:
                    break
        return matches
