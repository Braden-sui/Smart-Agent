from typing import Dict, Any, Optional
from .local_llm_client import RWKV7GGUFClient

class RWKV7OptimizedClient(RWKV7GGUFClient):
    """An optimized RWKV-7 client with state caching and other performance enhancements."""

    def __init__(self, model_path: str, context_size: int = 8192, cache_size: int = 10):
        super().__init__(model_path, context_size)
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        # Maintain insertion order for simple FIFO eviction expected by tests
        self.cache_keys: list[str] = []
        self.cache_size = cache_size

    def get_state(self, state_key: Optional[str] = None) -> Dict[str, Any]:
        """Get the current model state, using a cache if a key is provided."""
        if state_key and state_key in self.state_cache:
            # Move key to the end to represent recent access (optional LRU future)
            if state_key in self.cache_keys:
                self.cache_keys.remove(state_key)
                self.cache_keys.append(state_key)
            return self.state_cache[state_key].copy()
        
        return super().get_state()

    def set_state(self, state: Dict[str, Any], state_key: Optional[str] = None):
        """Set the model state, caching it if a key is provided."""
        super().set_state(state)
        if state_key:
            # Evict oldest if cache limit reached
            if len(self.cache_keys) >= self.cache_size:
                oldest_key = self.cache_keys.pop(0)
                self.state_cache.pop(oldest_key, None)

            # Store new key
            self.state_cache[state_key] = state.copy()
            if state_key in self.cache_keys:
                self.cache_keys.remove(state_key)
            self.cache_keys.append(state_key)
