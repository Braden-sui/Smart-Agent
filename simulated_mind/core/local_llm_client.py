from __future__ import annotations
from abc import ABC, abstractmethod 
from typing import Optional 

class LocalLLMClient(ABC):
    """Abstract base class for all local LLM clients used in simulated_mind.core."""

    @abstractmethod
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a completion for the given prompt."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying model is loaded and ready."""
        raise NotImplementedError

    # Optional hooks implemented by concrete subclasses
    def load(self) -> None:  # pragma: no cover
        pass

    def reset_conversation(self) -> None:  # pragma: no cover
        pass

class TransformersLocalLLMClient(LocalLLMClient):
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self._load_model()
    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            print("Transformers library not available. Install with: pip install transformers torch")
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.is_available():
            raise RuntimeError("Local model not available")
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

import os
import json
import copy
import threading
from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime
from simulated_mind.journal.journal import Journal



@dataclass(frozen=True)
class StateSnapshot:
    """
    An immutable snapshot of the LLM client's state at a point in time.
    All collections are tuples to enforce immutability.
    """
    version: int = 0
    conversation_history: Tuple[Dict[str, str], ...] = field(default_factory=tuple)
    context_tokens: Tuple[Any, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def with_update(self, **changes) -> 'StateSnapshot':
        """
        Creates a new StateSnapshot with updated values and an incremented version.
        """
        if 'version' in changes:
            raise ValueError("Version cannot be updated directly. It is auto-incremented.")

        # Make metadata changes on a copy
        if 'metadata' in changes:
            new_meta = self.metadata.copy()
            new_meta.update(changes['metadata'])
            changes['metadata'] = new_meta
        
        new_version = self.version + 1
        return replace(self, version=new_version, timestamp=datetime.utcnow(), **changes)

class ThreadSafeStateManager:
    """
    Manages state for an LLM client in a thread-safe and observable manner.
    """
    def __init__(self, max_history: int = 100, journal: Optional[Journal] = None):
        self._lock = threading.RLock()
        self._current: StateSnapshot = StateSnapshot()
        self._max_history = max_history
        self._journal = journal

    def get_current_snapshot(self) -> StateSnapshot:
        """Returns the current state snapshot. It is immutable and safe to share."""
        with self._lock:
            return self._current

    def atomic_update(self, updater: Callable[[StateSnapshot], StateSnapshot]) -> StateSnapshot:
        """
        Atomically updates the state using a provided function. The updater receives
        the current state and must return a new, updated StateSnapshot.
        """
        with self._lock:
            try:
                new_state = updater(self._current)
                if not isinstance(new_state, StateSnapshot):
                    error = TypeError("Updater function must return a StateSnapshot instance.")
                    if self._journal:
                        self._journal.log_event("state_manager:error", {"reason": "InvalidStateSnapshot", "details": str(error)})
                    raise error
            except Exception as e:
                if self._journal:
                    self._journal.log_event("state_manager:error", {"reason": "UpdaterFunctionFailed", "details": str(e)})
                raise

            # Enforce memory bounds on conversation history
            if len(new_state.conversation_history) > self._max_history:
                trimmed_history = new_state.conversation_history[-self._max_history:]
                if self._journal:
                    self._journal.log_event("state_trim", {
                        "trimmed_from": len(new_state.conversation_history),
                        "trimmed_to": len(trimmed_history)
                    })
                new_state = new_state.with_update(conversation_history=trimmed_history)
            
            self._current = new_state
            if self._journal:
                self._journal.log_event("state_update_success", {
                    "new_version": self._current.version,
                    "timestamp": self._current.timestamp.isoformat()
                })
            return self._current

    def set_snapshot(self, new_snapshot: StateSnapshot):
        """
        Directly sets the current state to a new snapshot, bypassing the updater.
        Useful for loading state from an external source.
        """
        if not isinstance(new_snapshot, StateSnapshot):
            error = TypeError("Provided state must be a StateSnapshot instance.")
            if self._journal:
                self._journal.log_event("state_manager:error", {"reason": "InvalidSetSnapshot", "details": str(error)})
            raise error
        with self._lock:
            self._current = new_snapshot
            if self._journal:
                self._journal.log_event("state_set_success", {
                    "set_to_version": self._current.version,
                    "timestamp": self._current.timestamp.isoformat()
                })

class RWKV7GGUFClient(LocalLLMClient):
    """RWKV-7 GGUF implementation for efficient local AI interaction with state support."""
    
    def __init__(self, model_path: str, context_size: int = 8192, max_history: int = 100, journal: Optional[Journal] = None):
        self.model_path = model_path
        self.context_size = context_size
        self.model: Optional[Llama] = None
        self.journal = journal
        
        # New thread-safe and memory-bounded state manager with journaling
        self._state_manager = ThreadSafeStateManager(max_history=max_history, journal=self.journal)
    
    def load(self):
        """Explicitly load the GGUF model."""
        if self.model is not None:
            print("Model already loaded.")
            return
        self._load_model()

    def _load_model(self):
        from llama_cpp import Llama
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=0,  # Set to >0 if CUDA/Metal is available and desired
                verbose=False
            )
            print(f"✅ RWKV-7 GGUF loaded: {os.path.basename(self.model_path)}")
            
            # State is managed by ThreadSafeStateManager, initialized with an empty state.
            
        except Exception as e:
            if isinstance(e, FileNotFoundError) or "No such file or directory" in str(e):
                print(f"❌ Failed to load RWKV-7 GGUF: Model file not found at {self.model_path}. Please ensure the path is correct and the model is downloaded.")
            else:
                print(f"❌ Failed to load RWKV-7 GGUF: {e}")
            self.model = None

    def _ensure_model_loaded(self):
        """Check if the model is loaded, and if not, load it."""
        if self.model is None:
            self.load()
    
    def atomic_state_update(self, updater: Callable[[StateSnapshot], StateSnapshot]):
        """
        Perform a thread-safe, atomic update to the client's state.

        Args:
            updater: A function that takes the current immutable StateSnapshot
                     and returns a new, updated StateSnapshot.
        """
        self._state_manager.atomic_update(updater)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current model state for GoT engine compatibility.
        
        Returns a dictionary representation of the immutable StateSnapshot.
        NOTE: This method is for API compatibility. For atomic updates, use
        the `atomic_state_update` method to avoid race conditions.
        """
        snapshot = self._state_manager.get_current_snapshot()
        
        # Convert immutable snapshot to a mutable dict for legacy compatibility
        history = list(snapshot.conversation_history)
        return {
            "version": snapshot.version,
            "conversation_history": history,
            "context_tokens": list(snapshot.context_tokens),
            "metadata": copy.deepcopy(snapshot.metadata),
            "timestamp": snapshot.timestamp.isoformat(),
            # Legacy fields for compatibility
            "step_count": len(history),
            "last_response": history[-1]['content'] if history else ""
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the model state from a dictionary.
        
        NOTE: This is a hard reset of the state. For atomic read-modify-write
        operations, use `atomic_state_update`.
        """
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")

        # Convert dict to StateSnapshot, providing defaults for missing keys
        current_snapshot = self._state_manager.get_current_snapshot()
        new_snapshot = StateSnapshot(
            version=state.get("version", current_snapshot.version + 1),
            conversation_history=tuple(state.get("conversation_history", [])),
            context_tokens=tuple(state.get("context_tokens", [])),
            metadata=copy.deepcopy(state.get("metadata", {})),
        )
        self._state_manager.set_snapshot(new_snapshot)
        
        # Store state in history for debugging/rollback
        self._state_history.append(state.copy())
    
    def save_state(self, filepath: str) -> bool:
        """Save current state to file."""
        try:
            current_state = self.get_state()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(current_state, f, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ Could not save state to {filepath}: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """Load state from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
                self.set_state(state)
                print(f"📖 State loaded from {filepath}")
                return True
            return False
        except Exception as e:
            print(f"⚠️ Could not load state: {e}")
            return False
    
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        self._ensure_model_loaded()
        if not self.is_available():
            raise RuntimeError("RWKV-7 GGUF model not available or not loaded.")
        
        try:
            full_prompt = self._build_minimal_prompt(prompt)
            
            response = self.model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.85,
                stop=["Human:", "AI:", "\nHuman:", "\nAI:", "\n\n"],
                echo=False
            )
            
            response_text = response['choices'][0]['text'].strip()
            
            # Update conversation history and state
            self._update_conversation_history(prompt, response_text)
            self._update_state_after_generation(prompt, response_text)
            
            return response_text
            
        except Exception as e:
            print(f"❌ RWKV generation failed: {e}")
            raise
    
    def _update_state_after_generation(self, prompt: str, response: str):
        """Update internal state after text generation."""
        def updater(current_snapshot: StateSnapshot) -> StateSnapshot:
            new_context_tokens = current_snapshot.context_tokens + (prompt, response)
            return current_snapshot.with_update(context_tokens=new_context_tokens)

        self.atomic_state_update(updater)
    
    def _build_minimal_prompt(self, base_prompt: str) -> str:
        """Minimal prompt providing context without role constraints."""
        context = ""
        snapshot = self._state_manager.get_current_snapshot()
        if snapshot.conversation_history:
            recent_history = snapshot.conversation_history[-2:] 
            for exchange in recent_history:
                context += f"Human: {exchange['user'].strip()}\nAI: {exchange['assistant'].strip()}\n\n"
        
        return f"{context}Human: {base_prompt.strip()}\nAI:"
    
    def _update_conversation_history(self, user_input: str, assistant_response: str):
        """Atomically updates the conversation history in the state."""
        new_turns = (
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        )

        def updater(current_snapshot: StateSnapshot) -> StateSnapshot:
            new_history = current_snapshot.conversation_history + new_turns
            return current_snapshot.with_update(conversation_history=new_history)

        self._state_manager.atomic_update(updater)
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def reset_conversation(self):
        """Resets the state to a new, empty StateSnapshot."""
        self._state_manager.set_snapshot(StateSnapshot())
        print("🔄 Conversation history and state reset")

    
    def load_conversation_state(self, filepath: str):
        """Legacy method - now delegates to load_state."""
        return self.load_state(filepath)

# -----------------------------------------------------------------------------
# Optional llama_cpp import for GGUF backend – provide dummy fallback for tests
# -----------------------------------------------------------------------------
try:
    from llama_cpp import Llama as _Llama
except Exception:  # pragma: no cover – fallback if llama_cpp unavailable
    class _Llama:  # type: ignore
        """Dummy replacement so unit tests can monkeypatch ``Llama`` safely."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("llama_cpp not installed – dummy class used")

# Expose as module-level symbol expected by unit tests
Llama = _Llama

def create_local_llm_client(backend: str = "rwkv7-gguf", **kwargs) -> "LocalLLMClient":
    if backend == "transformers":
        # Ensure TransformersLocalLLMClient is defined or imported correctly
        return TransformersLocalLLMClient(**kwargs) 
    elif backend == "rwkv7-gguf":
        # RWKV7GGUFClient expects 'model_path' in kwargs
        if 'model_path' not in kwargs:
            raise ValueError("model_path is required for rwkv7-gguf backend")
        return RWKV7GGUFClient(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported backends: 'transformers', 'rwkv7-gguf'.")
