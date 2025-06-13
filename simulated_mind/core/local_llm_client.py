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
from typing import Optional, List, Dict, Any


class RWKV7GGUFClient(LocalLLMClient):
    """RWKV-7 GGUF implementation for efficient local AI interaction with state support."""
    
    def __init__(self, model_path: str, context_size: int = 8192):
        self.model_path = model_path
        self.context_size = context_size
        self.model: Optional[Llama] = None
        self.conversation_history: List[Dict[str, str]] = []
        
        # State management for RWKV compatibility
        self._current_state: Optional[Dict[str, Any]] = None
        self._state_history: List[Dict[str, Any]] = []
    
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
            
            # Initialize empty state
            self._current_state = self._create_empty_state()
            
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
    
    def _create_empty_state(self) -> Dict[str, Any]:
        """Create an empty state dictionary for RWKV compatibility."""
        return {
            "version": "1.0",
            "context_tokens": [],
            "conversation_history": [],
            "step_count": 0,
            "last_response": "",
            "metadata": {}
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current model state for GoT engine compatibility."""
        if self._current_state is None:
            self._current_state = self._create_empty_state()
        
        # Synchronize conversation history
        self._current_state["conversation_history"] = self.conversation_history.copy()

        # Ensure step_count is at least the length of conversation history but
        # do *not* overwrite a custom value that may be higher (tests rely on
        # persisted custom step counts).
        if self._current_state.get("step_count", 0) < len(self.conversation_history):
            self._current_state["step_count"] = len(self.conversation_history)

        # Return a deep copy so that unit tests mutating the returned dict do
        # not inadvertently modify the internal state via shared references.
        return copy.deepcopy(self._current_state)
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the model state for GoT engine compatibility."""
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")
        
        self._current_state = state.copy()
        
        # Restore conversation history from state if available
        if "conversation_history" in state:
            self.conversation_history = state["conversation_history"].copy()
        
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
        if self._current_state is None:
            self._current_state = self._create_empty_state()
        
        self._current_state["step_count"] += 1
        self._current_state["last_response"] = response
        self._current_state["context_tokens"].extend([prompt, response])
        
        # Keep context manageable
        if len(self._current_state["context_tokens"]) > 20:
            self._current_state["context_tokens"] = self._current_state["context_tokens"][-20:]
    
    def _build_minimal_prompt(self, base_prompt: str) -> str:
        """Minimal prompt providing context without role constraints."""
        context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-2:] 
            for exchange in recent_history:
                context += f"Human: {exchange['user'].strip()}\nAI: {exchange['assistant'].strip()}\n\n"
        
        return f"{context}Human: {base_prompt.strip()}\nAI:"
    
    def _update_conversation_history(self, user_input: str, assistant_response: str):
        self.conversation_history.append({
            "user": user_input.strip(),
            "assistant": assistant_response.strip()
        })
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def reset_conversation(self):
        self.conversation_history = []
        self._current_state = self._create_empty_state()
        self._current_state["context_tokens"] = []
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
