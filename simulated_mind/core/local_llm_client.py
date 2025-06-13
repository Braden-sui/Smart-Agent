from abc import ABC, abstractmethod # Keep these imports for the remaining classes
from typing import Optional # Keep these imports for the remaining classes

# Base class definition moved to ensure correct inheritance order.
class LocalLLMClient:
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError
    def is_available(self) -> bool:
        raise NotImplementedError
    def reset_conversation(self):
        pass # Optional
    def save_conversation_state(self, filepath: str):
        pass # Optional
    def load_conversation_state(self, filepath: str):
        pass # Optional

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
from typing import Optional, List, Dict # These imports are now potentially duplicated, but harmless
from llama_cpp import Llama

# MockLocalLLMClient removed as per user request.

class RWKV7GGUFClient(LocalLLMClient):
    """RWKV-7 GGUF implementation for efficient local AI interaction."""
    
    def __init__(self, model_path: str, context_size: int = 8192):
        self.model_path = model_path
        self.context_size = context_size
        self.model: Optional[Llama] = None
        self.conversation_history: List[Dict[str, str]] = []
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_threads=os.cpu_count() or 4, # Use available cores
                n_gpu_layers=0,  # Set to >0 if CUDA/Metal is available and desired
                verbose=False
            )
            print(f"✅ RWKV-7 GGUF loaded: {os.path.basename(self.model_path)}")
        except Exception as e:
            # Be more specific about the error if possible, e.g., if model file not found
            if isinstance(e, FileNotFoundError) or "No such file or directory" in str(e):
                print(f"❌ Failed to load RWKV-7 GGUF: Model file not found at {self.model_path}. Please ensure the path is correct and the model is downloaded.")
            else:
                print(f"❌ Failed to load RWKV-7 GGUF: {e}")
            self.model = None
    
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.is_available() or self.model is None:
            raise RuntimeError("RWKV-7 GGUF model not available or not loaded.")
        
        try:
            full_prompt = self._build_minimal_prompt(prompt)
            
            response = self.model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.85,
                stop=["Human:", "AI:", "\nHuman:", "\nAI:", "\n\n"], # More robust stop tokens
                echo=False
            )
            
            response_text = response['choices'][0]['text'].strip()
            # Ensure we store the original user prompt, not the augmented one
            self._update_conversation_history(prompt, response_text) 
            return response_text
            
        except Exception as e:
            print(f"❌ RWKV generation failed: {e}")
            raise
    
    def _build_minimal_prompt(self, base_prompt: str) -> str:
        """Minimal prompt providing context without role constraints."""
        context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-2:] 
            for exchange in recent_history:
                # Ensure clean separation and formatting
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
        print("🔄 Conversation history reset")
    
    def save_conversation_state(self, filepath: str):
        try:
            # Ensure directory exists before trying to save
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            # print(f"💾 Conversation saved to {filepath}") # Optional: can be noisy
        except Exception as e:
            print(f"⚠️ Could not save conversation to {filepath}: {e}")
    
    def load_conversation_state(self, filepath: str):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.conversation_history = json.load(f)
                print(f"📖 Conversation loaded from {filepath}")
        except Exception as e:
            print(f"⚠️ Could not load conversation: {e}")

def create_local_llm_client(backend: str = "rwkv7-gguf", **kwargs) -> LocalLLMClient:
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
