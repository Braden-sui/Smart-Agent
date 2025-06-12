from abc import ABC, abstractmethod
from typing import Optional

class LocalLLMClient(ABC):
    """Abstract base class for local LLM inference."""
    @abstractmethod
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        pass
    @abstractmethod
    def is_available(self) -> bool:
        pass

class MockLocalLLMClient(LocalLLMClient):
    """Mock implementation for development and testing."""
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        if "organize" in prompt.lower() and "kitchen" in prompt.lower():
            return '["Assess current kitchen layout and identify issues", "Remove unnecessary items and declutter", "Group similar items by category", "Organize storage systems", "Create maintenance routine"]'
        elif "plan" in prompt.lower() and "week" in prompt.lower():
            return '["Review calendar and upcoming commitments", "Identify top 3 priorities", "Schedule important tasks", "Plan meals and shopping", "Block time for personal activities"]'
        elif "clean" in prompt.lower():
            return '["Create cleaning schedule", "Gather necessary supplies", "Declutter and organize first", "Clean systematically room by room", "Establish maintenance routine"]'
        else:
            goal_text = prompt.split('"')[1] if '"' in prompt else "the task"
            return f'["Break {goal_text} into smaller steps", "Gather necessary resources and information", "Create action plan and timeline", "Execute plan systematically", "Review progress and adjust as needed"]'
    def is_available(self) -> bool:
        return True

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

class RWKVLocalLLMClient(LocalLLMClient):
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        # TODO: Implement RWKV loading when ready
    def complete_text(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError("RWKV integration not yet implemented")
    def is_available(self) -> bool:
        return False  # TODO: Return True when RWKV is loaded

def create_local_llm_client(backend: str = "mock", **kwargs) -> LocalLLMClient:
    if backend == "mock":
        return MockLocalLLMClient()
    elif backend == "transformers":
        return TransformersLocalLLMClient(**kwargs)
    elif backend == "rwkv":
        return RWKVLocalLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
