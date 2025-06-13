from typing import List, Optional, Any, Dict, Union
import yaml
from simulated_mind.safety.guard import SafetyGuard
from simulated_mind.core.local_llm_client import RWKV7GGUFClient
from simulated_mind.journal.journal import Journal


class ConstitutionalViolationError(Exception):
    """Raised when a text violates a constitutional principle."""

    def __init__(self, message, principle: str):
        super().__init__(message)
        self.principle = principle


DEFAULT_CONSTITUTION: List[str] = [
    "Do not generate harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.",
    "Do not impersonate real people without explicit consent.",
    "Provide accurate information and avoid spreading misinformation.",
    "Do not engage in malicious activities like hacking or generating malware."
]


class RWKV7ConstitutionalSafetyGuard(SafetyGuard):
    """A safety guard that uses an RWKV7 model to ensure agent outputs adhere to a constitution."""

    def __init__(self, rwkv7_client: RWKV7GGUFClient, memory_dao, journal: Optional[Journal] = None):
        super().__init__(memory_dao, journal)
        self.rwkv7_client = rwkv7_client
        self.constitution: List[str] = []
        self.prompt_template: str = ""
        self._loaded = False

    def load(self, constitution: Optional[Union[Dict[str, Any], List[str]]] = None, constitution_path: Optional[str] = None):
        """Loads the constitution from a path, a dictionary, a list, or uses the default."""
        if constitution:
            if isinstance(constitution, dict):
                self._load_from_dict(constitution)
            elif isinstance(constitution, list):
                self.constitution = constitution
                self.prompt_template = (
                    "You are a constitutional safety guard. Your task is to determine if the following text violates any of the principles listed below. "
                    "If it does, respond with 'VIOLATION' followed by the number of the principle it violates (e.g., 'VIOLATION: 1'). "
                    "If it does not violate any principles, respond with 'OK'.\n\n"
                    "## Constitution:\n{principles}\n\n"
                    "## Text to Evaluate:\n\"{text_to_check}\"\n\n"
                    "## Judgement:\n"
                )
            else:
                raise TypeError("Constitution must be a dict, list, or None.")
        elif constitution_path:
            with open(constitution_path, 'r') as f:
                constitution_data = yaml.safe_load(f)
            self._load_from_dict(constitution_data)
        else:
            self.constitution = DEFAULT_CONSTITUTION
            self.prompt_template = (
                "You are a constitutional safety guard. Your task is to determine if the following text violates any of the principles listed below. "
                "If it does, respond with 'VIOLATION' followed by the number of the principle it violates (e.g., 'VIOLATION: 1'). "
                "If it does not violate any principles, respond with 'OK'.\n\n"
                "## Constitution:\n{principles}\n\n"
                "## Text to Evaluate:\n\"{text_to_check}\"\n\n"
                "## Judgement:\n"
            )

        if not self.constitution:
            raise ValueError("Constitution could not be loaded.")

        self._loaded = True

    def _load_from_dict(self, data: Dict[str, Any]):
        """Helper to load constitution from a dictionary."""
        self.prompt_template = data.get('prompt_template')
        if not self.prompt_template:
            raise ValueError("Constitution dictionary must contain a 'prompt_template'.")

        self.constitution = [
            value for key, value in data.items() if key.startswith('principle_')
        ]
        if not self.constitution:
            raise ValueError("Constitution dictionary must contain at least one 'principle_'.")

    def check_text(self, text: str) -> None:
        """Checks a given text against the constitution using the RWKV7 model.

        Raises:
            ConstitutionalViolationError: If the text violates a principle.
        """
        if not self._loaded:
            raise RuntimeError("Constitution not loaded. Call load() before check_text().")

        constitution_str = "\n".join(f"- {p}" for p in self.constitution)
        prompt = self.prompt_template.format(principles=constitution_str, text_to_check=text)

        response = self.rwkv7_client.complete_text(prompt, max_new_tokens=10).strip()

        if response.startswith("VIOLATION"):
            try:
                _, principle_num_str = response.split(':')
                principle_idx = int(principle_num_str.strip()) - 1
                violated_principle = self.constitution[principle_idx]
                raise ConstitutionalViolationError(
                    f"Text violates principle: '{violated_principle}'",
                    principle=violated_principle
                )
            except (ValueError, IndexError):
                raise ConstitutionalViolationError(
                    f"Text violates a constitutional principle, but the specific principle could not be determined from the response: {response}",
                    principle="Unknown"
                )

    def validate_patch(self, patch: Any) -> bool:
        """Validate a patch first using the parent's syntax/hash check, then a constitutional check."""
        if not super().validate_patch(patch):
            return False

        content: str = getattr(patch, "new_content", "")
        try:
            self.check_text(content)
        except ConstitutionalViolationError as exc:
            self.journal.log_event(
                "safety.validate_patch.constitutional_violation",
                {"target": str(getattr(patch, "target", "unknown")), "principle": exc.principle, "error": str(exc)},
            )
            return False

        return True
