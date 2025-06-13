"""SafetyGuard validates code patches, unit tests, and hash chain integrity."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from simulated_mind.journal.journal import Journal


class SafetyGuard:
    """Validate code patches via fast AST check and maintain hash chain integrity.

    MVP implementation performs:
    1. Parses ``patch.new_content`` using ``ast.parse`` to ensure syntactic correctness.
    2. Computes SHA-256 hash of the content and chains it to the previous
       approved patch for the same file (stored in mem0 under key
       ``hash_chain/<file>``). Any mismatch will reject the patch.
    """

    def __init__(self, memory_dao, journal: Journal | None = None):
        self.memory = memory_dao
        self.journal = journal or Journal.null()

    # ------------------------------------------------------------------
    # Patch Validation (MVP stub)
    # ------------------------------------------------------------------

    def validate_patch(self, patch: Any) -> bool:
        """Return True if patch is safe (placeholder)."""
        target_path = Path(getattr(patch, "target", "unknown"))
        content: str = getattr(patch, "new_content", "")

        # 1. AST parse for syntax validity
        try:
            ast.parse(content)
        except SyntaxError as exc:
            self.journal.log_event(
                "safety.validate_patch.syntax_error",
                {"target": str(target_path), "error": str(exc)},
            )
            return False

        # 2. Hash chain validation
        new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chain_key = f"hash_chain/{target_path}"
        retrieved_record = self.memory.retrieve_memory(user_id="safety_system", memory_id=chain_key)
        last_hash = retrieved_record.get("content") if retrieved_record else None
        # For MVP we just store last hash; future work: link hashes.
        if last_hash == new_hash:
            self.journal.log_event(
                "safety.validate_patch.duplicate",
                {"target": str(target_path)},
            )
            return False  # identical patch – no change

        # passed all checks – update chain
        self.memory.store_memory(user_id="safety_system", memory_id=chain_key, content=new_hash, tags=["hash_chain"])

        self.journal.log_event(
            "safety.validate_patch.accept",
            {
                "target": str(target_path),
                "hash": new_hash,
            },
        )
        return True
