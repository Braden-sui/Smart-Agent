"""CodeModifier applies patches to source files with safety validation.

MVP strategy:
1. Receive CodePatch (target path + new_content string).
2. Ask SafetyGuard to validate (placeholder returns True).
3. If approved, make a timestamped backup then write new content.
4. On failure, restore backup.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..safety.guard import SafetyGuard
from ..journal.journal import Journal


@dataclass
class CodePatch:
    target: Path
    new_content: str

    def __post_init__(self):
        if not self.target.is_file():
            raise FileNotFoundError(self.target)


class _PatchListener(Protocol):
    """Optional protocol for callbacks (for future event hooks)."""

    def on_apply(self, patch: CodePatch) -> None: ...


class CodeModifier:
    def __init__(
        self,
        guard: SafetyGuard,
        listener: _PatchListener | None = None,
        journal: Journal | None = None,
    ):
        self.guard = guard
        self.listener = listener
        self.journal = journal or Journal.null()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_patch(self, patch: CodePatch) -> bool:
        """Return True if patch applied, False otherwise."""
        # 1. Validate via SafetyGuard
        self.journal.log_event("modifier.apply_patch.start", {"target": str(patch.target)})
        if not self.guard.validate_patch(patch):
            self.journal.log_event("modifier.apply_patch.rejected", {"target": str(patch.target)})
            return False

        # 2. Backup existing file
        backup_path = patch.target.with_suffix(
            patch.target.suffix + f".bak.{int(time.time())}"
        )
        shutil.copy2(patch.target, backup_path)
        try:
            # 3. Write new content
            patch.target.write_text(patch.new_content, encoding="utf-8")
            # 4. Notify listener
            if self.listener:
                self.listener.on_apply(patch)
            self.journal.log_event("modifier.apply_patch.success", {"target": str(patch.target)})
            return True
        except Exception:
            # Restore backup if write fails
            shutil.copy2(backup_path, patch.target)
            self.journal.log_event("modifier.apply_patch.error", {"target": str(patch.target)})
            return False
