import textwrap
from pathlib import Path

import pytest

from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.safety.guard import SafetyGuard
from simulated_mind.modification.modifier import CodeModifier, CodePatch
from simulated_mind.logging.journal import Journal


@pytest.fixture()

def sample_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.py"
    file_path.write_text("x = 1\n", encoding="utf-8")
    return file_path


def test_patch_acceptance(sample_file: Path):
    memory = MemoryDAO(journal=Journal.null())
    guard = SafetyGuard(memory, journal=Journal.null())
    modifier = CodeModifier(guard, journal=Journal.null())

    new_content = textwrap.dedent(
        """
        x = 2
        def foo():
            return x * 2
        """
    )
    patch = CodePatch(target=sample_file, new_content=new_content)
    assert modifier.apply_patch(patch) is True
    assert sample_file.read_text(encoding="utf-8") == new_content



def test_patch_rejection_on_syntax_error(sample_file: Path):
    original = sample_file.read_text(encoding="utf-8")
    memory = MemoryDAO(journal=Journal.null())
    guard = SafetyGuard(memory, journal=Journal.null())
    modifier = CodeModifier(guard, journal=Journal.null())

    bad_content = "def broken(:\n    pass\n"
    patch = CodePatch(target=sample_file, new_content=bad_content)
    # Guard should reject invalid syntax and modifier returns False
    assert modifier.apply_patch(patch) is False
    # File content unchanged
    assert sample_file.read_text(encoding="utf-8") == original
