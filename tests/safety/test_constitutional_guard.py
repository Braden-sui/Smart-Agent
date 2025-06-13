import pytest
from unittest.mock import MagicMock, patch

from simulated_mind.core.local_llm_client import RWKV7GGUFClient
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.safety.constitutional_guard import RWKV7ConstitutionalSafetyGuard, ConstitutionalViolationError
from simulated_mind.safety.guard import SafetyGuard


@pytest.fixture
def mock_rwkv_client():
    """Fixture for a mock RWKV7GGUFClient."""
    client = MagicMock(spec=RWKV7GGUFClient)
    client.load.return_value = None
    return client
#
@pytest.fixture
def mock_memory_dao():
    """Fixture for a mock MemoryDAO."""
    dao = MagicMock(spec=MemoryDAO)
    dao.retrieve_memory.return_value = {}
    dao.store_memory.return_value = None
    return dao

#
@pytest.fixture
def constitutional_guard(mock_rwkv_client, mock_memory_dao):
    """Fixture for RWKV7ConstitutionalSafetyGuard with mocked dependencies."""
    guard = RWKV7ConstitutionalSafetyGuard(
        rwkv7_client=mock_rwkv_client, 
        memory_dao=mock_memory_dao
    )
    # Use a dummy constitution to avoid file I/O during testing
    dummy_constitution = {
        "principle_1": "Be helpful and harmless.",
        "prompt_template": "Analyze the following text based on the principle: {principles}. Text: '{text_to_check}'. Respond with OK or VIOLATION."
    }
    guard.load(constitution=dummy_constitution)
    return guard

def test_check_text_ok(constitutional_guard, mock_rwkv_client):
    """Test that check_text passes for compliant text."""
    mock_rwkv_client.complete_text.return_value = "OK"
    # Method should not raise any exception for compliant text
    constitutional_guard.check_text("This is a safe prompt.")
    mock_rwkv_client.complete_text.assert_called_once()


def test_check_text_violation(constitutional_guard, mock_rwkv_client):
    """Test that check_text raises ConstitutionalViolationError for non-compliant text."""
    mock_rwkv_client.complete_text.return_value = "VIOLATION: 1"
    with pytest.raises(ConstitutionalViolationError) as excinfo:
        constitutional_guard.check_text("This is a harmful prompt.")
    assert "violates principle" in str(excinfo.value)
    assert excinfo.value.principle == "Be helpful and harmless."


def test_validate_patch_ok(constitutional_guard, mock_rwkv_client):
    """Test that validate_patch passes for a compliant patch."""
    mock_rwkv_client.complete_text.return_value = "OK"
    patch_obj = MagicMock()
    patch_obj.target = "test.py"
    patch_obj.new_content = "print('hello world')"
    with patch.object(SafetyGuard, 'validate_patch', return_value=True):
        assert constitutional_guard.validate_patch(patch_obj) is True


def test_validate_patch_violation(constitutional_guard, mock_rwkv_client):
    """Test that validate_patch fails for a non-compliant patch."""
    mock_rwkv_client.complete_text.return_value = "VIOLATION: 4"
    patch_obj = MagicMock()
    patch_obj.target = "test.py"
    patch_obj.new_content = "import os; os.system('rm -rf /')"
    with patch.object(SafetyGuard, 'validate_patch', return_value=True):
        assert constitutional_guard.validate_patch(patch_obj) is False


def test_validate_patch_syntax_error(constitutional_guard):
    """Test that validate_patch fails for a patch with a syntax error."""
    patch_obj = MagicMock()
    patch_obj.target = "test.py"
    patch_obj.new_content = "print('hello world'"  # Missing closing parenthesis
    # The real super().validate_patch() will catch the syntax error
    assert constitutional_guard.validate_patch(patch_obj) is False


def test_validate_patch_parent_fails(constitutional_guard, mock_rwkv_client):
    """Test that if the parent validation fails, the patch is rejected."""
    patch_obj = MagicMock()
    patch_obj.target = "test.py"
    patch_obj.new_content = "print('hello world')"
    with patch.object(SafetyGuard, 'validate_patch', return_value=False):
        assert constitutional_guard.validate_patch(patch_obj) is False
    mock_rwkv_client.complete_text.assert_not_called()

