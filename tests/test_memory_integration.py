import unittest
from unittest.mock import patch, MagicMock
import logging

# Assuming MEM0_SDK_AVAILABLE might be dynamically checked or mocked
# For testing, we'll often want to control its perceived state.
from simulated_mind.memory.mem0_client import Mem0Client #, MEM0_SDK_AVAILABLE
from simulated_mind.memory.dao import MemoryDAO
from simulated_mind.logging.journal import Journal # Assuming a Journal class exists

# Disable most logging for tests unless specifically debugging
logging.basicConfig(level=logging.CRITICAL)

class TestMem0ClientIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_journal = MagicMock(spec=Journal)

    def test_initialization_fallback_no_apikey(self):
        """Test Mem0Client initializes in fallback mode without API key."""
        client = Mem0Client(journal=self.mock_journal)
        self.assertFalse(client.sdk_active)
        self.assertIsNone(client.client)
        self.mock_journal.log_event.assert_any_call("Mem0Client.Info", "Mem0Client initialized with in-memory store (no API key or SDK not found).")

    @patch('simulated_mind.memory.mem0_client.MEM0_SDK_AVAILABLE', False)
    def test_initialization_fallback_sdk_not_available(self):
        """Test Mem0Client initializes in fallback mode if API key is given but SDK is not available."""
        client = Mem0Client(api_key="fake_key", journal=self.mock_journal)
        self.assertFalse(client.sdk_active)
        self.assertIsNone(client.client)
        expected_label = "Mem0Client.Warning"
        actual_call_payload = None
        call_found_with_label = False

        for call_args_tuple in self.mock_journal.log_event.call_args_list:
            pos_args, kw_args = call_args_tuple
            if pos_args and pos_args[0] == expected_label:
                call_found_with_label = True
                actual_call_payload = kw_args.get('payload')
                if actual_call_payload is None and len(pos_args) > 1:
                    actual_call_payload = pos_args[1]
                break

        self.assertTrue(call_found_with_label, 
                        f"Log event with label '{expected_label}' not found. Actual calls: {self.mock_journal.log_event.call_args_list}")
        self.assertIsNotNone(actual_call_payload, f"Payload not found for log event '{expected_label}'. Actual calls: {self.mock_journal.log_event.call_args_list}")

        expected_message = "Mem0 API key provided, but 'mem0ai' SDK not installed. Falling back to in-memory store."
        self.assertIn(expected_message, actual_call_payload.get('message'), # Check if substring due to potential extra info
                         f"Payload message mismatch for '{expected_label}'. Expected: '{expected_message}', Got: '{actual_call_payload.get('message')}'. Full payload: {actual_call_payload}")
        self.assertIn("warning", actual_call_payload.get('level'), # Check if substring due to potential extra info
                         f"Payload level mismatch for '{expected_label}'. Expected: 'warning', Got: '{actual_call_payload.get('level')}'. Full payload: {actual_call_payload}")

    @patch('simulated_mind.memory.mem0_client.MEM0_SDK_AVAILABLE', True)
    @patch('simulated_mind.memory.mem0_client.MemoryClient')
    # The patch for MEM0_SDK_AVAILABLE provides a direct value (True), so it does not pass an argument.
    # Only the patch for Memory (which doesn't specify 'new') passes an argument.
    def test_initialization_sdk_success(self, MockMemorySDK):
        """Test Mem0Client initializes with SDK successfully."""
        mock_sdk_instance = MockMemorySDK.return_value
        client = Mem0Client(api_key="fake_key", journal=self.mock_journal)
        self.assertTrue(client.sdk_active)
        self.assertEqual(client.client, mock_sdk_instance)
        MockMemorySDK.assert_called_once_with(api_key="fake_key", org_id="default_org", project_id="default_project")
        self.mock_journal.log_event.assert_any_call("Mem0Client.SDK", "Mem0Client initialized with SDK")

    @patch('simulated_mind.memory.mem0_client.MEM0_SDK_AVAILABLE', True)
    @patch('simulated_mind.memory.mem0_client.MemoryClient')
    # The patch for MEM0_SDK_AVAILABLE provides a direct value (True), so it does not pass an argument.
    # Only the patch for Memory (which doesn't specify 'new') passes an argument.
    def test_initialization_sdk_failure_fallback(self, MockMemorySDK):
        """Test Mem0Client falls back to in-memory if SDK initialization fails."""
        MockMemorySDK.side_effect = Exception("SDK Init Error")
        client = Mem0Client(api_key="fake_key", journal=self.mock_journal)
        self.assertFalse(client.sdk_active)
        self.assertIsNone(client.client)
        expected_label = "Mem0Client.Error"
        actual_call_payload = None
        call_found_with_label = False

        for call_args_tuple in self.mock_journal.log_event.call_args_list:
            pos_args, kw_args = call_args_tuple
            if pos_args and pos_args[0] == expected_label:
                call_found_with_label = True
                actual_call_payload = kw_args.get('payload')
                if actual_call_payload is None and len(pos_args) > 1:
                    actual_call_payload = pos_args[1]
                break

        self.assertTrue(call_found_with_label, 
                        f"Log event with label '{expected_label}' not found. Actual calls: {self.mock_journal.log_event.call_args_list}")
        self.assertIsNotNone(actual_call_payload, f"Payload not found for log event '{expected_label}'. Actual calls: {self.mock_journal.log_event.call_args_list}")

        expected_message_content = "Failed to initialize Mem0 SDK: SDK Init Error" # The '. Falling back to in-memory store.' is part of the generic message in _log_error
        self.assertIn(expected_message_content, actual_call_payload.get('message'), # Check if substring 
                         f"Payload message mismatch for '{expected_label}'. Expected: '{expected_message_content}', Got: '{actual_call_payload.get('message')}'. Full payload: {actual_call_payload}")
        self.assertEqual(actual_call_payload.get('level'), "error", 
                         f"Payload level mismatch for '{expected_label}'. Expected: 'error', Got: '{actual_call_payload.get('level')}'. Full payload: {actual_call_payload}")
        self.assertEqual(actual_call_payload.get('exception_type'), "Exception", 
                         f"Payload exception_type mismatch for '{expected_label}'. Expected: 'Exception', Got: '{actual_call_payload.get('exception_type')}'. Full payload: {actual_call_payload}")

    def test_crud_fallback(self):
        """Test basic CRUD operations using the in-memory fallback."""
        client = Mem0Client(journal=self.mock_journal)
        user_id = "user123"
        memory_id = "mem_abc"
        content = {"data": "sample content"}
        tags = ["tag1", "tag2"]
        metadata = {"source": "test"}

        # Create
        client.create_memory(memory_id=memory_id, content=content, tags=tags, user_id=user_id, metadata=metadata)
        self.mock_journal.log_event.assert_any_call("Mem0Client.Warning", {'message': f"Using fallback storage for memory '{memory_id}'", 'level': 'warning'})
        
        # Retrieve
        retrieved = client.get_memory(memory_id, user_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['content'], content)
        self.assertEqual(retrieved['tags'], tags)
        self.assertEqual(retrieved['source'], "test")
        self.mock_journal.log_event.assert_any_call("Mem0Client.Warning", {'message': f"Using fallback storage to retrieve memory for user '{user_id}', id '{memory_id}'", 'level': 'warning'})

        # Update
        updated_content = {"data": "updated content"}
        updated_tags = ["tag3"]
        client.update_memory(memory_id, content=updated_content, tags=updated_tags, user_id=user_id)
        self.mock_journal.log_event.assert_any_call("Mem0Client.Warning", {'message': f"Updating memory for user '{user_id}', id '{memory_id}'.", 'level': 'warning'})
        retrieved_after_update = client.get_memory(memory_id, user_id)
        self.assertEqual(retrieved_after_update['content'], updated_content)
        self.assertEqual(retrieved_after_update['tags'], updated_tags)

        # Delete
        delete_status = client.delete_memory(memory_id, user_id)
        self.assertTrue(delete_status)
        self.mock_journal.log_event.assert_any_call("Mem0Client.Warning", {'message': f"Deleting memory for user '{user_id}', id '{memory_id}'.", 'level': 'warning'})
        self.assertIsNone(client.get_memory(memory_id, user_id))
        self.assertFalse(client.delete_memory("non_existent_mem", user_id)) # Test deleting non-existent

    def test_search_by_tags_fallback(self):
        """Test searching memories by tags using the in-memory fallback."""
        client = Mem0Client(journal=self.mock_journal)
        client.create_memory(memory_id="mem1", content="content1", tags=["apple", "fruit"], user_id="user1")
        client.create_memory(memory_id="mem2", content="content2", tags=["banana", "fruit"], user_id="user1")
        client.create_memory(memory_id="mem3", content="content3", tags=["apple", "food"], user_id="user1")
        client.create_memory(memory_id="mem4", content="content4", tags=["apple", "fruit"], user_id="user2") # Different user

        results_apple_fruit = client.search_memories_by_tags(["apple", "fruit"], "user1")
        self.mock_journal.log_event.assert_any_call(
            "Mem0Client.Warning", 
            {'message': "Using fallback tag search for tags: ['apple', 'fruit']", 'level': 'warning'}
        )
        self.assertEqual(len(results_apple_fruit), 1)
        self.assertEqual(results_apple_fruit[0]['memory_id'], "mem1")

        results_fruit = client.search_memories_by_tags(["fruit"], "user1")
        self.assertEqual(len(results_fruit), 2)
        mem_ids_fruit = {r['memory_id'] for r in results_fruit}
        self.assertSetEqual(mem_ids_fruit, {"mem1", "mem2"})

        results_food_user1 = client.search_memories_by_tags(["food"], "user1")
        self.assertEqual(len(results_food_user1), 1)
        self.assertEqual(results_food_user1[0]['memory_id'], "mem3")
        
        results_apple_user2 = client.search_memories_by_tags(["apple"], "user2")
        self.assertEqual(len(results_apple_user2), 1)
        self.assertEqual(results_apple_user2[0]['memory_id'], "mem4")

        results_empty = client.search_memories_by_tags(["non_existent_tag"], "user1")
        self.assertEqual(len(results_empty), 0)

    def test_kg_operations_fallback(self):
        """Test knowledge graph operations using the in-memory fallback."""
        client = Mem0Client(journal=self.mock_journal)
        user_id = "kg_user"
        client.add_relation("subject1", "predicateA", "objectX", user_id, {"source": "test_kg"})
        client.add_relation("subject1", "predicateB", "objectY", user_id)
        client.add_relation("subject2", "predicateA", "objectZ", user_id)
        client.add_relation("subject3", "predicateA", "objectX", "other_user")

        # Query by subject
        relations_s1 = client.get_relations_by_pattern(subject="subject1", user_id=user_id)
        self.assertEqual(len(relations_s1), 2)

        # Query by predicate
        relations_pA = client.get_relations_by_pattern(predicate="predicateA", user_id=user_id)
        self.assertEqual(len(relations_pA), 2)
        self.assertTrue(any(r['object'] == 'objectX' and r['meta'].get('source') == 'test_kg' for r in relations_pA))

        # Query by object
        relations_oX = client.get_relations_by_pattern(obj="objectX", user_id=user_id)
        self.assertEqual(len(relations_oX), 1)

        # Query by subject and predicate
        relations_s1_pB = client.get_relations_by_pattern(subject="subject1", predicate="predicateB", user_id=user_id)
        self.assertEqual(len(relations_s1_pB), 1)
        self.assertEqual(relations_s1_pB[0]['object'], "objectY")

        # Query with no matches
        relations_none = client.get_relations_by_pattern(subject="non_existent", user_id=user_id)
        self.assertEqual(len(relations_none), 0)
        
        # Query for other user
        relations_other_user = client.get_relations_by_pattern(obj="objectX", user_id="other_user")
        self.assertEqual(len(relations_other_user), 1)
        self.assertEqual(relations_other_user[0]['subject'], "subject3")


class TestMemoryDAOIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_journal = MagicMock(spec=Journal)
        # Mock for when DAO creates its own Mem0Client instance
        self.mock_mem0_client_internal = MagicMock(spec=Mem0Client)
        # Mock for when DAO is provided with a Mem0Client instance
        self.mock_mem0_client_provided = MagicMock(spec=Mem0Client)

    @patch('simulated_mind.memory.dao.Mem0Client')
    def test_dao_initialization_with_api_key(self, MockInternalMem0ClientConstructor):
        """Test MemoryDAO initializes its own Mem0Client when api_key is provided."""
        MockInternalMem0ClientConstructor.return_value = self.mock_mem0_client_internal
        api_key = "test_api_key"
        dao = MemoryDAO(api_key=api_key, journal=self.mock_journal)
        
        MockInternalMem0ClientConstructor.assert_called_once_with(api_key=api_key, journal=self.mock_journal)
        self.assertEqual(dao.client, self.mock_mem0_client_internal)
        self.mock_journal.log_event.assert_any_call(
            "MemoryDAO: Initialized new Mem0Client instance (API key provided: True)."
        )

    @patch('simulated_mind.memory.dao.Mem0Client')
    def test_dao_initialization_no_args_fallback(self, MockInternalMem0ClientConstructor):
        """Test MemoryDAO initializes its own Mem0Client in fallback if no args are provided to DAO."""
        MockInternalMem0ClientConstructor.return_value = self.mock_mem0_client_internal
        dao = MemoryDAO(journal=self.mock_journal) # No api_key, no client instance
        
        MockInternalMem0ClientConstructor.assert_called_once_with(api_key=None, journal=self.mock_journal)
        self.assertEqual(dao.client, self.mock_mem0_client_internal)
        self.mock_journal.log_event.assert_any_call(
            "MemoryDAO: Initialized new Mem0Client instance (API key provided: False)."
        )

    def test_dao_initialization_with_provided_client(self):
        """Test MemoryDAO uses a provided Mem0Client instance and does not create a new one."""
        dao = MemoryDAO(mem0_client_instance=self.mock_mem0_client_provided, journal=self.mock_journal)
        self.assertEqual(dao.client, self.mock_mem0_client_provided)
        self.mock_journal.log_event.assert_any_call(
            "MemoryDAO: Initialized with provided Mem0Client instance."
        )
        # Important: We do NOT patch Mem0Client here, so if DAO tried to create one, it would be a real one.
        # The absence of a patch and related call assertions implicitly tests that DAO doesn't create a client.

    def _test_dao_method_success(self, dao_method_name, client_method_name, method_args, client_expected_args, expected_return_value=None, is_bool_method=False):
        dao = MemoryDAO(mem0_client_instance=self.mock_mem0_client_provided, journal=self.mock_journal)
        # Reset mock for each specific method test run on the provided client
        self.mock_mem0_client_provided.reset_mock()
        mock_client_method = getattr(self.mock_mem0_client_provided, client_method_name)
        
        # Set return value for the client's method
        if is_bool_method: # For methods returning True/False
            mock_client_method.return_value = expected_return_value
        elif expected_return_value is not None:
            mock_client_method.return_value = expected_return_value
        
        dao_method_to_call = getattr(dao, dao_method_name)
        actual_return_value = dao_method_to_call(*method_args)

        mock_client_method.assert_called_once_with(*client_expected_args)
        
        log_message_map_success = {
            "store_memory": "Stored memory",
            "retrieve_memory": "Retrieved memory",
            "update_memory": "Updated memory",
            "delete_memory": "Deleted memory",
            "find_memories_by_tags": "Searched memories",
            "add_kg_relation": "Added KG relation",
            "query_kg_relations": "Queried KG relations"
        }
        expected_log_action = log_message_map_success[dao_method_name]
        self.mock_journal.log_event.assert_any_call(
            f"MemoryDAO.{dao_method_name}: {expected_log_action}.",
            payload=unittest.mock.ANY
        )

        if is_bool_method or expected_return_value is not None:
            self.assertEqual(actual_return_value, expected_return_value)

    def _test_dao_method_failure(self, dao_method_name, client_method_name, method_args, client_expected_args, expected_return_on_failure):
        dao = MemoryDAO(mem0_client_instance=self.mock_mem0_client_provided, journal=self.mock_journal)
        self.mock_mem0_client_provided.reset_mock() # Reset mock for each test
        mock_client_method = getattr(self.mock_mem0_client_provided, client_method_name)
        test_exception = Exception("Client Error")
        mock_client_method.side_effect = test_exception
        
        dao_method_to_call = getattr(dao, dao_method_name)
        actual_return_value = dao_method_to_call(*method_args)

        mock_client_method.assert_called_once_with(*client_expected_args)

        log_message_map_failure = {
            "store_memory": "Error storing memory",
            "retrieve_memory": "Error retrieving memory",
            "update_memory": "Error updating memory",
            "delete_memory": "Error deleting memory",
            "find_memories_by_tags": "Error searching memories",
            "add_kg_relation": "Error adding KG relation",
            "query_kg_relations": "Error querying KG relations"
        }
        expected_log_action_failure = log_message_map_failure[dao_method_name]
        # Check for the specific error log call
        found_error_log = False
        for call in self.mock_journal.log_event.call_args_list:
            args, kwargs = call
            if args[0] == f"MemoryDAO.{dao_method_name}: {expected_log_action_failure}." and \
               kwargs.get('payload', {}).get('error') == str(test_exception):
                found_error_log = True
                break
        self.assertTrue(found_error_log, f"Expected error log for {dao_method_name} not found or incorrect.")
        self.assertEqual(actual_return_value, expected_return_on_failure)

    # --- Test store_memory ---
    def test_store_memory_success(self):
        self._test_dao_method_success(
            "store_memory", "create_memory",
            method_args=("user1", "mem1", {"data": "content"}, ["tag1"], {"meta": "data"}),
            client_expected_args=("mem1", {"data": "content"}, ["tag1"], "user1", {"meta": "data"})
        )

    def test_store_memory_failure(self):
        self._test_dao_method_failure(
            "store_memory", "create_memory",
            method_args=("user1", "mem1", {"data": "content"}, ["tag1"], {"meta": "data"}),
            client_expected_args=("mem1", {"data": "content"}, ["tag1"], "user1", {"meta": "data"}),
            expected_return_on_failure=None # store_memory returns None on failure path in DAO
        )

    # --- Test retrieve_memory ---
    def test_retrieve_memory_success(self):
        ret_val = {"id": "mem1", "content": "data"}
        self._test_dao_method_success(
            "retrieve_memory", "get_memory",
            method_args=("user1", "mem1"),
            client_expected_args=("mem1", "user1"),
            expected_return_value=ret_val
        )

    def test_retrieve_memory_failure(self):
        self._test_dao_method_failure(
            "retrieve_memory", "get_memory",
            method_args=("user1", "mem1"),
            client_expected_args=("mem1", "user1"),
            expected_return_on_failure=None
        )

    # --- Test update_memory ---
    def test_update_memory_success(self):
        self._test_dao_method_success(
            "update_memory", "update_memory",
            method_args=("user1", "mem1", {"data": "new_content"}, ["tag_new"], {"meta_new": "val_new"}),
            client_expected_args=("mem1", "user1", {"data": "new_content"}, ["tag_new"], {"meta_new": "val_new"}),
            expected_return_value=True,
            is_bool_method=True
        )

    def test_update_memory_failure(self):
        self._test_dao_method_failure(
            "update_memory", "update_memory",
            method_args=("user1", "mem1", {"data": "new_content"}),
            client_expected_args=("mem1", "user1", {"data": "new_content"}, None, None), # DAO passes None for unspecifieds
            expected_return_on_failure=False
        )
        
    # --- Test delete_memory ---
    def test_delete_memory_success(self):
        self._test_dao_method_success(
            "delete_memory", "delete_memory",
            method_args=("user1", "mem1"),
            client_expected_args=("mem1", "user1"),
            expected_return_value=True,
            is_bool_method=True
        )

    def test_delete_memory_failure(self):
        self._test_dao_method_failure(
            "delete_memory", "delete_memory",
            method_args=("user1", "mem1"),
            client_expected_args=("mem1", "user1"),
            expected_return_on_failure=False
        )

    # --- Test find_memories_by_tags ---
    def test_find_memories_by_tags_success(self):
        ret_val = [{"id": "mem1", "content": "data"}]
        self._test_dao_method_success(
            "find_memories_by_tags", "search_memories_by_tags",
            method_args=("user1", ["tag1", "tag2"], 10),
            client_expected_args=(["tag1", "tag2"], "user1", 10),
            expected_return_value=ret_val
        )

    def test_find_memories_by_tags_failure(self):
        self._test_dao_method_failure(
            "find_memories_by_tags", "search_memories_by_tags",
            method_args=("user1", ["tag1", "tag2"], 10),
            client_expected_args=(["tag1", "tag2"], "user1", 10),
            expected_return_on_failure=[]
        )

    # --- Test add_kg_relation ---
    def test_add_kg_relation_success(self):
        self._test_dao_method_success(
            "add_kg_relation", "add_relation",
            method_args=("user1", "s1", "p1", "o1", {"meta": "kg"}),
            client_expected_args=("s1", "p1", "o1", "user1", {"meta": "kg"})
        )

    def test_add_kg_relation_failure(self):
        self._test_dao_method_failure(
            "add_kg_relation", "add_relation",
            method_args=("user1", "s1", "p1", "o1", {"meta": "kg"}),
            client_expected_args=("s1", "p1", "o1", "user1", {"meta": "kg"}),
            expected_return_on_failure=None # add_kg_relation returns None on failure path in DAO
        )

    # --- Test query_kg_relations ---
    def test_query_kg_relations_success(self):
        ret_val = [{"sub": "s1", "pred": "p1", "obj": "o1"}]
        self._test_dao_method_success(
            "query_kg_relations", "get_relations_by_pattern",
            method_args=("user1", "s1", "p1", None), # obj is optional
            client_expected_args=("s1", "p1", None, "user1"),
            expected_return_value=ret_val
        )

    def test_query_kg_relations_failure(self):
        self._test_dao_method_failure(
            "query_kg_relations", "get_relations_by_pattern",
            method_args=("user1", "s1", "p1", None),
            client_expected_args=("s1", "p1", None, "user1"),
            expected_return_on_failure=[]
        )


if __name__ == '__main__':
    unittest.main()
