"""Mock-client checks for persisted memory and append-only conversations."""

import json
import tempfile
import unittest
from pathlib import Path

from API.agent import StudyPlanningAgent
from API.harness.history import HistoryStore
from API.harness.memory import MemoryStore


class MockMessagesClient:
	"""Return scripted Messages API tool calls without network access."""

	def __init__(self, responses):
		self.responses = list(responses)
		self.requests = []
		self.messages = self

	def create(self, **request):
		self.requests.append(request)
		return self.responses.pop(0)


def tool_response(name, arguments, call_id):
	return {
		"stop_reason": "tool_use",
		"content": [
			{
				"type": "tool_use",
				"id": call_id,
				"name": name,
				"input": arguments,
			}
		],
		"usage": {"input_tokens": 20, "output_tokens": 5},
	}


def final_response(text):
	return {
		"stop_reason": "end_turn",
		"content": [{"type": "text", "text": text}],
		"usage": {"input_tokens": 22, "output_tokens": 6},
	}


class SavedStateAndHistoryTests(unittest.TestCase):
	def test_memory_store_preserves_other_saved_preferences(self):
		with tempfile.TemporaryDirectory() as directory:
			memory = MemoryStore(Path(directory) / "memory.json")
			memory.set("study_time_preference", "evening only")
			memory.set("study_location", "library")
			self.assertEqual(
				memory.all(),
				{
					"study_time_preference": "evening only",
					"study_location": "library",
				},
			)

	def test_mocked_memory_round_trip_across_agent_instances(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			client = MockMessagesClient(
				[
					tool_response(
						"save_memory",
						{"key": "study_time_preference", "value": "evening only"},
						"save-1",
					),
					final_response("Preference saved."),
					tool_response(
						"get_memory",
						{"key": "study_time_preference"},
						"read-1",
					),
					final_response("Your saved preference is evening only."),
				]
			)

			first_agent = StudyPlanningAgent(
				client, "mock-model", root, "session-one", prompt_caching=False
			)
			first_result = first_agent.run_turn(
				"I prefer studying in the evening only. Please remember this."
			)

			memory_path = root / "memory.json"
			self.assertEqual(
				json.loads(memory_path.read_text(encoding="utf-8"))[
					"study_time_preference"
				],
				"evening only",
			)
			self.assertEqual(
				MemoryStore(memory_path).get("study_time_preference"), "evening only"
			)
			self.assertEqual(first_result["text"], "Preference saved.")
			self.assertEqual(first_result["tool_calls"][0]["name"], "save_memory")

			second_agent = StudyPlanningAgent(
				client, "mock-model", root, "session-two", prompt_caching=False
			)
			second_result = second_agent.run_turn("Retrieve my saved study preference.")
			self.assertIn("evening only", second_result["text"])
			self.assertEqual(second_result["tool_calls"][0]["name"], "get_memory")

			first_history = HistoryStore(root / "history", "session-one").load()
			second_history = HistoryStore(root / "history", "session-two").load()
			self.assertEqual(
				[message["role"] for message in first_history],
				["user", "assistant", "user", "assistant"],
			)
			self.assertEqual(
				[message["role"] for message in second_history],
				["user", "assistant", "user", "assistant"],
			)
			self.assertEqual(len(client.requests), 4)
			self.assertTrue(
				all("cache_control" not in request for request in client.requests)
			)

	def test_history_records_are_append_only(self):
		with tempfile.TemporaryDirectory() as directory:
			history = HistoryStore(Path(directory), "stable-session")
			history.append({"role": "user", "content": "first"})
			first_record = history.path.read_text(encoding="utf-8")
			history.append({"role": "assistant", "content": "reply"})
			updated_records = history.path.read_text(encoding="utf-8")
			self.assertTrue(updated_records.startswith(first_record))
			self.assertEqual(len(updated_records.splitlines()), 2)
			self.assertEqual([entry["content"] for entry in history.load()], ["first", "reply"])

	def test_history_reloads_messages_in_order(self):
		with tempfile.TemporaryDirectory() as directory:
			history = HistoryStore(Path(directory), "ordered-session")
			expected = [
				{"role": "user", "content": "first prompt"},
				{"role": "assistant", "content": "first reply"},
				{"role": "user", "content": "second prompt"},
			]
			for message in expected:
				history.append(message)
			self.assertEqual(history.load(), expected)

	def test_history_is_isolated_between_sessions(self):
		with tempfile.TemporaryDirectory() as directory:
			first_session = HistoryStore(Path(directory), "session-one")
			second_session = HistoryStore(Path(directory), "session-two")
			first_session.append({"role": "user", "content": "first session"})
			second_session.append({"role": "user", "content": "second session"})
			self.assertEqual(first_session.load()[0]["content"], "first session")
			self.assertEqual(second_session.load()[0]["content"], "second session")
			self.assertNotEqual(first_session.path, second_session.path)


if __name__ == "__main__":
	unittest.main()
