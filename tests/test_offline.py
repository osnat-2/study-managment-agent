"""Offline checks for the raw Messages API harness."""

import json
import tempfile
import unittest
from pathlib import Path

from API.harness.caching import apply_cache_control
from API.harness.compaction import compact_messages
from API.harness.history import HistoryStore
from API.harness.loop import MessagesAgentLoop
from API.harness.memory import MemoryStore
from API.harness.tools import dispatch_tool
from API.harness.tokens import TokenUsageTracker
from API.main import run_scenario
from API.offline import OfflineMessagesClient
from shared import tools as shared_tools


class MessagesLoopTests(unittest.TestCase):
	def test_loop_runs_until_tool_use_stops(self):
		class FakeClient:
			def __init__(self):
				self.responses = [
					{
						"stop_reason": "tool_use",
						"content": [
							{
								"type": "tool_use",
								"id": "call-1",
								"name": "echo",
								"input": {"value": "hello"},
							}
						],
						"usage": {"input_tokens": 10, "output_tokens": 2},
					},
					{
						"stop_reason": "end_turn",
						"content": [{"type": "text", "text": "Done."}],
						"usage": {"input_tokens": 12, "output_tokens": 3},
					},
				]
				self.messages = self

			def create(self, **request):
				return self.responses.pop(0)

		client = FakeClient()
		result = MessagesAgentLoop(client, "test-model").run_turn(
			[{"role": "user", "content": "Use the echo tool."}],
			"system",
			tool_dispatcher=lambda blocks: (
				[{"type": "tool_result", "tool_use_id": "call-1", "content": "hello"}],
				[{"name": "echo", "is_error": False}],
			),
		)
		self.assertEqual(result["text"], "Done.")
		self.assertEqual(result["api_calls"], 2)
		self.assertEqual(result["tool_calls"][0]["name"], "echo")
		self.assertEqual(result["usage"]["input_tokens"], 22)


class PersistenceTests(unittest.TestCase):
	def test_history_is_append_only_and_reloadable(self):
		with tempfile.TemporaryDirectory() as directory:
			history = HistoryStore(Path(directory), "session/one")
			history.append({"role": "user", "content": "first"})
			history.append({"role": "assistant", "content": "second"})
			self.assertEqual(len(history.load()), 2)
			self.assertTrue(history.path.exists())

	def test_memory_survives_new_store_instance(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "memory.json"
			MemoryStore(path).set("study_time_preference", "evening only")
			self.assertEqual(
				MemoryStore(path).get("study_time_preference"), "evening only"
			)

	def test_compaction_and_cache_control(self):
		messages = [
			{"role": "user", "content": "old message " + str(index)}
			for index in range(12)
		]
		compacted = compact_messages(messages, token_threshold=1, keep_recent_messages=4)
		self.assertEqual(len(compacted), 5)
		self.assertIn("conversation_summary", compacted[0]["content"])
		system, tools = apply_cache_control("system", [{"name": "test"}])
		self.assertEqual(system[0]["cache_control"]["type"], "ephemeral")
		self.assertEqual(tools[0]["cache_control"]["type"], "ephemeral")
		usage = TokenUsageTracker("test-model")
		usage.record(
			{
				"input_tokens": 12,
				"output_tokens": 3,
				"cache_creation_input_tokens": 5,
				"cache_read_input_tokens": 8,
			}
		)
		self.assertEqual(usage.to_dict()["total_tokens"], 28)

	def test_tool_errors_are_returned(self):
		result, is_error = dispatch_tool("missing_tool", {})
		self.assertTrue(is_error)
		self.assertIn("Unknown tool", result)


class ScenarioTests(unittest.TestCase):
	def test_two_session_scenario_runs_without_live_api(self):
		project_root = Path(__file__).resolve().parents[1]
		scenario_path = project_root / "scenario.json"
		original_data_path = project_root / "shared" / "data.json"
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			data_path = root / "data.json"
			data_path.write_text(original_data_path.read_text(encoding="utf-8"), encoding="utf-8")
			original_shared_path = shared_tools.DATA_PATH
			try:
				result = run_scenario(
					scenario_path=scenario_path,
					results_path=root / "api.json",
					state_directory=root / "state",
					client=OfflineMessagesClient(),
					model="offline-test",
					data_file=data_path,
				)
			finally:
				shared_tools.DATA_PATH = original_shared_path
			self.assertEqual(result["aggregate_metrics"]["turns"], 6)
			self.assertEqual(result["aggregate_metrics"]["tool_calls"], 7)
			self.assertGreater(result["aggregate_metrics"]["input_tokens"], 0)
			self.assertTrue(result["turns"][0]["usage"]["estimated"])
			self.assertEqual(
				MemoryStore(root / "state" / "memory.json").get("study_time_preference"),
				"evening only",
			)
			self.assertEqual(json.loads(data_path.read_text(encoding="utf-8"))[0]["status"], "completed")
			self.assertEqual(len(json.loads((root / "api.json").read_text(encoding="utf-8"))["turns"]), 6)


if __name__ == "__main__":
	unittest.main()
