"""Offline checks for Claude Agent SDK configuration and scenario execution."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from SDK.agent import ALLOWED_TOOLS, SYSTEM_PROMPT, build_agent_options
from SDK.main import run_offline_scenario


class SDKConfigurationTests(unittest.TestCase):
	def test_sdk_options_are_strict_and_restricted(self):
		created_servers = []

		class FakeOptions:
			def __init__(self, **kwargs):
				self.values = kwargs

		def fake_tool(name, description, schema):
			return lambda handler: {"name": name, "schema": schema, "handler": handler}

		def fake_server(name, version, tools):
			server = {"name": name, "version": version, "tools": tools}
			created_servers.append(server)
			return server

		fake_sdk = types.ModuleType("claude_agent_sdk")
		fake_sdk.ClaudeAgentOptions = FakeOptions
		fake_sdk.tool = fake_tool
		fake_sdk.create_sdk_mcp_server = fake_server
		with patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk}):
			with tempfile.TemporaryDirectory() as directory:
				options = build_agent_options(
					model="test-model",
					cwd=Path(directory),
					data_path=Path(directory) / "data.json",
					memory_path=Path(directory) / "memory.json",
					resume_session_id="saved-session-id",
				)

		self.assertTrue(options.values["strict_mcp_config"])
		self.assertEqual(options.values["tools"], [])
		self.assertEqual(options.values["allowed_tools"], ALLOWED_TOOLS)
		self.assertEqual(options.values["resume"], "saved-session-id")
		self.assertEqual(options.values["system_prompt"], SYSTEM_PROMPT)
		self.assertEqual(len(created_servers[0]["tools"]), 5)

	def test_offline_scenario_persists_memory_and_resumes_session(self):
		project_root = Path(__file__).resolve().parents[1]
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			data_path = root / "data.json"
			data_path.write_text(
				(project_root / "shared" / "data.json").read_text(encoding="utf-8"),
				encoding="utf-8",
			)
			output = run_offline_scenario(
				scenario_path=project_root / "scenario.json",
				results_path=root / "sdk.json",
				state_directory=root / "state",
				data_path=data_path,
			)
			self.assertEqual(output["aggregate_metrics"]["turns"], 6)
			self.assertEqual(output["aggregate_metrics"]["tool_calls"], 10)
			self.assertEqual(
				output["session_ids"]["session_1"], output["session_ids"]["session_2"]
			)
			self.assertEqual(
				json.loads((root / "state" / "memory.json").read_text(encoding="utf-8"))[
					"study_time_preference"
				],
				"evening only",
			)
			self.assertEqual(
				json.loads(data_path.read_text(encoding="utf-8"))[0]["status"], "completed"
			)
			self.assertTrue(output["turns"][3]["response"].startswith("Saved preference:"))
			self.assertIn("evening-only", output["turns"][4]["response"])
			self.assertIn("evening only", output["turns"][5]["response"])


if __name__ == "__main__":
	unittest.main()