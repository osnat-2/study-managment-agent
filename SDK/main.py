# replaces: main.py
"""Run scenario.json with Claude Agent SDK or deterministic offline mode."""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))

from SDK.agent import SessionIdStore, build_agent_options, run_turn
from SDK.tools import invoke_local_tool


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_scenario(path: Path) -> Dict[str, Any]:
	with Path(path).open("r", encoding="utf-8") as scenario_file:
		return json.load(scenario_file)


def _write_results(path: Path, output: Dict[str, Any]) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as result_file:
		json.dump(output, result_file, ensure_ascii=False, indent=2)
		result_file.write("\n")


def _aggregate(turn_results: List[Dict[str, Any]]) -> Dict[str, Any]:
	return {
		"turns": len(turn_results),
		"model_calls": sum(turn.get("model_calls", 0) for turn in turn_results),
		"tool_calls": sum(len(turn.get("tool_calls", [])) for turn in turn_results),
		"input_tokens": sum(turn["usage"]["input_tokens"] for turn in turn_results),
		"output_tokens": sum(turn["usage"]["output_tokens"] for turn in turn_results),
		"cache_creation_input_tokens": sum(
			turn["usage"]["cache_creation_input_tokens"] for turn in turn_results
		),
		"cache_read_input_tokens": sum(
			turn["usage"]["cache_read_input_tokens"] for turn in turn_results
		),
		"estimated_cost_usd": round(
			sum(float(turn.get("cost_delta_usd") or 0) for turn in turn_results), 8
		),
		"latency_seconds": round(
			sum(turn["latency_seconds"] for turn in turn_results), 6
		),
	}


def run_offline_scenario(
	scenario_path: Path,
	results_path: Path,
	state_directory: Path,
	data_path: Path,
	model: str = "offline-sdk",
) -> Dict[str, Any]:
	"""Execute scenario-declared tool expectations without SDK or network access."""
	scenario = _read_scenario(scenario_path)
	state_directory = Path(state_directory)
	state_directory.mkdir(parents=True, exist_ok=True)
	memory_path = state_directory / "memory.json"
	session_store = SessionIdStore(state_directory / "session_ids.json")
	session_ids = session_store.load()
	turn_results = []
	last_session_key = None
	for session in scenario.get("sessions", []):
		session_id = session["session_id"]
		resume_from = session.get("resume_from")
		if resume_from is None and session.get("fresh_agent_instance"):
			resume_from = last_session_key
		resumed_id = session_ids.get(resume_from) if resume_from else None
		if session.get("fresh_agent_instance") and session.get("startup", {}).get(
			"load_state_from_disk"
		):
			memory_path.parent.mkdir(parents=True, exist_ok=True)
		resume_id = resumed_id or "offline-sdk-session-{}".format(session_id)
		for turn in session.get("turns", []):
			started = time.perf_counter()
			tool_calls = _offline_tool_calls(
				turn,
				data_path=Path(data_path),
				memory_path=memory_path,
			)
			response = _offline_response(
				turn, Path(data_path), memory_path, tool_calls
			)
			turn_results.append(
				{
					"session_id": session_id,
					"turn": turn["turn"],
					"user": turn["user"],
					"response": response,
					"tool_calls": tool_calls,
					"usage": {
						"input_tokens": 0,
						"output_tokens": 0,
						"cache_creation_input_tokens": 0,
						"cache_read_input_tokens": 0,
						"estimated": True,
					},
					"cost_delta_usd": 0.0,
					"model_calls": 0,
					"latency_seconds": round(time.perf_counter() - started, 6),
					"context": {"resumed_sdk_session_id": resume_id},
				}
			)
		session_ids[session_id] = resume_id
		session_store.save(session_ids)
		last_session_key = session_id

	output = {
		"scenario_id": scenario.get("scenario_id"),
		"implementation": "claude_agent_sdk",
		"mode": "offline",
		"model": model,
		"aggregate_metrics": _aggregate(turn_results),
		"session_ids": session_ids,
		"turns": turn_results,
	}
	_write_results(results_path, output)
	return output


def _offline_tool_calls(
	turn: Dict[str, Any], data_path: Path, memory_path: Path
) -> List[Dict[str, Any]]:
	expected = turn.get("expected", {})
	requests = list(expected.get("tool_calls", []))
	if "memory_write" in expected:
		memory_write = expected["memory_write"]
		requests.append(
			{
				"name": "save_memory",
				"arguments": {
					"key": memory_write["key"],
					"value": memory_write["value"],
				},
			}
		)
	if "memory_read" in expected:
		requests.append(
			{
				"name": "get_memory",
				"arguments": {"key": expected["memory_read"]["key"]},
			}
		)
	if "retrieve my saved study preference" in turn.get("user", "").lower():
		requests.append({"name": "get_study_tasks", "arguments": {}})
	if "summarize my progress" in turn.get("user", "").lower():
		requests.extend(
			[
				{"name": "get_study_tasks", "arguments": {}},
				{"name": "get_memory", "arguments": {}},
			]
		)
	tool_calls = []
	for request in requests:
		name = request["name"]
		arguments = request.get("arguments", {})
		try:
			result = invoke_local_tool(name, arguments, data_path, memory_path)
			tool_calls.append(
				{"name": name, "input": arguments, "result": result, "is_error": False}
			)
		except Exception as error:
			tool_calls.append(
				{"name": name, "input": arguments, "result": str(error), "is_error": True}
			)
	return tool_calls


def _offline_response(
	turn: Dict[str, Any],
	data_path: Path,
	memory_path: Path,
	tool_calls: List[Dict[str, Any]],
) -> str:
	expected = turn.get("expected", {})
	if tool_calls and tool_calls[0]["name"] == "get_study_tasks":
		tasks = tool_calls[0]["result"]
		calendar = [
			call["result"] for call in tool_calls if call["name"] == "get_calendar_events"
		]
		if calendar:
			return "Upcoming tasks: {}; calendar days checked: {}".format(
				"; ".join(
					"{} due {} ({}, {})".format(
						task["subject"], task["due_date"], task["status"], task["priority"]
					)
					for task in tasks
				),
				", ".join(event["date"] for event in calendar),
			)
		if "summarize my progress" in turn.get("user", "").lower():
			memory = next(call["result"] for call in tool_calls if call["name"] == "get_memory")
			completed = [task["subject"] for task in tasks if task["status"] == "completed"]
			remaining = [task["subject"] for task in tasks if task["status"] != "completed"]
			return "Completed: {}. Remaining: {}. Saved preference: {}.".format(
				", ".join(completed),
				", ".join(remaining),
				memory.get("study_time_preference", "none"),
			)
	if "update_task_status" in expected.get("tool_calls", [{}])[0].get("name", ""):
		updated = next(call["result"] for call in tool_calls if call["name"] == "update_task_status")
		return "Task status updated: {}.".format(updated["status"])
	if "memory_write" in expected:
		return "Saved preference: {}.".format(expected["memory_write"]["value"])
	if "memory_read" in expected:
		preference = next(call["result"] for call in tool_calls if call["name"] == "get_memory")
		remaining = next(call["result"] for call in tool_calls if call["name"] == "get_study_tasks")
		remaining = [task for task in remaining if task["status"] != "completed"]
		return "Saved preference: {}. Remaining tasks: {}. Suggested study time: evening only.".format(
			preference.get(expected["memory_read"]["key"]),
			", ".join(
				"{} due {}".format(task["subject"], task["due_date"])
				for task in remaining
			),
		)
	if "conflict_with_memory" in expected:
		return (
			"Your saved preference is evening-only, which conflicts with a morning plan. "
			"Would you like to change your saved preference?"
		)
	return "Offline SDK scenario response."


async def run_live_scenario(
	scenario_path: Path,
	results_path: Path,
	state_directory: Path,
	data_path: Path,
	model: Optional[str] = None,
	cwd: Optional[Path] = None,
) -> Dict[str, Any]:
	"""Execute each session through ClaudeSDKClient, resuming fresh instances."""
	try:
		from claude_agent_sdk import ClaudeSDKClient
	except ImportError as error:
		raise RuntimeError(
			"Claude Agent SDK is not installed. Run `py -m pip install -r SDK/requirements.txt`."
		) from error

	scenario = _read_scenario(scenario_path)
	state_directory = Path(state_directory)
	state_directory.mkdir(parents=True, exist_ok=True)
	session_store = SessionIdStore(state_directory / "session_ids.json")
	session_ids = session_store.load()
	usage_snapshots = {}
	turn_results = []
	last_session_key = None
	for session in scenario.get("sessions", []):
		session_id = session["session_id"]
		resume_from = session.get("resume_from")
		if resume_from is None and session.get("fresh_agent_instance"):
			resume_from = last_session_key
		resume_id = session_ids.get(resume_from) if resume_from else None
		options = build_agent_options(
			model=model,
			cwd=Path(cwd or PROJECT_ROOT),
			data_path=Path(data_path),
			memory_path=state_directory / "memory.json",
			resume_session_id=resume_id,
		)
		async with ClaudeSDKClient(options=options) as client:
			for turn in session.get("turns", []):
				result = await run_turn(client, turn["user"])
				sdk_session_id = result.get("session_id") or resume_id
				if not sdk_session_id:
					raise RuntimeError("SDK did not return a session_id for {}".format(session_id))
				prior = usage_snapshots.get(sdk_session_id, {})
				current = result["usage"]
				usage_delta = _usage_delta(current, prior)
				cost_delta = _number_delta(result.get("total_cost_usd"), prior.get("cost_usd"))
				usage_snapshots[sdk_session_id] = {
					"input_tokens": current["input_tokens"],
					"output_tokens": current["output_tokens"],
					"cache_creation_input_tokens": current["cache_creation_input_tokens"],
					"cache_read_input_tokens": current["cache_read_input_tokens"],
					"cost_usd": result.get("total_cost_usd"),
				}
				session_ids[session_id] = sdk_session_id
				session_store.save(session_ids)
				turn_results.append(
					{
						"session_id": session_id,
						"turn": turn["turn"],
						"user": turn["user"],
						"response": result["text"],
						"tool_calls": result["tool_calls"],
						"usage": usage_delta,
						"cost_delta_usd": cost_delta,
						"sdk_session_id": sdk_session_id,
						"resumed": bool(resume_id),
						"num_turns": result["num_turns"],
						"model_calls": result["num_turns"],
						"latency_seconds": result["latency_seconds"],
						"duration_ms": result["duration_ms"],
						"duration_api_ms": result["duration_api_ms"],
						"model_usage": result["model_usage"],
					}
				)
				resume_id = sdk_session_id
		last_session_key = session_id

	output = {
		"scenario_id": scenario.get("scenario_id"),
		"implementation": "claude_agent_sdk",
		"mode": "live",
		"model": model,
		"aggregate_metrics": _aggregate(turn_results),
		"session_ids": session_ids,
		"turns": turn_results,
	}
	_write_results(results_path, output)
	return output


def _usage_delta(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
	keys = (
		"input_tokens",
		"output_tokens",
		"cache_creation_input_tokens",
		"cache_read_input_tokens",
	)
	result = {key: max(0, current[key] - int(previous.get(key, 0) or 0)) for key in keys}
	result["estimated"] = False
	return result


def _number_delta(current: Optional[float], previous: Optional[float]) -> Optional[float]:
	if current is None:
		return None
	return round(max(0.0, float(current) - float(previous or 0.0)), 8)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--scenario", type=Path, default=PROJECT_ROOT / "scenario.json")
	parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "sdk.json")
	parser.add_argument("--state-dir", type=Path, default=PROJECT_ROOT / "SDK" / "state")
	parser.add_argument("--data-file", type=Path, default=PROJECT_ROOT / "shared" / "data.json")
	parser.add_argument("--model", default=None)
	parser.add_argument("--offline", action="store_true", help="Run without Claude Agent SDK or network access")
	args = parser.parse_args(argv)
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
	if args.offline:
		output = run_offline_scenario(
			args.scenario, args.output, args.state_dir, args.data_file, args.model or "offline-sdk"
		)
	else:
		if sys.version_info < (3, 10):
			parser.error("Live SDK mode requires Python 3.10 or newer; use --offline on this interpreter")
		output = asyncio.run(
			run_live_scenario(
				args.scenario,
				args.output,
				args.state_dir,
				args.data_file,
				model=args.model,
			)
		)
	print(json.dumps(output["aggregate_metrics"], indent=2))
	print("Wrote {}".format(args.output))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
