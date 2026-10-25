"""Execute scenario.json with the raw Messages API agent and record metrics."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in (None, ""):
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))

from API.agent import StudyPlanningAgent
from API.client import MessagesAPIClient
from API.offline import OfflineMessagesClient


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_scenario(
	scenario_path: Path,
	results_path: Path,
	state_directory: Path,
	client: Any,
	model: str = "claude-3-5-sonnet-20241022",
	data_file: Optional[Path] = None,
) -> Dict[str, Any]:
	"""Run all turns and persist transcript summaries and aggregate metrics."""
	with Path(scenario_path).open("r", encoding="utf-8") as scenario_file:
		scenario = json.load(scenario_file)

	if data_file is not None:
		from shared import tools as shared_tools

		shared_tools.DATA_PATH = Path(data_file)

	turn_results = []
	aggregate = {
		"turns": 0,
		"api_calls": 0,
		"tool_calls": 0,
		"input_tokens": 0,
		"output_tokens": 0,
		"estimated_cost_usd": 0.0,
		"latency_seconds": 0.0,
	}
	for session in scenario.get("sessions", []):
		agent = StudyPlanningAgent(
			client=client,
			model=model,
			state_directory=state_directory,
			session_id=session["session_id"],
		)
		for turn in session.get("turns", []):
			result = agent.run_turn(turn["user"])
			turn_result = {
				"session_id": session["session_id"],
				"turn": turn["turn"],
				"user": turn["user"],
				"response": result["text"],
				"tool_calls": result["tool_calls"],
				"usage": result["usage"],
				"api_calls": result["api_calls"],
				"latency_seconds": result["latency_seconds"],
				"context": result["context"],
			}
			turn_results.append(turn_result)
			aggregate["turns"] += 1
			aggregate["api_calls"] += result["api_calls"]
			aggregate["tool_calls"] += len(result["tool_calls"])
			aggregate["input_tokens"] += result["usage"]["input_tokens"]
			aggregate["output_tokens"] += result["usage"]["output_tokens"]
			aggregate["estimated_cost_usd"] += result["usage"]["estimated_cost_usd"]
			aggregate["latency_seconds"] += result["latency_seconds"]

	aggregate["estimated_cost_usd"] = round(aggregate["estimated_cost_usd"], 8)
	aggregate["latency_seconds"] = round(aggregate["latency_seconds"], 6)
	output = {
		"scenario_id": scenario.get("scenario_id"),
		"implementation": "raw_messages_api",
		"mode": "offline" if isinstance(client, OfflineMessagesClient) else "live",
		"model": model,
		"aggregate_metrics": aggregate,
		"turns": turn_results,
	}
	results_path = Path(results_path)
	results_path.parent.mkdir(parents=True, exist_ok=True)
	with results_path.open("w", encoding="utf-8") as results_file:
		json.dump(output, results_file, indent=2, ensure_ascii=False)
		results_file.write("\n")
	return output


def main(argv: Optional[list] = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--scenario", type=Path, default=PROJECT_ROOT / "scenario.json")
	parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "api.json")
	parser.add_argument("--state-dir", type=Path, default=PROJECT_ROOT / "API" / "state")
	parser.add_argument("--data-file", type=Path, help="Task JSON file to update during the run")
	parser.add_argument("--model", default="claude-3-5-sonnet-20241022")
	parser.add_argument("--offline", action="store_true", help="Run scripted responses without network access")
	parser.add_argument("--live", action="store_true", help="Require a live ANTHROPIC_API_KEY")
	args = parser.parse_args(argv)
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

	use_offline = args.offline or not os.environ.get("ANTHROPIC_API_KEY")
	if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
		parser.error("--live requires ANTHROPIC_API_KEY")
	client = OfflineMessagesClient() if use_offline and not args.live else MessagesAPIClient()
	output = run_scenario(
		args.scenario,
		args.output,
		args.state_dir,
		client,
		args.model,
		data_file=args.data_file,
	)
	print(json.dumps(output["aggregate_metrics"], indent=2))
	print("Wrote {}".format(args.output))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
