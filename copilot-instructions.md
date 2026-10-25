# Project Overview & Guidelines

We are building a Python AI Agent application for managing educational tasks and study planning using two different implementations (Raw Messages API vs. Claude Agent SDK).

## Project Structure
1. `shared/`: Shared domain logic and tools used by both implementations (`tools.py`, `data.json`).
2. `API/`: Raw API implementation. Contains a `harness/` directory where each file handles a single responsibility (`loop.py`, `tools.py`, `history.py`, `memory.py`, `context.py`, `compaction.py`, `caching.py`, `tokens.py`).
3. `SDK/`: SDK implementation built on Claude Agent SDK / Claude Code SDK. Any file replacing a harness responsibility from the API implementation must include the comment `# replaces: <file_name>.py` at the top.
4. `tests/`: Offline test cases using mock clients without requiring live API keys.
5. `scenario.json`: A test scenario covering 2 sessions (6–8 turns total) testing tool usage, session persistence, external calendar tools, and conflicting memory prompts.
6. `results/`: Execution output files (`api.json`, `sdk.json`).

## Coding Standards
- Clean, modular Python code with type hinting.
- In `API/harness/`, each component must log or indicate its execution responsibility clearly.
- Agent Tools:
  1. `get_study_tasks`: Internal tool to retrieve tasks from `shared/data.json`.
  2. `update_task_status`: Internal tool to update or add tasks in `shared/data.json`.
  3. `get_calendar_events`: External tool calling an external calendar API (or simulating an HTTP API call for dates/events).