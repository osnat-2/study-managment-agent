"""Append-only JSONL storage for per-session conversation history."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List


LOGGER = logging.getLogger(__name__)


class HistoryStore:
	"""Persist each conversation message as one immutable JSONL record."""

	def __init__(self, directory: Path, session_id: str) -> None:
		safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)
		self.path = Path(directory) / (safe_id + ".jsonl")
		self.path.parent.mkdir(parents=True, exist_ok=True)

	def load(self) -> List[Dict[str, Any]]:
		"""Load all messages previously appended for this session."""
		if not self.path.exists():
			return []
		messages = []
		with self.path.open("r", encoding="utf-8") as history_file:
			for line_number, line in enumerate(history_file, 1):
				if line.strip():
					try:
						messages.append(json.loads(line))
					except json.JSONDecodeError as error:
						raise ValueError(
							"Invalid conversation history at line {} in {}".format(
								line_number, self.path
							)
						) from error
		LOGGER.debug("Loaded %d messages from %s", len(messages), self.path)
		return messages

	def append(self, message: Dict[str, Any]) -> None:
		"""Append one message without rewriting prior conversation records."""
		with self.path.open("a", encoding="utf-8") as history_file:
			history_file.write(json.dumps(message, ensure_ascii=False) + "\n")
		LOGGER.debug("Appended %s message to %s", message.get("role"), self.path)
