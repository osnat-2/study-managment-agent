"""Disk-backed key/value memory shared by independent agent instances."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


class MemoryStore:
	"""Load and atomically persist long-term user memory as JSON."""

	def __init__(self, path: Path) -> None:
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)

	def all(self) -> Dict[str, Any]:
		if not self.path.exists():
			return {}
		with self.path.open("r", encoding="utf-8") as memory_file:
			memory = json.load(memory_file)
		if not isinstance(memory, dict):
			raise ValueError("Long-term memory must contain a JSON object")
		LOGGER.debug("Loaded %d persistent memory entries", len(memory))
		return memory

	def get(self, key: Optional[str] = None) -> Any:
		memory = self.all()
		return memory if key is None else memory.get(key)

	def set(self, key: str, value: Any) -> None:
		memory = self.all()
		memory[key] = value
		self.path.parent.mkdir(parents=True, exist_ok=True)
		file_descriptor, temporary_path = tempfile.mkstemp(
			prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
		)
		try:
			with os.fdopen(file_descriptor, "w", encoding="utf-8") as memory_file:
				json.dump(memory, memory_file, ensure_ascii=False, indent=2)
				memory_file.write("\n")
			os.replace(temporary_path, str(self.path))
		except Exception:
			if os.path.exists(temporary_path):
				os.remove(temporary_path)
			raise
		LOGGER.info("Saved persistent memory key %s", key)
