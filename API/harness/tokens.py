"""Track Messages API token usage and estimate per-turn costs."""

import logging
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)

MODEL_PRICES_PER_MILLION = {
	"claude-3-5-sonnet-20241022": {
		"input": 3.0,
		"output": 15.0,
		"cache_creation": 3.75,
		"cache_read": 0.3,
	},
	"claude-3-7-sonnet-20250219": {
		"input": 3.0,
		"output": 15.0,
		"cache_creation": 3.75,
		"cache_read": 0.3,
	},
	"claude-sonnet-4-20250514": {
		"input": 3.0,
		"output": 15.0,
		"cache_creation": 3.75,
		"cache_read": 0.3,
	},
}


class TokenUsageTracker:
	"""Aggregate API-reported or approximate token usage for one turn."""

	def __init__(self, model: str) -> None:
		self.model = model
		self.input_tokens = 0
		self.output_tokens = 0
		self.cache_creation_input_tokens = 0
		self.cache_read_input_tokens = 0
		self.estimated = False

	def record(
		self,
		usage: Optional[Any],
		estimated_input_tokens: int = 0,
		estimated_output_tokens: int = 0,
	) -> None:
		"""Add one response's token usage, falling back to rough text estimates."""
		if usage is None:
			self.input_tokens += estimated_input_tokens
			self.output_tokens += estimated_output_tokens
			self.estimated = True
			LOGGER.debug("Recorded estimated token usage for %s", self.model)
			return

		self.input_tokens += self._read(usage, "input_tokens")
		self.output_tokens += self._read(usage, "output_tokens")
		self.cache_creation_input_tokens += self._read(
			usage, "cache_creation_input_tokens"
		)
		self.cache_read_input_tokens += self._read(usage, "cache_read_input_tokens")

	def to_dict(self) -> Dict[str, Any]:
		"""Return totals and an estimated USD cost."""
		prices = MODEL_PRICES_PER_MILLION.get(
			self.model, MODEL_PRICES_PER_MILLION["claude-3-5-sonnet-20241022"]
		)
		cost = (
			self.input_tokens * prices["input"]
			+ self.output_tokens * prices["output"]
			+ self.cache_creation_input_tokens * prices["cache_creation"]
			+ self.cache_read_input_tokens * prices["cache_read"]
		) / 1_000_000
		return {
			"input_tokens": self.input_tokens,
			"output_tokens": self.output_tokens,
			"cache_creation_input_tokens": self.cache_creation_input_tokens,
			"cache_read_input_tokens": self.cache_read_input_tokens,
			"total_tokens": (
				self.input_tokens
				+ self.output_tokens
				+ self.cache_creation_input_tokens
				+ self.cache_read_input_tokens
			),
			"estimated": self.estimated,
			"estimated_cost_usd": round(cost, 8),
		}

	@staticmethod
	def _read(usage: Any, key: str) -> int:
		if isinstance(usage, dict):
			value = usage.get(key, 0)
		else:
			value = getattr(usage, key, 0)
		return int(value or 0)
