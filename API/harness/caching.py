"""Add ephemeral prompt-cache breakpoints to stable request prefixes."""

import copy
import logging
from typing import Any, Dict, List, Tuple, Union


LOGGER = logging.getLogger(__name__)
SystemPrompt = Union[str, List[Dict[str, Any]]]


def apply_cache_control(
	system: SystemPrompt,
	tools: List[Dict[str, Any]],
	enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	"""Return API-ready system blocks and tools, caching their stable prefixes."""
	if isinstance(system, str):
		blocks = [{"type": "text", "text": system}]
	else:
		blocks = copy.deepcopy(system)
	tool_definitions = copy.deepcopy(tools)
	if not enabled:
		return blocks, tool_definitions

	if blocks:
		blocks[0]["cache_control"] = {"type": "ephemeral"}
	if tool_definitions:
		tool_definitions[-1]["cache_control"] = {"type": "ephemeral"}
	LOGGER.debug("Applied cache control to system and tool prefixes")
	return blocks, tool_definitions
