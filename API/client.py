"""Minimal synchronous Anthropic Messages API client using the stdlib."""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class MessagesAPIError(RuntimeError):
	"""Raised when the Anthropic Messages API returns an error."""


class MessagesAPIClient:
	"""Submit raw requests to Anthropic's Messages API endpoint."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		base_url: str = "https://api.anthropic.com/v1/messages",
		timeout: int = 90,
	) -> None:
		self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
		if not self.api_key:
			raise ValueError("Set ANTHROPIC_API_KEY or use --offline")
		self.base_url = base_url
		self.timeout = timeout

	class _Messages:
		def __init__(self, client: "MessagesAPIClient") -> None:
			self.client = client

		def create(self, **request_data: Any) -> Dict[str, Any]:
			return self.client.create(**request_data)

	@property
	def messages(self) -> "MessagesAPIClient._Messages":
		return self._Messages(self)

	def create(self, **request_data: Any) -> Dict[str, Any]:
		payload = json.dumps(request_data).encode("utf-8")
		request = urllib.request.Request(
			self.base_url,
			data=payload,
			headers={
				"content-type": "application/json",
				"anthropic-version": "2023-06-01",
				"x-api-key": self.api_key,
			},
			method="POST",
		)
		try:
			with urllib.request.urlopen(request, timeout=self.timeout) as response:
				return json.loads(response.read().decode("utf-8"))
		except urllib.error.HTTPError as error:
			body = error.read().decode("utf-8", errors="replace")
			raise MessagesAPIError(
				"Messages API returned HTTP {}: {}".format(error.code, body)
			) from error
		except urllib.error.URLError as error:
			raise MessagesAPIError("Could not reach Messages API: {}".format(error)) from error