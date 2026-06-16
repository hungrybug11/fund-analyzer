"""WeCom (企业微信) group bot webhook client.

Supports sending text and markdown messages via WeCom group bot webhook API.

API docs: https://developer.work.weixin.qq.com/document/path/91770
"""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class WecomError(Exception):
    """Raised when WeCom API returns an error."""


class WecomBot:
    """A simple WeCom group bot that sends messages via webhook."""

    def __init__(self, webhook_url: str, timeout: int = 15):
        """
        Args:
            webhook_url: Full webhook URL from WeCom group bot settings.
                         Format: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
            timeout: HTTP request timeout in seconds.
        """
        if not webhook_url.startswith("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="):
            raise ValueError(f"Invalid WeCom webhook URL: {webhook_url[:60]}...")
        self._webhook_url = webhook_url
        self._timeout = timeout

    # ── Public API ──────────────────────────────────────────────────

    def send_text(self, text: str, mentioned_list: list[str] | None = None) -> dict[str, Any]:
        """Send a plain text message.

        Args:
            text: Message content.
            mentioned_list: List of user IDs to @mention, or ["@all"] for everyone.

        Returns:
            API response dict.
        """
        payload: dict[str, Any] = {
            "msgtype": "text",
            "text": {"content": text},
        }
        if mentioned_list:
            payload["text"]["mentioned_list"] = mentioned_list
        return self._post(payload)

    def send_markdown(self, content: str) -> dict[str, Any]:
        """Send a markdown message.

        Args:
            content: Markdown-formatted content. Supports:
                     # ~ ### headers, **bold**, *italic*,
                     [link](url), >quote, - list, 1. ordered list.

        Returns:
            API response dict.
        """
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
        return self._post(payload)

    # ── Internal ────────────────────────────────────────────────────

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request to the webhook URL."""
        try:
            resp = requests.post(
                self._webhook_url,
                json=payload,
                timeout=self._timeout,
            )
            logger.debug("WeCom webhook response %s: %s", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            raise WecomError(f"HTTP request failed: {e}") from e

        if resp.status_code != 200:
            raise WecomError(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        code = data.get("errcode", -1)
        if code != 0:
            msg = data.get("errmsg", "unknown error")
            raise WecomError(f"API error (errcode={code}): {msg}")

        return data
