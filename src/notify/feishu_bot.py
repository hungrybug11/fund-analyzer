"""Feishu group bot webhook client.

Supports sending text, rich text (post), and interactive card messages
via Feishu (飞书) group bot webhook API.

API docs: https://open.feishu.cn/document/uAjLw4CM/ukzMukzMukzM/feishu-bot/bot-overview
"""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class FeishuError(Exception):
    """Raised when Feishu API returns an error."""


class FeishuBot:
    """A simple Feishu group bot that sends messages via webhook."""

    def __init__(self, webhook_url: str, timeout: int = 15):
        """
        Args:
            webhook_url: Full webhook URL from Feishu group bot settings.
                         Format: https://open.feishu.cn/open-apis/bot/v2/hook/{token}
            timeout: HTTP request timeout in seconds.
        """
        if not webhook_url.startswith("https://open.feishu.cn/open-apis/bot/v2/hook/"):
            raise ValueError(f"Invalid Feishu webhook URL: {webhook_url[:50]}...")
        self._webhook_url = webhook_url
        self._timeout = timeout

    # ── Public API ──────────────────────────────────────────────────

    def send_text(self, text: str) -> dict[str, Any]:
        """Send a plain text message.

        Args:
            text: Message content (supports @user syntax via <at user_id></at>).

        Returns:
            API response dict (typically {"code": 0, "msg": "success"}).
        """
        payload = {
            "msg_type": "text",
            "content": {"text": text},
        }
        return self._post(payload)

    def send_post(self, title: str, content_lines: list[list[dict]], zh_locale: bool = True) -> dict[str, Any]:
        """Send a rich-text (post) message.

        Args:
            title: Post title (shown as bold heading).
            content_lines: List of lines, each line is a list of inline elements.
                           Each element: {"tag": "text", "text": "..."} or
                                         {"tag": "a", "text": "...", "href": "..."}
            zh_locale: Whether to send as zh_cn (default) or en_us.

        Returns:
            API response dict.
        """
        locale = "zh_cn" if zh_locale else "en_us"
        payload = {
            "msg_type": "post",
            "content": {
                "post": {
                    locale: {
                        "title": title,
                        "content": content_lines,
                    }
                }
            },
        }
        return self._post(payload)

    def send_card(self, card: dict[str, Any]) -> dict[str, Any]:
        """Send an interactive card message.

        Args:
            card: The card JSON object (https://open.feishu.cn/document/uAjLw4CM/ukzMukzMukzM/feishu-cards/card-components).

        Returns:
            API response dict.
        """
        payload = {
            "msg_type": "interactive",
            "card": card,
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
            logger.debug("Feishu webhook response %s: %s", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            raise FeishuError(f"HTTP request failed: {e}") from e

        if resp.status_code != 200:
            raise FeishuError(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        code = data.get("code", -1)
        if code != 0:
            msg = data.get("msg", data.get("message", "unknown error"))
            raise FeishuError(f"API error (code={code}): {msg}")

        return data
