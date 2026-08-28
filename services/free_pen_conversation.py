from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class FreePenConversationMemory:
    system_message: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_turns: int = 30

    def append_user_message(self, content: Any) -> None:
        self.messages.append({"role": "user", "content": content})

    def append_assistant_message(self, response: dict[str, Any]) -> None:
        if response.get("role") == "assistant" and isinstance(response.get("tool_calls"), list):
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.get("content"),
                    "tool_calls": list(response["tool_calls"]),
                }
            )
        else:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(response, ensure_ascii=False),
                }
            )

    def append_tool_result(self, tool_call_id: str, content: dict[str, Any]) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(content, ensure_ascii=False),
            }
        )


    def build_messages_for_request(self) -> list[dict[str, Any]]:
        trimmed_messages = self._trimmed_messages()
        return [self.system_message, *trimmed_messages]

    def save(
        self,
        path: Path,
        *,
        message_sanitizer: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    ) -> None:
        system_message = self.system_message
        messages = self.messages
        if message_sanitizer is not None:
            system_message = message_sanitizer([self.system_message])[0]
            messages = message_sanitizer(self.messages)
        payload = {
            "system_message": system_message,
            "messages": messages,
            "max_turns": self.max_turns,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _trimmed_messages(self) -> list[dict[str, Any]]:
        if self.max_turns <= 0:
            return []
        assistant_seen = 0
        start_index = 0
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].get("role") == "assistant":
                assistant_seen += 1
                if assistant_seen > self.max_turns:
                    start_index = index + 1
                    break
        return self.messages[start_index:]


__all__ = ["FreePenConversationMemory"]
