from __future__ import annotations

import json
from pathlib import Path

from services.free_pen_conversation import FreePenConversationMemory
from services.free_pen_prompt import build_free_pen_tool_system_prompt


def _assistant_tool_message() -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "start_path",
                    "arguments": json.dumps({"x": 12, "y": 52, "reason": "start"}, ensure_ascii=False),
                },
            }
        ],
    }


def test_system_prompt_is_first_message() -> None:
    memory = FreePenConversationMemory(
        system_message={"role": "system", "content": build_free_pen_tool_system_prompt()},
        max_turns=3,
    )
    memory.append_user_message("round 1 user")
    memory.append_assistant_message(_assistant_tool_message())

    messages = memory.build_messages_for_request()

    assert messages[0]["role"] == "system"
    assert "Use function tool calls only" in messages[0]["content"]
    assert "Do not write JSON manually" in messages[0]["content"]
    assert "Do not trace the overlay." in messages[0]["content"]
    assert "Return JSON only" not in messages[0]["content"]
    assert '"decision"' not in messages[0]["content"]
    assert '"tool_call"' not in messages[0]["content"]


def test_conversation_appends_assistant_tool_calls(tmp_path: Path) -> None:
    memory = FreePenConversationMemory(
        system_message={"role": "system", "content": build_free_pen_tool_system_prompt()},
        max_turns=3,
    )
    memory.append_user_message("round 1 user")
    memory.append_assistant_message(_assistant_tool_message())
    output_path = tmp_path / "conversation_messages.json"
    memory.save(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["messages"][-1]["role"] == "assistant"
    assert payload["messages"][-1]["content"] is None
    assert payload["messages"][-1]["tool_calls"][0]["function"]["name"] == "start_path"


def test_conversation_sliding_window_configurable() -> None:
    memory = FreePenConversationMemory(
        system_message={"role": "system", "content": build_free_pen_tool_system_prompt()},
        max_turns=2,
    )
    for index in range(1, 5):
        memory.append_user_message(f"user {index}")
        memory.append_assistant_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{index:03d}",
                        "type": "function",
                        "function": {
                            "name": "inspect_history",
                            "arguments": json.dumps({"last_n": index, "reason": "inspect"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        )

    messages = memory.build_messages_for_request()
    assistant_messages = [message for message in messages if message["role"] == "assistant"]

    assert len(assistant_messages) == 2
    assert assistant_messages[0]["tool_calls"][0]["function"]["arguments"] == '{"last_n": 3, "reason": "inspect"}'
    assert assistant_messages[1]["tool_calls"][0]["function"]["arguments"] == '{"last_n": 4, "reason": "inspect"}'
