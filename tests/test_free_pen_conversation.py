from __future__ import annotations

import json
from pathlib import Path

from services.free_pen_conversation import FreePenConversationMemory
from services.free_pen_prompt import build_free_pen_tool_system_prompt


def test_system_prompt_is_first_message() -> None:
    memory = FreePenConversationMemory(
        system_message={"role": "system", "content": build_free_pen_tool_system_prompt()},
        max_turns=3,
    )
    memory.append_user_message("round 1 user")
    memory.append_assistant_message({"decision": "tool_call", "tool_call": {"tool": "inspect_history", "last_n": 8}})

    messages = memory.build_messages_for_request()

    assert messages[0]["role"] == "system"
    assert "Allowed tools:" in messages[0]["content"]
    assert "Do not trace the overlay." in messages[0]["content"]


def test_conversation_appends_assistant_response(tmp_path: Path) -> None:
    memory = FreePenConversationMemory(
        system_message={"role": "system", "content": build_free_pen_tool_system_prompt()},
        max_turns=3,
    )
    memory.append_user_message("round 1 user")
    memory.append_assistant_message({"decision": "stalled", "reason": "need more context"})
    output_path = tmp_path / "conversation_messages.json"
    memory.save(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["messages"][-1]["role"] == "assistant"
    assert '"decision": "stalled"' in payload["messages"][-1]["content"]


def test_conversation_sliding_window_configurable() -> None:
    memory = FreePenConversationMemory(
        system_message={"role": "system", "content": build_free_pen_tool_system_prompt()},
        max_turns=2,
    )
    for index in range(1, 5):
        memory.append_user_message(f"user {index}")
        memory.append_assistant_message({"decision": "tool_call", "tool_call": {"tool": "inspect_history", "last_n": index}})

    messages = memory.build_messages_for_request()
    assistant_messages = [message for message in messages if message["role"] == "assistant"]

    assert len(assistant_messages) == 2
    assert "last_n\": 3" in assistant_messages[0]["content"]
    assert "last_n\": 4" in assistant_messages[1]["content"]
