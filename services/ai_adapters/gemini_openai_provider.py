from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from services.ai_adapters.base import VisionReviewAdapter
from services.ai_adapters.common import (
    MAX_REVIEW_IMAGE_BYTES,
    ProviderConfigurationError,
    collect_image_paths,
    encode_image_as_data_url,
    get_review_messages,
    load_response_schema,
    parse_json_response_text,
    resolve_api_key,
)

if TYPE_CHECKING:
    from services.ai_agent import AIReviewInput


@dataclass(slots=True)
class GeminiOpenAICompatibleVisionAdapter(VisionReviewAdapter):
    model: str = "gemini-3-flash-preview"
    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    client: Any | None = None
    image_detail: str = "auto"
    max_image_bytes: int = MAX_REVIEW_IMAGE_BYTES
    timeout_seconds: float | None = None
    response_schema: dict[str, Any] | None = None
    response_schema_path: Path | None = None

    def review(self, prompt: str, review_input: AIReviewInput) -> dict[str, Any]:
        print(f"review GeminiOpenAICompatibleVisionAdapter with model={self.model}, base_url={self.base_url}, image_detail={self.image_detail}")

        client = self._resolve_client()
        review_messages = get_review_messages(review_input)
        tools = tuple(getattr(review_input, "tools", ()) or ())
        tool_choice = getattr(review_input, "tool_choice", None)
        if review_messages is not None:
            request_messages = [self._convert_message(message) for message in review_messages]
        else:
            content: list[dict[str, Any]] = []
            for image_path in collect_image_paths(review_input, max_image_bytes=self.max_image_bytes):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image_as_data_url(image_path, max_image_bytes=self.max_image_bytes),
                            "detail": self.image_detail,
                        },
                    }
                )
            content.append({"type": "text", "text": prompt})
            request_messages = [{"role": "user", "content": content}]

        kwargs_create = {
            "model": self.model,
            "messages": request_messages,
            "timeout": self.timeout_seconds,
        }

        if tools:
            kwargs_create["tools"] = list(tools)
            if tool_choice and str(tool_choice).strip().lower() != "none":
                kwargs_create["tool_choice"] = tool_choice
        else:
            response_schema = self.response_schema or load_response_schema(self.response_schema_path)
            if response_schema:
                kwargs_create["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ai_review_response",
                        "schema": response_schema,
                        "strict": True,
                    },
                }

        response = client.chat.completions.create(**kwargs_create)

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("gemini provider response does not contain choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError("gemini provider response does not contain a message")

        tool_calls = getattr(message, "tool_calls", None)
        if tools:
            if tool_calls and len(tool_calls) == 1:
                tool_call = tool_calls[0]
                function_obj = getattr(tool_call, "function", None)
                if function_obj is None:
                    raise ValueError("tool_call does not contain function")

                name = getattr(function_obj, "name", "")
                arguments = getattr(function_obj, "arguments", "")

                from services.free_pen_native_tools import parse_native_tool_call
                parsed = parse_native_tool_call(name, arguments)
                raw_tool_call_dict = {
                    "id": getattr(tool_call, "id", None),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
                parsed["_parsed_from"] = "native_tool_call"
                parsed["_raw_tool_calls"] = [raw_tool_call_dict]
                parsed["_assistant_message"] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [raw_tool_call_dict],
                }
                parsed["_raw_response"] = {
                    "tool_calls": [raw_tool_call_dict],
                    "content": None,
                }
                return parsed
            if tool_calls and len(tool_calls) > 1:
                raise ValueError("too_many_tool_calls")
            raise ValueError("missing_native_tool_calls")

        # Fallback for non-tool-use cases (e.g., AI Review)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return parse_json_response_text(content, provider_name="gemini")
        raise ValueError("gemini provider response does not contain content or tool_calls")

    def _convert_message(self, message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role", "user")).strip().lower() or "user"
            # Native assistant tool_calls message.
        if role == "assistant" and "tool_calls" in message:
            return {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message["tool_calls"],
            }
            # Native tool result message.
        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": str(message["tool_call_id"]),
                "content": str(message.get("content", "")),
            }
    
        content = message.get("content")
        if isinstance(content, str):
            return {"role": role, "content": [{"type": "text", "text": content}]}
        if not isinstance(content, list):
            raise ValueError("Gemini review_input.messages content must be a string or list")
        converted_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Gemini review_input.messages parts must be dicts")
            part_type = str(part.get("type", "")).strip().lower()
            if part_type == "text":
                converted_parts.append({"type": "text", "text": str(part.get("text", ""))})
                continue
            if part_type == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
                    raise ValueError("Gemini image_url parts must contain image_url.url")
                converted_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(image_url["url"]),
                            "detail": str(image_url.get("detail", self.image_detail)),
                        },
                    }
                )
                continue
            raise ValueError(f"unsupported Gemini message part type: {part_type}")
        return {"role": role, "content": converted_parts}

    def _resolve_client(self) -> Any:
        if self.client is not None:
            return self.client

        api_key = self.api_key or resolve_api_key("GEMINI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                "Gemini provider requires GEMINI_API_KEY or an explicit api_key"
            )

        try:
            module = importlib.import_module("openai")
            client_class = getattr(module, "OpenAI")
        except (ImportError, AttributeError) as exc:
            raise ProviderConfigurationError(
                "Gemini provider requires the optional `openai` package"
            ) from exc

        return client_class(api_key=api_key, base_url=self.base_url)

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("gemini provider response does not contain choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError("gemini provider response does not contain a message")

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        raise ValueError("gemini provider response does not contain text content")


__all__ = ["GeminiOpenAICompatibleVisionAdapter"]
