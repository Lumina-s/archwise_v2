from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


class OpenAICompatibleStructuredChat:
    def __init__(
        self,
        llm_client: Any,
        *,
        temperature: float,
        max_tokens: int,
        request_timeout: float | None | object,
    ) -> None:
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

    def with_structured_output(self, schema: type[BaseModel] | dict[str, Any], **_: Any) -> Runnable[Any, Any]:
        parser = (
            PydanticOutputParser(pydantic_object=schema)
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else JsonOutputParser()
        )

        async def ainvoke(input_value: Any) -> Any:
            messages = self._normalize_messages(input_value)
            payload: dict[str, Any] = {
                "model": self.llm_client.model,
                "messages": messages,
                "thinking": {"type": "disabled"},
                "temperature": self.temperature,
                "stream": False,
                "response_format": {"type": "json_object"},
                "max_tokens": self.max_tokens,
            }
            content = await self.llm_client._chat(payload, request_timeout=self.request_timeout)
            if content is None:
                raise RuntimeError(self.llm_client.last_error or "Structured LLM call returned empty content.")
            return parser.parse(content)

        return RunnableLambda(ainvoke)

    @staticmethod
    def _normalize_messages(input_value: Any) -> list[dict[str, str]]:
        if hasattr(input_value, "to_messages"):
            raw_messages = input_value.to_messages()
        elif isinstance(input_value, Sequence) and not isinstance(input_value, str):
            raw_messages = input_value
        else:
            raw_messages = [("user", str(input_value))]

        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if isinstance(item, dict):
                role = str(item["role"])
                content = str(item["content"])
            elif isinstance(item, tuple) and len(item) == 2:
                role = str(item[0])
                content = str(item[1])
            else:
                role = OpenAICompatibleStructuredChat._message_role(item)
                content = str(getattr(item, "content"))
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _message_role(message: Any) -> str:
        message_type = str(getattr(message, "type", "user"))
        return {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "chat": "user",
        }.get(message_type, message_type)
