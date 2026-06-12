from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    raw: dict[str, Any] | None = None


class UnifiedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    tenant_id: str
    model: str
    messages: list[ChatMessage]
    stream: bool
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    logprobs: bool = False
    prompt_version: str = "default"
    has_tools: bool = False
    raw_body: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class UnifiedDelta(BaseModel):
    content: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    raw: dict[str, Any] | None = None


class UnifiedResponse(BaseModel):
    content: str
    finish_reason: str
    usage: Usage
    model: str
    mean_logprob: float | None = None


class ChatCompletionIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    logprobs: bool = False
    user: str | None = None
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def normalized_stop(self) -> list[str] | None:
        if self.stop is None:
            return None
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop

    def include_stream_usage(self) -> bool:
        if self.stream_options is None:
            return False
        return self.stream_options.get("include_usage") is True
