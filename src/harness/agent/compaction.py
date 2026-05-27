from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, model_validator

from harness.models import ModelMessage, ModelProvider
from harness.tools.redaction import redact

COMPACTION_SUMMARY_PREFIX = "Prior conversation summary:"
MIN_COMPACTION_SUMMARY_CHARS = len(COMPACTION_SUMMARY_PREFIX) + 4


class ContextCompactionError(RuntimeError):
    pass


class ContextCompactionPolicy(BaseModel):
    enabled: bool = True
    max_context_chars: int = Field(default=120_000, ge=1)
    trigger_ratio: float = Field(default=0.8, gt=0.0, le=1.0)
    preserve_recent_messages: int = Field(default=8, ge=1)
    summarizer_input_max_chars: int = Field(default=24_000, ge=1)
    summary_max_chars: int = Field(default=4_000, ge=1)

    @property
    def trigger_chars(self) -> int:
        return max(1, int(self.max_context_chars * self.trigger_ratio))


class ContextCompactionRequest(BaseModel):
    session_id: str
    messages: list[ModelMessage]
    iteration: int
    protected_message_count: int = Field(default=0, ge=0)
    pinned_messages: list[ModelMessage] = Field(default_factory=list)
    redaction_secrets: dict[str, str] = Field(default_factory=dict)


class ContextCompactionResult(BaseModel):
    messages: list[ModelMessage]
    compacted: bool = False
    original_chars: int = 0
    compacted_chars: int = 0
    summarized_messages: int = 0
    summary: str | None = None

    @model_validator(mode="after")
    def require_summary_when_compacted(self) -> ContextCompactionResult:
        if self.compacted and not self.summary:
            raise ValueError("summary is required when compacted=true")
        if self.compacted and self.summary:
            if not self.summary.startswith(COMPACTION_SUMMARY_PREFIX):
                raise ValueError(
                    f"summary must start with {COMPACTION_SUMMARY_PREFIX!r}"
                )
            matching_summaries = [
                message
                for message in self.messages
                if message.role == "system" and message.content == self.summary
            ]
            if len(matching_summaries) != 1:
                raise ValueError(
                    "compacted messages must include exactly one system summary "
                    "message matching summary"
                )
        return self


class ContextCompactor(ABC):
    @abstractmethod
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        raise NotImplementedError


class NoopContextCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        original_chars = estimate_message_chars(request.messages)
        return ContextCompactionResult(
            messages=request.messages,
            original_chars=original_chars,
            compacted_chars=original_chars,
        )


COMPACTION_SYSTEM_PROMPT = """Create a checkpoint summary for an agent context window.

Preserve user goals, constraints, decisions, unresolved tasks, tool results that
affect future behavior, failures, attempted fixes, and important artifact or
file references. Do not include secrets or credentials. Be concise and write a
standalone summary that can replace the selected raw turns."""


class RollingSummaryContextCompactor(ContextCompactor):
    def __init__(
        self,
        model: ModelProvider,
        policy: ContextCompactionPolicy | None = None,
    ) -> None:
        self.model = model
        self.policy = policy or ContextCompactionPolicy()

    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        original_chars = estimate_message_chars(request.messages)
        if not self.policy.enabled or original_chars <= self.policy.trigger_chars:
            return ContextCompactionResult(
                messages=request.messages,
                original_chars=original_chars,
                compacted_chars=original_chars,
            )

        leading, rest = _split_protected_messages(
            request.messages,
            protected_message_count=request.protected_message_count,
        )
        if len(rest) <= self.policy.preserve_recent_messages:
            return ContextCompactionResult(
                messages=request.messages,
                original_chars=original_chars,
                compacted_chars=original_chars,
            )

        pinned, rest_without_pinned = _extract_pinned_messages(rest, request.pinned_messages)
        tail = rest_without_pinned[-self.policy.preserve_recent_messages :]
        older = rest_without_pinned[: -self.policy.preserve_recent_messages]
        if not older:
            return ContextCompactionResult(
                messages=request.messages,
                original_chars=original_chars,
                compacted_chars=original_chars,
            )
        summary_source = [*older, *tail, *pinned]
        summary = await self._summarize(
            summary_source,
            secrets=request.redaction_secrets,
        )
        summary_message = ModelMessage(role="system", content=summary)
        compacted_messages = [
            *leading,
            summary_message,
            *tail,
            *pinned,
        ]
        compacted_chars = estimate_message_chars(compacted_messages)
        if compacted_chars > self.policy.max_context_chars:
            compacted_messages, summary_message = _fit_to_budget(
                leading=leading,
                summary=summary_message,
                pinned=pinned,
                tail=tail,
                max_chars=self.policy.max_context_chars,
            )
            compacted_chars = estimate_message_chars(compacted_messages)
        return ContextCompactionResult(
            messages=compacted_messages,
            compacted=True,
            original_chars=original_chars,
            compacted_chars=compacted_chars,
            summarized_messages=len(summary_source),
            summary=summary_message.content,
        )

    async def _summarize(self, messages: list[ModelMessage], *, secrets: dict[str, str]) -> str:
        transcript = _bounded_transcript(
            messages,
            secrets=secrets,
            max_chars=self.policy.summarizer_input_max_chars,
        )
        response = await self.model.complete(
            [
                ModelMessage(role="system", content=COMPACTION_SYSTEM_PROMPT),
                ModelMessage(
                    role="user",
                    content=(
                        "Summarize these selected turns for future context:\n\n"
                        f"{transcript}"
                    ),
                ),
            ]
        )
        summary = _single_line(redact(response.content, secrets))
        if len(summary) > self.policy.summary_max_chars:
            summary = summary[: max(0, self.policy.summary_max_chars - 3)].rstrip() + "..."
        return redact(f"{COMPACTION_SUMMARY_PREFIX}\n{summary}", secrets)


def estimate_message_chars(messages: list[ModelMessage]) -> int:
    return sum(len(message.role) + len(message.content) + 2 for message in messages)


def _split_protected_messages(
    messages: list[ModelMessage],
    *,
    protected_message_count: int,
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    if protected_message_count > 0:
        return messages[:protected_message_count], messages[protected_message_count:]
    index = 0
    while index < len(messages) and messages[index].role == "system":
        index += 1
    return messages[:index], messages[index:]


def _is_compaction_summary(message: ModelMessage) -> bool:
    return (
        message.role == "system"
        and message.content.startswith(COMPACTION_SUMMARY_PREFIX)
    )


def _bounded_transcript(
    messages: list[ModelMessage],
    *,
    secrets: dict[str, str],
    max_chars: int,
) -> str:
    selected: list[tuple[int, str]] = []
    used = 0

    prioritized = [
        (index, message)
        for index, message in enumerate(messages)
        if _is_compaction_summary(message)
    ]
    recent = [
        (index, message)
        for index, message in reversed(list(enumerate(messages)))
        if not _is_compaction_summary(message)
    ]
    used = _append_bounded_lines(
        selected,
        prioritized,
        secrets=secrets,
        max_chars=max(1, max_chars // 2),
        used=used,
    )
    _append_bounded_lines(
        selected,
        recent,
        secrets=secrets,
        max_chars=max_chars,
        used=used,
    )
    selected.sort(key=lambda item: item[0])
    return "\n".join(line for _, line in selected)


def _append_bounded_lines(
    selected: list[tuple[int, str]],
    messages: list[tuple[int, ModelMessage]],
    *,
    secrets: dict[str, str],
    max_chars: int,
    used: int,
) -> int:
    for index, message in messages:
        line = f"{message.role}: {_single_line(redact(message.content, secrets))}"
        separator_chars = 1 if selected else 0
        available = max_chars - used - separator_chars
        if available <= 0:
            break
        if len(line) > available:
            if available <= 3:
                break
            line = "..." + line[-(available - 3) :]
        selected.append((index, line))
        used += separator_chars + len(line)
    return used


def _extract_pinned_messages(
    messages: list[ModelMessage],
    pinned_messages: list[ModelMessage],
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    remaining = list(messages)
    pinned: list[ModelMessage] = []
    for pinned_message in pinned_messages:
        for index in range(len(remaining) - 1, -1, -1):
            candidate = remaining[index]
            if (
                candidate.role == pinned_message.role
                and candidate.content == pinned_message.content
            ):
                pinned.insert(0, candidate)
                del remaining[index]
                break
    return pinned, remaining


def _fit_to_budget(
    *,
    leading: list[ModelMessage],
    summary: ModelMessage,
    pinned: list[ModelMessage],
    tail: list[ModelMessage],
    max_chars: int,
) -> tuple[list[ModelMessage], ModelMessage]:
    messages = [*leading, summary, *tail, *pinned]
    while tail and estimate_message_chars(messages) > max_chars:
        tail = tail[1:]
        messages = [*leading, summary, *tail, *pinned]
    if estimate_message_chars(messages) <= max_chars:
        return messages, summary
    remaining = max_chars - estimate_message_chars([*leading, *tail, *pinned])
    if remaining <= len(summary.role) + MIN_COMPACTION_SUMMARY_CHARS + 2:
        raise ContextCompactionError(
            "Required context exceeds max_context_chars after compaction"
        )
    content_budget = max(0, remaining - len(summary.role) - 2 - 3)
    if content_budget < MIN_COMPACTION_SUMMARY_CHARS:
        raise ContextCompactionError(
            "Required context exceeds max_context_chars after compaction"
        )
    clipped_summary = ModelMessage(
        role=summary.role,
        content=summary.content[:content_budget].rstrip() + "...",
    )
    return [*leading, clipped_summary, *tail, *pinned], clipped_summary


def _single_line(text: str) -> str:
    return " ".join(text.split())
