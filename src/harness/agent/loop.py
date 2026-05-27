from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass

from pydantic import BaseModel, Field

from harness.agent.compaction import (
    ContextCompactionRequest,
    ContextCompactionResult,
    ContextCompactor,
    NoopContextCompactor,
)
from harness.agent.lease import SessionLeaseProvider, StorageSessionLeaseProvider
from harness.agent.protocol import parse_model_action
from harness.memory import MemoryExchange, MemoryManager
from harness.models import ModelMessage, ModelProvider
from harness.observability.sink import EventSink
from harness.schemas import ToolCall, ToolResult, TurnRecord
from harness.storage.base import StorageBackend
from harness.tools.gateway import ToolExecutionGateway
from harness.tools.redaction import redact, redact_with_detected_secrets, sensitive_values


class AgentRunResult(BaseModel):
    session_id: str
    final: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    iterations: int
    memory_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PendingContextCompaction:
    result: ContextCompactionResult
    covers_through_turn_id: str


class AgentLoop:
    def __init__(
        self,
        *,
        model: ModelProvider,
        tools: ToolExecutionGateway,
        storage: StorageBackend,
        system_prompt: str | None = None,
        max_iterations: int = 8,
        tool_concurrency: int = 8,
        container_cleanup_delay_minutes: float = 5,
        lease_provider: SessionLeaseProvider | None = None,
        memory: MemoryManager | None = None,
        context_compactor: ContextCompactor | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.storage = storage
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_iterations = max_iterations
        if tool_concurrency < 1:
            raise ValueError("tool_concurrency must be at least 1")
        self.tool_concurrency = tool_concurrency
        self.container_cleanup_delay_minutes = container_cleanup_delay_minutes
        self.lease_provider = lease_provider or StorageSessionLeaseProvider(storage)
        self.events = EventSink(storage)
        self.memory = memory
        if self.memory is not None and self.memory.events is None:
            self.memory.events = self.events
        self.context_compactor = context_compactor or NoopContextCompactor()
        self._system_message = self._build_system_message()

    async def run(self, session_id: str, user_message: str) -> AgentRunResult:
        lease = await self.lease_provider.enter_session(session_id)
        run_task = asyncio.create_task(self._run_inner(session_id, user_message))
        lost_task = asyncio.create_task(lease.wait_lost())
        try:
            try:
                done, _pending = await asyncio.wait(
                    {run_task, lost_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if lost_task in done:
                    run_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await run_task
                    await lost_task
                lost_task.cancel()
                with suppress(asyncio.CancelledError):
                    await lost_task
                return await run_task
            finally:
                await self.tools.end_session(
                    session_id,
                    cleanup_delay_minutes=self.container_cleanup_delay_minutes,
                )
        finally:
            run_task.cancel()
            lost_task.cancel()
            await lease.release()

    async def _run_inner(self, session_id: str, user_message: str) -> AgentRunResult:
        rolling_summary_turn, existing_turns = _context_turns_from_storage(
            await self.storage.list_turns(session_id, limit=None)
        )
        messages = [ModelMessage(role="system", content=self._system_message)]
        current_user_message = ModelMessage(role="user", content=user_message)
        memory_ids: list[str] = []
        if self.memory is not None:
            memory_context = await self.memory.context_for(
                user_message,
                session_id=session_id,
            )
            if memory_context.content:
                memory_ids = memory_context.memory_ids
                messages.append(ModelMessage(role="system", content=memory_context.content))
        protected_message_count = len(messages)
        if rolling_summary_turn is not None and rolling_summary_turn.content:
            messages.append(
                ModelMessage(role=rolling_summary_turn.role, content=rolling_summary_turn.content)
            )
        messages.extend(
            ModelMessage(role=turn.role, content=turn.content) for turn in existing_turns
        )
        messages.append(current_user_message)
        user_turn = turn_from_message(
            session_id=session_id,
            role="user",
            content=redact(user_message),
            metadata={"memory_ids": memory_ids} if memory_ids else None,
        )
        await self.storage.save_turn(user_turn)
        context_boundary_turn_id = user_turn.id
        all_tool_results: list[ToolResult] = []
        redaction_secrets = sensitive_values({"user_message": user_message})

        for iteration in range(1, self.max_iterations + 1):
            await self.events.emit(
                "agent.iteration.started",
                session_id=session_id,
                iteration=iteration,
            )
            try:
                messages, pending_compaction = await self._compact_context(
                    session_id=session_id,
                    messages=messages,
                    iteration=iteration,
                    covers_through_turn_id=context_boundary_turn_id,
                    protected_message_count=protected_message_count,
                    pinned_messages=[current_user_message],
                    redaction_secrets=redaction_secrets,
                )
                response = await self.model.complete(messages)
                action = parse_model_action(response)
            except Exception as exc:
                await self.events.emit(
                    "agent.run.failed",
                    session_id=session_id,
                    iteration=iteration,
                    error=redact(str(exc), redaction_secrets),
                )
                raise

            if action.kind == "final":
                assistant_turn = turn_from_message(
                    session_id=session_id,
                    role="assistant",
                    content=redact(action.content),
                    metadata={"iteration": iteration},
                )
                await self.storage.save_turn(assistant_turn)
                context_boundary_turn_id = assistant_turn.id
                await self._persist_compaction_checkpoint(
                    session_id=session_id,
                    iteration=iteration,
                    pending=pending_compaction,
                    pinned_messages=[current_user_message],
                    redaction_secrets=redaction_secrets,
                )
                await self.events.emit(
                    "agent.run.finished",
                    session_id=session_id,
                    iterations=iteration,
                )
                await self._capture_memory(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=action.content,
                    tool_results=all_tool_results,
                    source_turn_id=user_turn.id,
                )
                return AgentRunResult(
                    session_id=session_id,
                    final=action.content,
                    tool_results=all_tool_results,
                    iterations=iteration,
                    memory_ids=memory_ids,
                )

            calls = [
                ToolCall(session_id=session_id, name=call.name, arguments=call.arguments)
                for call in action.tool_calls
            ]
            redaction_secrets.update(
                sensitive_values(
                    {"tool_calls": [call.model_dump(mode="json") for call in action.tool_calls]}
                )
            )
            raw_tool_call_content = json.dumps(
                {"tool_calls": [call.model_dump(mode="json") for call in action.tool_calls]},
                sort_keys=True,
            )
            assistant_turn = turn_from_message(
                session_id=session_id,
                role="assistant",
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                **call.model_dump(mode="json"),
                                "arguments": redact(
                                    call.arguments,
                                    sensitive_values(call.arguments),
                                ),
                            }
                            for call in action.tool_calls
                        ]
                    },
                    sort_keys=True,
                ),
                metadata={"iteration": iteration, "kind": "tool_calls"},
            )
            await self.storage.save_turn(assistant_turn)
            context_boundary_turn_id = assistant_turn.id
            await self._persist_compaction_checkpoint(
                session_id=session_id,
                iteration=iteration,
                pending=pending_compaction,
                pinned_messages=[current_user_message],
                redaction_secrets=redaction_secrets,
            )
            messages.append(ModelMessage(role="assistant", content=raw_tool_call_content))
            tool_results = await self.tools.execute_many(
                calls,
                max_concurrency=self.tool_concurrency,
            )
            all_tool_results.extend(tool_results)
            redaction_secrets.update(
                sensitive_values(
                    {"tool_results": [result.model_dump(mode="json") for result in tool_results]}
                )
            )
            raw_tool_result_content = json.dumps(
                [result.model_dump(mode="json") for result in tool_results],
                sort_keys=True,
            )
            tool_turn = turn_from_message(
                session_id=session_id,
                role="tool",
                content=json.dumps(
                    [
                        redact_with_detected_secrets(result.model_dump(mode="json"))
                        for result in tool_results
                    ],
                    sort_keys=True,
                ),
                metadata={"iteration": iteration},
            )
            await self.storage.save_turn(tool_turn)
            context_boundary_turn_id = tool_turn.id
            messages.append(ModelMessage(role="tool", content=raw_tool_result_content))

        final = "Agent stopped before producing a final answer."
        await self.storage.save_turn(
            turn_from_message(
                session_id=session_id,
                role="assistant",
                content=redact(final),
                metadata={"stopped": "max_iterations"},
            )
        )
        await self.events.emit(
            "agent.run.stopped",
            session_id=session_id,
            reason="max_iterations",
        )
        return AgentRunResult(
            session_id=session_id,
            final=final,
            tool_results=all_tool_results,
            iterations=self.max_iterations,
            memory_ids=memory_ids,
        )

    async def _compact_context(
        self,
        *,
        session_id: str,
        messages: list[ModelMessage],
        iteration: int,
        covers_through_turn_id: str,
        protected_message_count: int,
        pinned_messages: list[ModelMessage],
        redaction_secrets: dict[str, str],
    ) -> tuple[list[ModelMessage], PendingContextCompaction | None]:
        result = await self.context_compactor.compact(
            ContextCompactionRequest(
                session_id=session_id,
                messages=messages,
                iteration=iteration,
                protected_message_count=protected_message_count,
                pinned_messages=pinned_messages,
                redaction_secrets=redaction_secrets,
            )
        )
        _validate_compaction_boundaries(
            original_messages=messages,
            messages=result.messages,
            protected_messages=messages[:protected_message_count],
            pinned_messages=pinned_messages,
            compacted=result.compacted,
            require_pinned_suffix=result.compacted,
        )
        if not result.compacted:
            return result.messages, None
        compacted_messages = _redact_compaction_summary_messages(
            result.messages,
            redaction_secrets,
        )
        return compacted_messages, PendingContextCompaction(
            result=result,
            covers_through_turn_id=covers_through_turn_id,
        )

    async def _persist_compaction_checkpoint(
        self,
        *,
        session_id: str,
        iteration: int,
        pending: PendingContextCompaction | None,
        pinned_messages: list[ModelMessage],
        redaction_secrets: dict[str, str],
    ) -> None:
        if pending is None:
            return
        result = pending.result
        await self.storage.save_turn(
            turn_from_message(
                session_id=session_id,
                role="system",
                content=redact(
                    _checkpoint_summary_content(result.summary or "", pinned_messages),
                    redaction_secrets,
                ),
                metadata={
                    "kind": "context_compaction_summary",
                    "iteration": iteration,
                    "original_chars": result.original_chars,
                    "compacted_chars": result.compacted_chars,
                    "summarized_messages": result.summarized_messages,
                    "committed": True,
                    "covers_through_turn_id": pending.covers_through_turn_id,
                },
            )
        )
        await self.events.emit(
            "agent.context.compacted",
            session_id=session_id,
            iteration=iteration,
            original_chars=result.original_chars,
            compacted_chars=result.compacted_chars,
            summarized_messages=result.summarized_messages,
        )

    async def _capture_memory(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_results: list[ToolResult],
        source_turn_id: str,
    ) -> None:
        if self.memory is None:
            return
        await self.memory.capture(
            MemoryExchange(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                tool_outputs=[
                    result.model_dump(mode="json")
                    for result in tool_results
                    if result.status == "ok"
                ],
                metadata={"source_turn_id": source_turn_id},
            )
        )

    def _build_system_message(self) -> str:
        definitions = sorted(
            self.tools.registry.list_definitions(),
            key=lambda definition: definition.name,
        )
        if not definitions:
            return self.system_prompt
        catalog = [
            {
                "name": definition.name,
                "description": definition.description,
                "input_schema": definition.input_schema,
                "output_schema": definition.output_schema,
                "required_capabilities": definition.required_capabilities,
                "required_secrets": definition.required_secrets,
                "execution_mode": definition.execution_mode.value,
            }
            for definition in definitions
        ]
        return (
            f"{self.system_prompt}\n\n"
            "Available tools are listed below as JSON. Use only these tool names and "
            "provide arguments matching each input_schema. Capability grants and "
            "secret resolution are enforced by the harness; secret names are listed "
            "only so you know which tools require credentials.\n\n"
            f"{json.dumps({'tools': catalog}, sort_keys=True)}"
        )


def turn_from_message(
    *,
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
):
    from harness.schemas import TurnRecord

    return TurnRecord(session_id=session_id, role=role, content=content, metadata=metadata or {})


def _redact_compaction_summary_messages(
    messages: list[ModelMessage],
    redaction_secrets: dict[str, str],
) -> list[ModelMessage]:
    return [
        ModelMessage(role=message.role, content=redact(message.content, redaction_secrets))
        if message.role == "system"
        else message
        for message in messages
    ]


def _validate_compaction_boundaries(
    original_messages: list[ModelMessage],
    messages: list[ModelMessage],
    *,
    protected_messages: list[ModelMessage],
    pinned_messages: list[ModelMessage],
    compacted: bool,
    require_pinned_suffix: bool,
) -> None:
    if not compacted and messages != original_messages:
        raise ValueError("Context compactor must not change messages when compacted=false")
    if messages[: len(protected_messages)] != protected_messages:
        raise ValueError("Context compactor must preserve protected leading messages")
    if not pinned_messages:
        return
    if require_pinned_suffix and messages[-len(pinned_messages) :] != pinned_messages:
        raise ValueError("Context compactor must preserve pinned messages as the suffix")
    for pinned_message in pinned_messages:
        if not any(
            message.role == pinned_message.role and message.content == pinned_message.content
            for message in messages
        ):
            raise ValueError("Context compactor must preserve pinned messages")


def _checkpoint_summary_content(
    summary: str,
    pinned_messages: list[ModelMessage],
) -> str:
    if not pinned_messages:
        return summary
    pinned = "\n".join(
        f"{message.role}: {message.content}" for message in pinned_messages
    )
    return f"{summary}\n\nPinned messages at checkpoint:\n{pinned}"


def _context_turns_from_storage(
    turns: list[TurnRecord],
) -> tuple[TurnRecord | None, list[TurnRecord]]:
    normal_turns = [
        turn for turn in turns if turn.metadata.get("kind") != "context_compaction_summary"
    ]
    normal_turn_indexes = {turn.id: index for index, turn in enumerate(normal_turns)}
    ordered_turn_indexes = {turn.id: index for index, turn in enumerate(turns)}
    latest_summary: TurnRecord | None = None
    latest_covered_index: int | None = None
    for turn in turns:
        if turn.metadata.get("kind") != "context_compaction_summary":
            continue
        if turn.metadata.get("committed") is not True:
            continue
        covered_id = turn.metadata.get("covers_through_turn_id")
        if not isinstance(covered_id, str):
            continue
        covered_index = normal_turn_indexes.get(covered_id)
        if covered_index is None:
            continue
        if ordered_turn_indexes[covered_id] >= ordered_turn_indexes[turn.id]:
            continue
        latest_summary = turn
        latest_covered_index = covered_index
    if latest_summary is None or latest_covered_index is None:
        return None, normal_turns
    return latest_summary, normal_turns[latest_covered_index + 1 :]


DEFAULT_SYSTEM_PROMPT = """You are running inside an agent harness.

Respond with one of these JSON shapes:

{"final": "answer for the user"}

or

{"tool_calls": [{"name": "tool.name", "arguments": {"key": "value"}}]}

Tool calls in the same response may run concurrently. For tool use, return only
tool_calls and wait for results. Never include final with tool_calls or
simulate results.
"""
