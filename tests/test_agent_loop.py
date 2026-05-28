from __future__ import annotations

import json

import anyio
import pytest

from harness.agent import (
    AgentLoop,
    AgentSessionManager,
    ContextCompactionError,
    ContextCompactionPolicy,
    ContextCompactionRequest,
    ContextCompactionResult,
    ContextCompactor,
    RollingSummaryContextCompactor,
    SessionLeaseError,
    StorageSessionLeaseProvider,
)
from harness.memory import (
    HashEmbeddingProvider,
    MemoryCandidate,
    MemoryExchange,
    MemoryExtractor,
    MemoryManager,
    MemoryPolicy,
    MemoryStore,
)
from harness.models import ModelMessage, ModelProvider, ModelResponse
from harness.schemas import ExecutionMode, MemoryScope, ToolDefinition, TurnRecord
from harness.storage import SQLiteStorage
from harness.tools import CapabilityGrant, CapabilityPolicy, ToolExecutionGateway, ToolRegistry

pytestmark = pytest.mark.anyio
COMPACTION_PROMPT_MARKER = "Create a checkpoint summary"


class ScriptedModel(ModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.messages_seen: list[list[ModelMessage]] = []

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        self.messages_seen.append(messages)
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


class RoutingModel(ModelProvider):
    def __init__(self) -> None:
        self.messages_seen: list[list[ModelMessage]] = []

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        self.messages_seen.append(messages)
        if messages and COMPACTION_PROMPT_MARKER in messages[0].content:
            return ModelResponse(content="model summary preserved old context")
        return ModelResponse(content='{"final": "done"}')


class EchoingSummaryModel(ModelProvider):
    def __init__(self) -> None:
        self.messages_seen: list[list[ModelMessage]] = []

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        self.messages_seen.append(messages)
        if messages and COMPACTION_PROMPT_MARKER in messages[0].content:
            return ModelResponse(content=messages[1].content)
        return ModelResponse(content='{"final": "done"}')


class StaticMemoryExtractor(MemoryExtractor):
    async def extract(self, exchange: MemoryExchange) -> list[MemoryCandidate]:
        return [
            MemoryCandidate(
                text=f"remembered {exchange.user_message}",
                scope=MemoryScope.SESSION,
            )
        ]


class SecretRepeatingCompactionModel(ModelProvider):
    def __init__(self) -> None:
        self.messages_seen: list[list[ModelMessage]] = []
        self.primary_calls = 0

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        self.messages_seen.append(messages)
        if messages and COMPACTION_PROMPT_MARKER in messages[0].content:
            return ModelResponse(content="summary repeats plain-shared-secret")
        self.primary_calls += 1
        if self.primary_calls == 1:
            return ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "echo",
                                "arguments": {
                                    "clientSecret": "plain-shared-secret",
                                    "value": "plain-shared-secret",
                                },
                            }
                        ]
                    }
                )
            )
        return ModelResponse(content='{"final": "done"}')


class FailingCompactionModel(ModelProvider):
    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        if messages and COMPACTION_PROMPT_MARKER in messages[0].content:
            raise RuntimeError("api_key=sk-compaction-secret")
        return ModelResponse(content='{"final": "done"}')


class FailingAfterToolModel(ModelProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        _ = messages
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "echo",
                                "arguments": {
                                    "clientSecret": "plain-shared-secret",
                                    "value": "ok",
                                },
                            }
                        ]
                    }
                )
            )
        raise RuntimeError("provider echoed plain-shared-secret")


class FailingModel(ModelProvider):
    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        _ = messages
        raise RuntimeError("api_key=sk-live-secret")


class BlockingModel(ModelProvider):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        _ = messages
        self.started.set()
        await self.release.wait()
        return ModelResponse(content='{"final": "done"}')


class LeakyCustomCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        original_chars = sum(len(message.content) for message in request.messages)
        if request.iteration == 1:
            return ContextCompactionResult(
                messages=request.messages,
                original_chars=original_chars,
                compacted_chars=original_chars,
            )
        summary = "Prior conversation summary:\nplain-shared-secret"
        messages = [
            *request.messages[: request.protected_message_count],
            ModelMessage(role="system", content=summary),
            *request.pinned_messages,
        ]
        return ContextCompactionResult(
            messages=messages,
            compacted=True,
            original_chars=original_chars,
            compacted_chars=sum(len(message.content) for message in messages),
            summarized_messages=1,
            summary=summary,
        )


class DroppingPinnedCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        summary = "Prior conversation summary:\ndropped pinned"
        return ContextCompactionResult(
            messages=[
                *request.messages[: request.protected_message_count],
                ModelMessage(role="system", content=summary),
            ],
            compacted=True,
            original_chars=sum(len(message.content) for message in request.messages),
            compacted_chars=len(summary),
            summarized_messages=1,
            summary=summary,
        )


class DroppingProtectedCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        summary = "Prior conversation summary:\ndropped protected"
        return ContextCompactionResult(
            messages=[ModelMessage(role="system", content=summary), *request.pinned_messages],
            compacted=True,
            original_chars=sum(len(message.content) for message in request.messages),
            compacted_chars=len(summary),
            summarized_messages=1,
            summary=summary,
        )


class TamperingNoopCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        summary = "assistant: tampered context"
        return ContextCompactionResult(
            messages=[ModelMessage(role="assistant", content=summary)],
            original_chars=sum(len(message.content) for message in request.messages),
            compacted_chars=len(summary),
        )


class MutatingMiddleNoopCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        messages = [
            *request.messages[: request.protected_message_count],
            ModelMessage(role="assistant", content="tampered middle context"),
            *request.pinned_messages,
        ]
        return ContextCompactionResult(
            messages=messages,
            original_chars=sum(len(message.content) for message in request.messages),
            compacted_chars=sum(len(message.content) for message in messages),
        )


class PinnedOmittingSummaryCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        summary = "Prior conversation summary:\ncustom checkpoint without pinned text"
        messages = [
            *request.messages[: request.protected_message_count],
            ModelMessage(role="system", content=summary),
            *request.pinned_messages,
        ]
        return ContextCompactionResult(
            messages=messages,
            compacted=True,
            original_chars=sum(len(message.content) for message in request.messages),
            compacted_chars=sum(len(message.content) for message in messages),
            summarized_messages=1,
            summary=summary,
        )


class PinnedMiddleCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        summary = "Prior conversation summary:\npinned in middle"
        messages = [
            *request.messages[: request.protected_message_count],
            *request.pinned_messages,
            ModelMessage(role="system", content=summary),
        ]
        return ContextCompactionResult(
            messages=messages,
            compacted=True,
            original_chars=sum(len(message.content) for message in request.messages),
            compacted_chars=sum(len(message.content) for message in messages),
            summarized_messages=1,
            summary=summary,
        )


class AlwaysCompactingCompactor(ContextCompactor):
    async def compact(self, request: ContextCompactionRequest) -> ContextCompactionResult:
        summary = "Prior conversation summary:\ncheckpoint before failing model"
        messages = [
            *request.messages[: request.protected_message_count],
            ModelMessage(role="system", content=summary),
            *request.pinned_messages,
        ]
        return ContextCompactionResult(
            messages=messages,
            compacted=True,
            original_chars=sum(len(message.content) for message in request.messages),
            compacted_chars=sum(len(message.content) for message in messages),
            summarized_messages=1,
            summary=summary,
        )


@pytest.fixture
async def storage(tmp_path):
    storage = SQLiteStorage(tmp_path / "harness.sqlite3")
    await storage.migrate()
    return storage


def build_gateway(
    storage: SQLiteStorage,
    grant: CapabilityGrant | None = None,
) -> ToolExecutionGateway:
    registry = ToolRegistry()

    async def add(arguments: dict, _secrets: dict[str, str]) -> dict:
        return {"value": arguments["left"] + arguments["right"]}

    async def echo(arguments: dict, _secrets: dict[str, str]) -> dict:
        return {"value": arguments["value"]}

    registry.register(
        ToolDefinition(
            name="math.add",
            description="Add two numbers",
            required_capabilities=["math:add"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        add,
    )
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo value",
            required_capabilities=["echo"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo,
    )
    return ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(grant or CapabilityGrant.all()),
        storage=storage,
    )


def turn_from_message_for_test(session_id: str, role: str, content: str) -> TurnRecord:
    return TurnRecord(session_id=session_id, role=role, content=content)


async def test_agent_loop_returns_final_answer(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    result = await loop.run(session.id, "hello")

    assert result.final == "done"
    turns = await storage.list_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]


async def test_agent_loop_executes_tool_then_finishes(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {"name": "math.add", "arguments": {"left": 2, "right": 3}}
                        ]
                    }
                )
            ),
            ModelResponse(content='{"final": "5"}'),
        ]
    )
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    result = await loop.run(session.id, "add")

    assert result.final == "5"
    assert result.tool_results[0].output == {"value": 5}
    turns = await storage.list_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant", "tool", "assistant"]


async def test_agent_loop_surfaces_tool_catalog_to_model(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    await loop.run(session.id, "what can you do?")

    system = model.messages_seen[0][0].content
    assert "Available tools" in system
    assert '"name": "math.add"' in system
    assert '"required_capabilities": ["math:add"]' in system
    assert '"execution_mode": "in_process"' in system


async def test_agent_loop_injects_memory_context_when_configured(storage):
    session = await AgentSessionManager(storage).create()
    memory = MemoryManager(
        MemoryStore(storage, HashEmbeddingProvider()),
        MemoryPolicy(namespace="project", scopes=[MemoryScope.AGENT]),
    )
    await memory.remember("docker is the approved sandbox backend", namespace="project")
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        memory=memory,
    )

    result = await loop.run(session.id, "which sandbox should I use?")

    assert result.memory_ids
    assert any(
        message.role == "system" and "docker is the approved sandbox backend" in message.content
        for message in model.messages_seen[0]
    )
    turns = await storage.list_turns(session.id)
    assert turns[0].metadata["memory_ids"] == result.memory_ids


async def test_agent_loop_compacts_context_when_over_budget(storage):
    session = await AgentSessionManager(storage).create()
    old_turn = "old context " * 20
    recent_turn = "recent context"
    await storage.save_turn(
        turn_from_message_for_test(session.id, "user", old_turn),
    )
    await storage.save_turn(
        turn_from_message_for_test(session.id, "assistant", "older assistant " * 20),
    )
    await storage.save_turn(
        turn_from_message_for_test(session.id, "user", recent_turn),
    )
    model = RoutingModel()
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=RollingSummaryContextCompactor(
            model,
            ContextCompactionPolicy(
                max_context_chars=1_500,
                trigger_ratio=1.0,
                preserve_recent_messages=2,
                summary_max_chars=300,
            )
        ),
    )

    result = await loop.run(session.id, "latest request")

    assert result.final == "done"
    summary_call = model.messages_seen[0]
    assert COMPACTION_PROMPT_MARKER in summary_call[0].content
    assert " ".join(old_turn.split()) in summary_call[1].content
    seen = model.messages_seen[1]
    assert "Available tools" in seen[0].content
    assert any("model summary preserved old context" in message.content for message in seen)
    assert seen[-1].content == "latest request"
    assert not any(message.content == old_turn for message in seen)
    events = await storage.list_events(session_id=session.id, limit=20)
    assert any(event.event_type == "agent.context.compacted" for event in events)
    turns = await storage.list_turns(session.id)
    summaries = [
        turn for turn in turns if turn.metadata.get("kind") == "context_compaction_summary"
    ]
    assert summaries
    assert summaries[0].metadata["summarized_messages"] >= 1


async def test_agent_loop_compaction_redacts_detected_tool_secrets(storage):
    session = await AgentSessionManager(storage).create()
    model = SecretRepeatingCompactionModel()
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        max_iterations=2,
        context_compactor=RollingSummaryContextCompactor(
            model,
            ContextCompactionPolicy(
                max_context_chars=1_500,
                trigger_ratio=0.5,
                preserve_recent_messages=1,
                summary_max_chars=300,
            ),
        ),
    )

    await loop.run(session.id, "use clientSecret plain-shared-secret")

    summary_index = next(
        index
        for index, seen in enumerate(model.messages_seen)
        if seen and COMPACTION_PROMPT_MARKER in seen[0].content
    )
    summary_call = model.messages_seen[summary_index]
    compacted_primary_call = model.messages_seen[summary_index + 1]
    transcript = "\n".join(turn.content for turn in await storage.list_turns(session.id))
    assert "plain-shared-secret" not in str([message.content for message in summary_call])
    assert "plain-shared-secret" not in str(
        [
            message.content
            for message in compacted_primary_call
            if message.content.startswith("Prior conversation summary:")
        ]
    )
    assert "plain-shared-secret" not in transcript
    assert "[REDACTED]" in str([message.content for message in summary_call])


async def test_agent_loop_compaction_pins_current_user_request(storage):
    session = await AgentSessionManager(storage).create()
    for index in range(8):
        await storage.save_turn(
            turn_from_message_for_test(
                session.id,
                "assistant" if index % 2 else "user",
                f"older message {index} " * 20,
            )
        )
    model = RoutingModel()
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=RollingSummaryContextCompactor(
            model,
            ContextCompactionPolicy(
                max_context_chars=1_200,
                trigger_ratio=1.0,
                preserve_recent_messages=1,
                summary_max_chars=120,
            ),
        ),
    )

    await loop.run(session.id, "must preserve exact acceptance criteria")

    compacted_call = model.messages_seen[1]
    assert compacted_call[-1].content == "must preserve exact acceptance criteria"


async def test_context_compaction_trims_tail_to_budget(storage):
    session = await AgentSessionManager(storage).create()
    model = RoutingModel()
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=RollingSummaryContextCompactor(
            model,
            ContextCompactionPolicy(
                max_context_chars=1_200,
                trigger_ratio=1.0,
                preserve_recent_messages=6,
                summary_max_chars=80,
            ),
        ),
    )
    for index in range(10):
        await storage.save_turn(
            turn_from_message_for_test(session.id, "user", f"turn {index} " + ("x" * 120))
        )

    await loop.run(session.id, "latest request")

    compacted_call = model.messages_seen[1]
    assert sum(len(message.role) + len(message.content) + 2 for message in compacted_call) <= 1_200
    assert compacted_call[-1].content == "latest request"


async def test_context_compaction_clipped_summary_matches_result_contract():
    messages = [ModelMessage(role="system", content="system")]
    messages.extend(
        ModelMessage(role="user", content=f"old turn {index} " + ("x" * 120))
        for index in range(8)
    )
    model = ScriptedModel([ModelResponse(content="summary " + ("y" * 500))])
    compactor = RollingSummaryContextCompactor(
        model,
        ContextCompactionPolicy(
            max_context_chars=260,
            trigger_ratio=1.0,
            preserve_recent_messages=1,
            summary_max_chars=800,
        ),
    )

    result = await compactor.compact(
        ContextCompactionRequest(
            session_id="session",
            messages=messages,
            iteration=1,
            protected_message_count=1,
        )
    )

    assert result.compacted
    assert result.summary is not None
    assert result.summary.endswith("...")
    assert sum(message.content == result.summary for message in result.messages) == 1
    assert sum(len(message.role) + len(message.content) + 2 for message in result.messages) <= 260


async def test_context_compaction_raises_domain_error_when_prefix_cannot_fit():
    messages = [
        ModelMessage(role="system", content="system"),
        ModelMessage(role="user", content="old turn " + ("x" * 120)),
        ModelMessage(role="assistant", content="old answer " + ("y" * 120)),
        ModelMessage(role="user", content="latest"),
    ]
    model = ScriptedModel([ModelResponse(content="summary " + ("z" * 120))])
    compactor = RollingSummaryContextCompactor(
        model,
        ContextCompactionPolicy(
            max_context_chars=45,
            trigger_ratio=1.0,
            preserve_recent_messages=1,
            summary_max_chars=120,
        ),
    )

    with pytest.raises(ContextCompactionError, match="Required context"):
        await compactor.compact(
            ContextCompactionRequest(
                session_id="session",
                messages=messages,
                iteration=1,
                protected_message_count=1,
            )
        )


async def test_context_compaction_bounds_summarizer_input():
    messages = [ModelMessage(role="system", content="system")]
    messages.extend(
        ModelMessage(role="user", content=f"old turn {index} " + ("x" * 120))
        for index in range(20)
    )
    model = RoutingModel()
    compactor = RollingSummaryContextCompactor(
        model,
        ContextCompactionPolicy(
            max_context_chars=500,
            trigger_ratio=1.0,
            preserve_recent_messages=1,
            summarizer_input_max_chars=300,
            summary_max_chars=120,
        ),
    )

    await compactor.compact(
        ContextCompactionRequest(
            session_id="session",
            messages=messages,
            iteration=1,
            protected_message_count=1,
        )
    )

    summary_call = model.messages_seen[0]
    transcript = summary_call[1].content.split("\n\n", 1)[1]
    assert len(transcript) <= 300
    assert "old turn 18" in transcript
    assert "old turn 0" not in transcript


async def test_context_compaction_resummarizes_prior_compaction_summary():
    messages = [
        ModelMessage(role="system", content="primary system prompt"),
        ModelMessage(role="system", content="Prior conversation summary:\nold summary"),
        ModelMessage(role="user", content="old request " * 80),
        ModelMessage(role="assistant", content="old answer " * 80),
        ModelMessage(role="user", content="latest request"),
    ]
    model = RoutingModel()
    compactor = RollingSummaryContextCompactor(
        model,
        ContextCompactionPolicy(
            max_context_chars=500,
            trigger_ratio=1.0,
            preserve_recent_messages=1,
            summarizer_input_max_chars=160,
            summary_max_chars=120,
        ),
    )

    result = await compactor.compact(
        ContextCompactionRequest(
            session_id="session",
            messages=messages,
            iteration=1,
            protected_message_count=1,
        )
    )

    assert result.compacted
    summary_prompt = model.messages_seen[0][1].content
    assert len(summary_prompt.split("\n\n", 1)[1]) <= 160
    assert "Prior conversation summary" in summary_prompt
    assert "old summary" in summary_prompt
    assert result.messages[0].content == "primary system prompt"
    assert (
        sum(
            message.content.startswith("Prior conversation summary:")
            for message in result.messages
        )
        == 1
    )


async def test_context_compaction_preserves_protected_system_prompt_with_summary_prefix():
    prompt = "Prior conversation summary:\nthis is a real system instruction"
    messages = [
        ModelMessage(role="system", content=prompt),
        ModelMessage(role="user", content="old request " * 40),
        ModelMessage(role="assistant", content="old answer " * 40),
        ModelMessage(role="user", content="latest request"),
    ]
    model = RoutingModel()
    compactor = RollingSummaryContextCompactor(
        model,
        ContextCompactionPolicy(
            max_context_chars=500,
            trigger_ratio=1.0,
            preserve_recent_messages=1,
            summary_max_chars=120,
        ),
    )

    result = await compactor.compact(
        ContextCompactionRequest(
            session_id="session",
            messages=messages,
            iteration=1,
            protected_message_count=1,
        )
    )

    assert result.compacted
    assert result.messages[0].content == prompt
    assert model.messages_seen[0][1].content.count("Prior conversation summary") == 0


def test_context_compaction_result_requires_summary_when_compacted():
    with pytest.raises(ValueError, match="summary is required"):
        ContextCompactionResult(messages=[], compacted=True)


def test_context_compaction_result_requires_matching_system_summary_message():
    with pytest.raises(ValueError, match="must start"):
        ContextCompactionResult(
            messages=[ModelMessage(role="system", content="summary")],
            compacted=True,
            summary="summary",
        )
    with pytest.raises(ValueError, match="exactly one"):
        ContextCompactionResult(
            messages=[ModelMessage(role="assistant", content="Prior conversation summary:\ntext")],
            compacted=True,
            summary="Prior conversation summary:\ntext",
        )


async def test_agent_loop_replays_summary_checkpoint_without_losing_current_turn(storage):
    session = await AgentSessionManager(storage).create()
    await storage.save_turn(
        turn_from_message_for_test(session.id, "user", "recent tail before compaction")
    )
    model = EchoingSummaryModel()
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=RollingSummaryContextCompactor(
            model,
            ContextCompactionPolicy(
                max_context_chars=1_000,
                trigger_ratio=1.0,
                preserve_recent_messages=1,
                summary_max_chars=800,
            ),
        ),
    )

    await loop.run(session.id, "current request that triggered compaction")
    follow_up_model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    follow_up_loop = AgentLoop(
        model=follow_up_model,
        tools=build_gateway(storage),
        storage=storage,
    )

    await follow_up_loop.run(session.id, "follow up")

    seen = "\n".join(message.content for message in follow_up_model.messages_seen[0])
    assert "recent tail before compaction" in seen
    assert "current request that triggered compaction" in seen


async def test_agent_loop_replays_latest_persisted_compaction_summary(storage):
    session = await AgentSessionManager(storage).create()
    raw_pre_summary = TurnRecord(
        session_id=session.id,
        role="user",
        content="raw pre-summary context",
    )
    await storage.save_turn(raw_pre_summary)
    await storage.save_turn(
        TurnRecord(
            session_id=session.id,
            role="system",
            content="Prior conversation summary:\ninternal prior summary",
            metadata={
                "kind": "context_compaction_summary",
                "committed": True,
                "covers_through_turn_id": raw_pre_summary.id,
            },
        )
    )
    await storage.save_turn(
        TurnRecord(
            session_id=session.id,
            role="assistant",
            content="raw post-summary context",
        )
    )
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    await loop.run(session.id, "hello")

    seen = "\n".join(message.content for message in model.messages_seen[0])
    assert "internal prior summary" in seen
    assert "raw post-summary context" in seen
    assert "raw pre-summary context" not in seen


async def test_agent_loop_ignores_non_boolean_committed_checkpoint(storage):
    session = await AgentSessionManager(storage).create()
    raw_pre_summary = TurnRecord(
        session_id=session.id,
        role="user",
        content="raw pre-summary context",
    )
    await storage.save_turn(raw_pre_summary)
    await storage.save_turn(
        TurnRecord(
            session_id=session.id,
            role="system",
            content="Prior conversation summary:\nstring committed summary",
            metadata={
                "kind": "context_compaction_summary",
                "committed": "true",
                "covers_through_turn_id": raw_pre_summary.id,
            },
        )
    )
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    await loop.run(session.id, "hello")

    seen = "\n".join(message.content for message in model.messages_seen[0])
    assert "string committed summary" not in seen
    assert "raw pre-summary context" in seen


async def test_agent_loop_ignores_checkpoint_covering_future_turn(storage):
    session = await AgentSessionManager(storage).create()
    raw_pre_summary = TurnRecord(
        session_id=session.id,
        role="user",
        content="raw pre-summary context",
    )
    await storage.save_turn(raw_pre_summary)
    future_turn = TurnRecord(
        session_id=session.id,
        role="assistant",
        content="future raw turn",
    )
    await storage.save_turn(
        TurnRecord(
            session_id=session.id,
            role="system",
            content="Prior conversation summary:\nmalformed future checkpoint",
            metadata={
                "kind": "context_compaction_summary",
                "committed": True,
                "covers_through_turn_id": future_turn.id,
            },
        )
    )
    await storage.save_turn(future_turn)
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    await loop.run(session.id, "hello")

    seen = "\n".join(message.content for message in model.messages_seen[0])
    assert "malformed future checkpoint" not in seen
    assert "raw pre-summary context" in seen
    assert "future raw turn" in seen


async def test_agent_loop_records_redacted_compaction_failure(storage):
    session = await AgentSessionManager(storage).create()
    await storage.save_turn(
        turn_from_message_for_test(session.id, "user", "old context " * 20)
    )
    await storage.save_turn(
        turn_from_message_for_test(session.id, "assistant", "older answer " * 20)
    )
    model = FailingCompactionModel()
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=RollingSummaryContextCompactor(
            model,
            ContextCompactionPolicy(
                max_context_chars=1_000,
                trigger_ratio=1.0,
                preserve_recent_messages=1,
                summary_max_chars=120,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="api_key"):
        await loop.run(session.id, "hello")

    events = await storage.list_events(session_id=session.id, limit=20)
    failed = next(event for event in events if event.event_type == "agent.run.failed")
    assert "sk-compaction-secret" not in str(failed.payload)
    summaries = [
        turn
        for turn in await storage.list_turns(session.id)
        if turn.metadata.get("kind") == "context_compaction_summary"
    ]
    assert summaries == []


async def test_agent_loop_records_collected_secret_redacted_failure_event(storage):
    session = await AgentSessionManager(storage).create()
    loop = AgentLoop(
        model=FailingAfterToolModel(),
        tools=build_gateway(storage),
        storage=storage,
        max_iterations=2,
    )

    with pytest.raises(RuntimeError, match="plain-shared-secret"):
        await loop.run(session.id, "call a tool then fail")

    events = await storage.list_events(session_id=session.id, limit=20)
    failed = next(event for event in events if event.event_type == "agent.run.failed")
    assert "plain-shared-secret" not in str(failed.payload)
    assert "[REDACTED]" in str(failed.payload)


async def test_agent_loop_redacts_custom_compaction_summary_with_collected_secrets(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "echo",
                                "arguments": {
                                    "clientSecret": "plain-shared-secret",
                                    "value": "ok",
                                },
                            }
                        ]
                    }
                )
            ),
            ModelResponse(content='{"final": "done"}'),
        ]
    )
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        max_iterations=2,
        context_compactor=LeakyCustomCompactor(),
    )

    await loop.run(session.id, "call a tool then compact")

    transcript = "\n".join(turn.content for turn in await storage.list_turns(session.id))
    summary_messages = [
        message.content
        for message in model.messages_seen[1]
        if message.content.startswith("Prior conversation summary:")
    ]
    assert "plain-shared-secret" not in transcript
    assert "plain-shared-secret" not in "\n".join(summary_messages)
    assert "[REDACTED]" in transcript
    assert "[REDACTED]" in "\n".join(summary_messages)


async def test_agent_loop_rejects_compactor_that_drops_pinned_user_message(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=DroppingPinnedCompactor(),
    )

    with pytest.raises(ValueError, match="preserve pinned"):
        await loop.run(session.id, "must stay pinned")


async def test_agent_loop_rejects_compactor_that_drops_protected_messages(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=DroppingProtectedCompactor(),
    )

    with pytest.raises(ValueError, match="preserve protected"):
        await loop.run(session.id, "must keep system prompt")


async def test_agent_loop_rejects_tampered_uncompacted_messages(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=TamperingNoopCompactor(),
    )

    with pytest.raises(ValueError, match="compacted=false"):
        await loop.run(session.id, "must not be tampered")


async def test_agent_loop_rejects_mutated_uncompacted_middle_messages(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=MutatingMiddleNoopCompactor(),
    )

    with pytest.raises(ValueError, match="compacted=false"):
        await loop.run(session.id, "must not mutate middle")


async def test_agent_loop_rejects_compactor_with_pinned_message_not_suffix(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=PinnedMiddleCompactor(),
    )

    with pytest.raises(ValueError, match="preserve pinned messages as the suffix"):
        await loop.run(session.id, "must stay last")


async def test_agent_loop_persists_pinned_messages_with_custom_summary(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=PinnedOmittingSummaryCompactor(),
    )

    await loop.run(session.id, "pinned request omitted by custom summary")
    follow_up_model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    follow_up_loop = AgentLoop(
        model=follow_up_model,
        tools=build_gateway(storage),
        storage=storage,
    )

    await follow_up_loop.run(session.id, "follow up")

    seen = "\n".join(message.content for message in follow_up_model.messages_seen[0])
    assert "custom checkpoint without pinned text" in seen
    assert "pinned request omitted by custom summary" in seen


async def test_agent_loop_ignores_trailing_checkpoint_after_model_failure(storage):
    session = await AgentSessionManager(storage).create()
    failing_model = FailingModel()
    loop = AgentLoop(
        model=failing_model,
        tools=build_gateway(storage),
        storage=storage,
        context_compactor=AlwaysCompactingCompactor(),
    )

    with pytest.raises(RuntimeError, match="api_key"):
        await loop.run(session.id, "request before failing model")

    follow_up_model = ScriptedModel([ModelResponse(content='{"final": "done"}')])
    follow_up_loop = AgentLoop(
        model=follow_up_model,
        tools=build_gateway(storage),
        storage=storage,
    )

    await follow_up_loop.run(session.id, "follow up")

    seen = "\n".join(message.content for message in follow_up_model.messages_seen[0])
    assert "request before failing model" in seen
    assert "checkpoint before failing model" not in seen


async def test_agent_loop_preserves_denied_tool_result(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {"name": "math.add", "arguments": {"left": 1, "right": 1}}
                        ]
                    }
                )
            ),
            ModelResponse(content='{"final": "denied"}'),
        ]
    )
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage, CapabilityGrant(frozenset())),
        storage=storage,
    )

    result = await loop.run(session.id, "add")

    assert result.tool_results[0].status == "denied"
    assert result.final == "denied"


async def test_agent_loop_runs_tool_batch(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {"name": "echo", "arguments": {"value": "a"}},
                            {"name": "echo", "arguments": {"value": "b"}},
                        ]
                    }
                )
            ),
            ModelResponse(content='{"final": "done"}'),
        ]
    )
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    result = await loop.run(session.id, "echo both")

    assert [tool_result.output["value"] for tool_result in result.tool_results] == ["a", "b"]


async def test_agent_loop_redacts_tool_call_arguments_in_transcript(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "echo",
                                "arguments": {
                                    "api_key": "sk-live-secret",
                                    "authorization": "Bearer secret-token-value",
                                    "value": "ok",
                                },
                            }
                        ]
                    }
                )
            ),
            ModelResponse(content='{"final": "done"}'),
        ]
    )
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    await loop.run(session.id, "use a secret-looking argument")

    transcript = "\n".join(turn.content for turn in await storage.list_turns(session.id))
    assert "sk-live-secret" not in transcript
    assert "secret-token-value" not in transcript
    assert '"api_key": "[REDACTED]"' in transcript
    assert '"authorization": "[REDACTED]"' in transcript


async def test_agent_loop_redacts_copied_secret_values_at_rest(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "echo",
                                "arguments": {
                                    "clientSecret": "plain-shared-secret",
                                    "value": "plain-shared-secret",
                                },
                            }
                        ]
                    }
                )
            ),
            ModelResponse(content='{"final": "done"}'),
        ]
    )
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    await loop.run(session.id, "use a copied secret")

    transcript = "\n".join(turn.content for turn in await storage.list_turns(session.id))
    assert "plain-shared-secret" not in transcript


async def test_agent_loop_keeps_raw_active_transcript_for_model(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {"tool_calls": [{"name": "echo", "arguments": {"value": "sk-not-a-key"}}]}
                )
            ),
            ModelResponse(content='{"final": "done"}'),
        ]
    )
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    await loop.run(session.id, "literal sk-not-a-key")

    second_iteration = model.messages_seen[1]
    assert any(message.content == "literal sk-not-a-key" for message in second_iteration)
    assert any("sk-not-a-key" in message.content for message in second_iteration)
    stored = "\n".join(turn.content for turn in await storage.list_turns(session.id))
    assert "sk-not-a-key" not in stored


async def test_agent_loop_redacts_user_and_final_turns_at_rest(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel([ModelResponse(content='{"final": "saw api_key sk-final-secret"}')])
    loop = AgentLoop(model=model, tools=build_gateway(storage), storage=storage)

    result = await loop.run(session.id, "my api_key is sk-user-secret")

    assert result.final == "saw api_key sk-final-secret"
    transcript = "\n".join(turn.content for turn in await storage.list_turns(session.id))
    assert "sk-user-secret" not in transcript
    assert "sk-final-secret" not in transcript
    assert transcript.count("[REDACTED]") >= 2


async def test_agent_loop_stops_after_max_iterations_and_records_event(storage):
    session = await AgentSessionManager(storage).create()
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {"tool_calls": [{"name": "echo", "arguments": {"value": "again"}}]}
                )
            ),
        ]
    )
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        max_iterations=1,
    )

    result = await loop.run(session.id, "loop")

    assert result.final == "Agent stopped before producing a final answer."
    events = await storage.list_events(session_id=session.id, limit=20)
    assert any(event.event_type == "agent.run.stopped" for event in events)


async def test_agent_loop_captures_memory_after_max_iterations(storage):
    session = await AgentSessionManager(storage).create()
    memory = MemoryManager(
        MemoryStore(storage=storage, embeddings=HashEmbeddingProvider()),
        policy=MemoryPolicy(namespace="project", auto_capture=True),
        extractor=StaticMemoryExtractor(),
    )
    model = ScriptedModel(
        [
            ModelResponse(
                content=json.dumps(
                    {"tool_calls": [{"name": "echo", "arguments": {"value": "again"}}]}
                )
            ),
        ]
    )
    loop = AgentLoop(
        model=model,
        tools=build_gateway(storage),
        storage=storage,
        max_iterations=1,
        memory=memory,
    )

    await loop.run(session.id, "loop")

    memories = await storage.list_memories("project", scopes=[MemoryScope.SESSION])
    assert [item.text for item in memories] == ["remembered loop"]


async def test_agent_loop_records_redacted_model_failure_event(storage):
    session = await AgentSessionManager(storage).create()
    loop = AgentLoop(model=FailingModel(), tools=build_gateway(storage), storage=storage)

    with pytest.raises(RuntimeError, match="api_key"):
        await loop.run(session.id, "fail")

    events = await storage.list_events(session_id=session.id, limit=20)
    failed = next(event for event in events if event.event_type == "agent.run.failed")
    assert "sk-live-secret" not in str(failed.payload)
    assert "[REDACTED]" in str(failed.payload)


async def test_agent_loop_enforces_exclusive_session_entry(storage):
    session = await AgentSessionManager(storage).create()
    blocking_model = BlockingModel()
    first = AgentLoop(model=blocking_model, tools=build_gateway(storage), storage=storage)
    second = AgentLoop(
        model=ScriptedModel([ModelResponse(content='{"final": "second"}')]),
        tools=build_gateway(storage),
        storage=storage,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(first.run, session.id, "first")
        await blocking_model.started.wait()
        with pytest.raises(SessionLeaseError, match="already active"):
            await second.run(session.id, "second")
        blocking_model.release.set()

    result = await second.run(session.id, "second")
    assert result.final == "second"


async def test_session_lease_provider_heartbeats_active_sessions(storage):
    session = await AgentSessionManager(storage).create()
    provider = StorageSessionLeaseProvider(
        storage,
        owner_id="runtime-a",
        ttl_seconds=0.2,
        heartbeat_seconds=0.02,
    )
    lease = await provider.enter_session(session.id)

    await anyio.sleep(0.12)

    assert not await storage.try_acquire_session_lease(
        session.id,
        "runtime-b",
        ttl_seconds=0.2,
    )
    await lease.release()
    assert await storage.try_acquire_session_lease(
        session.id,
        "runtime-b",
        ttl_seconds=0.2,
    )
    await storage.release_session_lease(session.id, "runtime-b")


async def test_agent_loop_aborts_when_session_lease_is_lost(storage):
    session = await AgentSessionManager(storage).create()
    blocking_model = BlockingModel()
    provider = StorageSessionLeaseProvider(
        storage,
        owner_id="runtime-a",
        ttl_seconds=0.05,
        heartbeat_seconds=0.01,
    )
    loop = AgentLoop(
        model=blocking_model,
        tools=build_gateway(storage),
        storage=storage,
        lease_provider=provider,
    )

    async def run_agent() -> None:
        with pytest.raises(SessionLeaseError, match="lost ownership"):
            await loop.run(session.id, "hold")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_agent)
        await blocking_model.started.wait()
        await storage.release_session_lease(session.id, "runtime-a")


async def test_agent_loop_rejects_invalid_tool_concurrency(storage):
    model = ScriptedModel([ModelResponse(content='{"final": "done"}')])

    with pytest.raises(ValueError, match="tool_concurrency"):
        AgentLoop(
            model=model,
            tools=build_gateway(storage),
            storage=storage,
            tool_concurrency=0,
        )
