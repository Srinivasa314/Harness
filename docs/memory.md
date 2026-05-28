# Memory

Memory is durable retrieval state for later model turns. It is separate from
context compaction: memory retrieves facts and preferences across time, while
compaction summarizes old transcript turns to stay within the active context
budget.

## Components

- `EmbeddingProvider`: turns text into vectors.
- `MemoryStore`: persists memory records and vectors.
- `MemoryManager`: retrieves, formats, and optionally captures memories around
  an agent run.
- `MemoryPolicy`: controls namespace, limits, formatting budget, and
  auto-capture behavior.

Built-in embedding providers:

- `MiniLMEmbeddingProvider`: local sentence-transformers embeddings.
- `OpenAIEmbeddingProvider`: OpenAI embeddings API.
- `HashEmbeddingProvider`: deterministic vectors for tests.

Applications can inject these components directly or register named embedding
providers for settings-based selection.

Each stored memory records the embedding provider, model, and vector dimension
that produced it. Retrieval only compares memories produced by the active
embedding configuration, so changing providers, models, or OpenAI dimensions
does not silently mix incompatible vectors. Use a new `HARNESS_MEMORY_NAMESPACE`
when you intentionally want a clean memory set for a different embedding
configuration.

## Scopes

Supported memory scopes:

- `session`: visible only inside the source session.
- `agent`: shared within the configured namespace.
- `global`: shared broadly within the configured namespace.

Retrieval filters session memories to the active session, searches agent and
global memories within the namespace, ranks by similarity to the query
embedding, and returns a bounded context block for the next model call.

## Stored Records

Stored memories include:

- redacted text;
- metadata;
- scope and namespace;
- confidence;
- source references;
- timestamps;
- vector data.

Memory text and retrieval queries are redacted before embedding. This protects
against obvious accidental secret persistence while still allowing retrieval over
the redacted text that is stored.

## Auto-Capture

Auto-capture is optional. When enabled, the manager can summarize useful facts
from a completed run and store them as scoped memories. Applications can also
write memories explicitly through `MemoryStore` or the manager.

Use explicit writes when the application already knows what should become
memory. Use auto-capture when the agent transcript is the source of truth and
occasional summarization calls are acceptable.

Good durable memories are usually stable user preferences, project conventions,
or facts that should influence future sessions. Avoid storing noisy run facts
that are only useful for the current task, such as a specific PR's changed-file
count. The GitHub PR review example uses auto-capture for user review
preferences expressed in replies, then retrieves those preferences for later PR
reviews in the same namespace.

## Compaction

Compaction is not memory. The rolling compactor summarizes older turns from the
current session when the transcript approaches the context budget. It uses the
configured `ModelProvider` by default and persists checkpoint summaries as
turns, so later runs can reconstruct a compact transcript.

Disable compaction with:

```bash
export HARNESS_CONTEXT_COMPACTION_ENABLED=false
```

Replace it by passing a custom `ContextCompactor` to `RuntimeContext.agent_loop()`
or directly to `AgentLoop`.
