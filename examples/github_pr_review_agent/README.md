# GitHub PR Review Agent

This example runs a real GitHub pull-request review agent. It demonstrates the
main runtime features together:

- OpenAI or Codex model provider.
- MiniLM or OpenAI embeddings and scoped memory.
- Capability-gated tools.
- Environment-backed GitHub credential resolution.
- Docker container tool execution.
- Context compaction.
- SQLite observability records and dashboard inspection.

Required settings:

```bash
export HARNESS_GITHUB_REPO=owner/repository
export HARNESS_GITHUB_PR=123
export HARNESS_REPO_PATH=/path/to/local/checkout
export HARNESS_SECRET_GITHUB_TOKEN=...
```

Set `HARNESS_OPENAI_API_KEY` when using OpenAI for inference or embeddings.

MiniLM is used by default. To use OpenAI embeddings instead:

```bash
export HARNESS_EMBEDDING_PROVIDER=openai
export HARNESS_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Run:

```bash
uv sync --extra dev --extra embeddings
uv run python examples/github_pr_review_agent/agent.py
```

`HARNESS_REPO_PATH` should point to a local checkout for the repository being
reviewed. If it is omitted, the example analyzes the current working directory.
The Dockerized project-check tool mounts that checkout read-only at `/repo` so
it can inspect files without being able to modify the host directory.

The example requires Docker for the sandboxed project-check tool.
It keeps its SQLite database by default so memories can carry across runs. Set
`HARNESS_EXAMPLE_RESET_DB=1` to start from a clean database.

To post the review as a PR comment:

```bash
export HARNESS_GITHUB_COMMENT=1
```

When enabled, commenting is done by the agent through the `github.pr_comment`
tool after it has run `github.pr_context` and `repo.project_check`.
GitHub shows the comment author based on the token used. A personal token posts
as that user; use a GitHub App or machine-user token if the comment should have
a dedicated bot identity. The comment body is marked as coming from the Harness
PR review agent.
