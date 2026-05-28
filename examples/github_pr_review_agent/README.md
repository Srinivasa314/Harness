# GitHub PR Review Agent

This example runs a real GitHub pull-request review agent. It demonstrates the
main runtime features together:

- OpenAI or Codex model provider.
- MiniLM or OpenAI embeddings and scoped memory.
- Capability-gated tools.
- Environment-backed GitHub App credential resolution.
- Agent-directed Docker bash execution.
- Context compaction.
- SQLite observability records and dashboard inspection.

Required settings:

```bash
export HARNESS_GITHUB_REPO=owner/repository
export HARNESS_GITHUB_PR=123
export HARNESS_REPO_PATH=/path/to/local/checkout
export HARNESS_GITHUB_APP_ID=12345
export HARNESS_GITHUB_INSTALLATION_ID=67890
export HARNESS_SECRET_GITHUB_APP_PRIVATE_KEY='-----BEGIN RSA PRIVATE KEY-----...'
```

The GitHub App must be installed on the target repository and needs pull-request
read access, contents read access, checks read access, and issues write access
to post review comments. The example mints an installation token from the App
private key for every GitHub tool call. Personal access tokens are not supported
by this example, so PR comments are clearly authored by the installed App.
If the private key is stored on one line, encode newlines as `\n`; the example
normalizes them before signing the App JWT.

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

`HARNESS_REPO_PATH` should point to a clean local clone for the repository being
reviewed. Prefer a temporary clone created specifically for the review so local
ignored files, caches, and credentials are not present in the mounted tree. If
it is omitted, the example analyzes the current working directory.
The `repo.bash` tool mounts that checkout read-only at `/repo` so the agent can
explore files and configuration with shell commands without being able to modify
the host directory. The container has no network access.

The example requires Docker for the sandboxed project-check tool.
It keeps its SQLite database by default so memories can carry across runs. Set
`HARNESS_EXAMPLE_RESET_DB=1` to start from a clean database.

To teach review preferences, reply on the GitHub PR after the agent's comment,
then rerun the example:

```bash
uv run python examples/github_pr_review_agent/agent.py
```

The example reads PR comments after the latest Harness agent comment, ignores
agent comments, skips comments already processed in its local SQLite tracking
table, and stores durable preference memories at agent scope for later PR
reviews in the same namespace. Each run uses a new Harness session; memory
provides continuity across runs. The agent reads unread PR replies through the
`github.pr_replies` tool. Replies are accepted only from the PR author or users
GitHub marks as members/owners of the repository organization. Relevant stored
preferences are retrieved by Harness memory and injected into model context
automatically.

Commenting is done by the agent through the `github.pr_comment` tool after it
has run `github.pr_context` and explored the checkout with `repo.bash`.
GitHub shows the comment author as the installed GitHub App. The comment body is
also marked as coming from the Harness PR review agent.
