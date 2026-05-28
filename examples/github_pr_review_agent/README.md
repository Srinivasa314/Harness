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

Use this example only with repositories and pull requests you trust. The agent
clones the PR, runs arbitrary read-only shell commands inside a Docker
container, and publishes its review text to the PR through a GitHub App comment.

Required settings:

```bash
export HARNESS_GITHUB_REPO=owner/repository
export HARNESS_GITHUB_PR=123
export HARNESS_GITHUB_APP_ID=12345
export HARNESS_GITHUB_INSTALLATION_ID=67890
export HARNESS_SECRET_GITHUB_APP_PRIVATE_KEY='-----BEGIN RSA PRIVATE KEY-----...'
```

The GitHub App must be installed on the target repository and needs contents
read access, checks read access, issues write access, and pull-request write
access to post review comments. The example mints an installation token from the App
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

At startup, the example uses the GitHub App installation token to fetch
`refs/pull/<number>/head` into a temporary review workspace, removes `.git`, and
mounts the resulting working tree read-only at `/repo`. Local ignored files,
caches, and credentials are not present in the container. The container has no
network access.

The example requires Docker for the sandboxed project-check tool.
It keeps its SQLite database by default so memories can carry across runs. Set
`HARNESS_EXAMPLE_RESET_DB=1` to start from a clean database.
Context compaction is enabled with conservative defaults for this example. Use
`HARNESS_PR_REVIEW_CONTEXT_MAX_CHARS` and
`HARNESS_PR_REVIEW_COMPACTION_TRIGGER_RATIO` to tune when long reviews compact.

To teach review preferences, reply on the GitHub PR after the agent's comment,
then rerun the example:

```bash
uv run python examples/github_pr_review_agent/agent.py
```

The example reads PR comments after the latest Harness agent comment, ignores
agent comments, skips comments already processed in its local SQLite tracking
table, and can store durable preference memories at agent scope for later PR
reviews in the same namespace. Each run uses a new Harness session; memory
provides continuity across runs. The agent reads unread PR replies through the
`github.pr_replies` tool. Replies are accepted only after the latest Harness
agent comment, only from users GitHub marks as repository owners, and only when
the comment directly mentions the bot, such as `@harness-pr-review-agent`.
Set `HARNESS_GITHUB_APP_SLUG` if your installed App uses another mention slug.
When a reply contains reusable review guidance, the agent stores it by calling
the generic `memory.store` tool; relevant stored preferences are later retrieved
by Harness memory and injected into model context automatically.

Commenting is done by the agent through the `github.pr_comment` tool after it
has run `github.pr_context` and explored the checkout with `repo.bash`.
GitHub shows the comment author as the installed GitHub App. The comment body is
also marked as coming from the Harness PR review agent.
