from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

DEFAULT_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)


def filtered_env(
    *,
    source: Mapping[str, str] | None = None,
    inherit: Iterable[str] = DEFAULT_ENV_ALLOWLIST,
) -> dict[str, str]:
    env_source = os.environ if source is None else source
    return {key: env_source[key] for key in inherit if key in env_source}
