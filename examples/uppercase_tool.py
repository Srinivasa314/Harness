from __future__ import annotations

import json
import sys


def main() -> None:
    payload = json.load(sys.stdin)
    text = payload["arguments"]["text"]
    print(json.dumps({"value": text.upper()}))


if __name__ == "__main__":
    main()
