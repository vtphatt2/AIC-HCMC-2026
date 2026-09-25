import argparse
import json
import sys

from .codex import CodexPlanner
from .search_agent import SearchAgent
from .tools import VortaClient


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the VORTA Search Agent terminal slice")
    parser.add_argument("query", help="Official/raw query")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--planner-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    client = VortaClient(args.base_url)
    try:
        result = SearchAgent(
            CodexPlanner(timeout_seconds=args.planner_timeout),
            client,
        ).run(args.query, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
