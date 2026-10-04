"""CLI for the per-stage latency report.

    uv run python scripts/latency_report.py            # every call in calls/
    uv run python scripts/latency_report.py calls/*.jsonl

Analysis lives in backend/analytics/latency.py, because the API serves it too and
a request handler must not import a CLI script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.analytics.latency import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
