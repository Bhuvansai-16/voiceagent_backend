"""CLI for the LLM quality metrics.

    uv run python scripts/quality_metrics.py            # every call in calls/
    uv run python scripts/quality_metrics.py calls/*.jsonl

Analysis lives in backend/analytics/quality.py, because the API serves it too and
a request handler must not import a CLI script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.analytics.quality import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
