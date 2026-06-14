#!/usr/bin/env python3
"""Convenience script: create the schema and ingest all sample data.

    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apply_advisor.ingest import run_full_ingest  # noqa: E402

if __name__ == "__main__":
    stats = run_full_ingest()
    print(
        f"Ingested {stats['programs']} programs across {stats['universities']} "
        f"universities, {stats['admission']} admission / {stats['language']} language / "
        f"{stats['document']} document / {stats['sources']} source rows, and "
        f"{stats['policy_chunks']} policy chunks."
    )
