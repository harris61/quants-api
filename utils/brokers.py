"""
Broker category utilities.
"""

import csv
from pathlib import Path
from typing import Dict

from config import BASE_DIR


def load_broker_categories(path: Path = None) -> Dict[str, dict]:
    """Load broker categories from CSV into a code -> metadata map."""
    csv_path = path or (BASE_DIR / "data" / "broker_categories.csv")
    if not csv_path.exists():
        return {}

    categories = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = (row.get("broker_code") or "").strip().upper()
            if not code:
                continue
            categories[code] = {
                "broker_name": (row.get("broker_name") or "").strip(),
                "category": (row.get("category") or "").strip(),
                "source": (row.get("source") or "").strip(),
            }

    return categories
