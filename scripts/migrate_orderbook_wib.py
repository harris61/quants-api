"""
Migrate orderbook snapshot timestamps from UTC to WIB (+07:00).
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def run(apply: bool, force_date: str | None) -> None:
    db_path = Path(__file__).resolve().parents[1] / "database" / "quants.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM orderbook_snapshots")
    total = cur.fetchone()[0]
    if total == 0:
        print("No orderbook snapshots found.")
        conn.close()
        return

    cur.execute("SELECT id, captured_at FROM orderbook_snapshots")
    rows = cur.fetchall()

    updates = []
    for row_id, captured_at in rows:
        dt = _parse_dt(captured_at)
        if dt is None:
            continue
        new_dt = dt + timedelta(hours=7)
        if force_date:
            new_date = force_date
        else:
            new_date = new_dt.date().isoformat()
        updates.append((new_dt.strftime("%Y-%m-%d %H:%M:%S.%f"), new_date, row_id))

    print(f"Snapshots found: {total}")
    print(f"Snapshots parsed: {len(updates)}")

    if not apply:
        print("Dry run only. Use --apply to write changes.")
        conn.close()
        return

    cur.executemany(
        "UPDATE orderbook_snapshots SET captured_at = ?, date = ? WHERE id = ?",
        updates,
    )
    conn.commit()
    print(f"Updated {cur.rowcount} rows.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shift orderbook snapshots from UTC to WIB.")
    parser.add_argument("--apply", action="store_true", help="Apply updates to the database")
    parser.add_argument(
        "--force-date",
        type=str,
        help="Override date column with a fixed YYYY-MM-DD value",
    )
    args = parser.parse_args()

    run(args.apply, args.force_date)


if __name__ == "__main__":
    main()
