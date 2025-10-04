#!/usr/bin/env python3
"""
Migrate exercise_sessions data to compliance_events table.

Rules:
- intervention_id = 5 if time is >= 16:00 (4 PM), otherwise 4
- Convert timestamp from ISO format to datetime format
- value_bool = 1 (True)
- notes = NULL
- source = 'migration'
"""

import sqlite3
from datetime import datetime

DB_PATH = r"c:\Users\jspai\Projects\hrv-lab\hrv_lab.sqlite3"

def migrate():
    # Set timeout to wait for database lock to be released
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()

    try:
        # Get all exercise session timestamps
        cursor.execute("SELECT timestamp FROM exercise_sessions ORDER BY timestamp")
        sessions = cursor.fetchall()

        print(f"Found {len(sessions)} exercise sessions to migrate")

        # Check current max ID in compliance_events
        cursor.execute("SELECT MAX(id) FROM compliance_events")
        max_id = cursor.fetchone()[0] or 0
        print(f"Current max compliance_events ID: {max_id}")

        # Prepare insert data
        inserts = []
        for row in sessions:
            timestamp_iso = row[0]

            # Parse the ISO timestamp (e.g., "2025-08-07T13:44:51")
            dt = datetime.fromisoformat(timestamp_iso)

            # Determine intervention_id based on hour
            intervention_id = 5 if dt.hour >= 16 else 4

            # Format timestamp for compliance_events
            # Format: "2025-08-07 13:44:51.000000"
            ts_formatted = dt.strftime('%Y-%m-%d %H:%M:%S.%f')

            inserts.append((
                intervention_id,
                ts_formatted,
                1,  # value_bool
                None,  # notes
                'migration'  # source
            ))

        # Insert all records
        cursor.executemany(
            """INSERT INTO compliance_events (intervention_id, ts, value_bool, notes, source)
               VALUES (?, ?, ?, ?, ?)""",
            inserts
        )

        conn.commit()

        # Verify
        cursor.execute("SELECT COUNT(*) FROM compliance_events WHERE source = 'migration'")
        count = cursor.fetchone()[0]
        print(f"[OK] Successfully inserted {count} compliance events from exercise sessions")

        # Show breakdown by intervention
        cursor.execute("""
            SELECT intervention_id, COUNT(*)
            FROM compliance_events
            WHERE source = 'migration'
            GROUP BY intervention_id
        """)
        for intervention_id, count in cursor.fetchall():
            print(f"  - Intervention {intervention_id}: {count} events")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
