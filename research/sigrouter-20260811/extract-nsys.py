#!/usr/bin/env python3
"""Extract exact CUDA memcpy byte/count groups from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    try:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        memcpy_table = "CUPTI_ACTIVITY_KIND_MEMCPY"
        if memcpy_table not in tables:
            raise SystemExit(f"missing {memcpy_table}; tables={sorted(tables)}")
        columns = [
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{memcpy_table}")')
        ]
        required = {"copyKind", "bytes"}
        if not required.issubset(columns):
            raise SystemExit(f"unexpected memcpy schema: {columns}")

        enum_rows: list[dict[str, object]] = []
        for table in sorted(name for name in tables if "MEMCPY" in name and name != memcpy_table):
            enum_columns = [
                str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')
            ]
            rows = connection.execute(f'SELECT * FROM "{table}"').fetchall()
            enum_rows.append(
                {
                    "table": table,
                    "columns": enum_columns,
                    "rows": [list(row) for row in rows],
                }
            )

        groups = [
            {"copy_kind": int(kind), "bytes": int(size), "count": int(count)}
            for kind, size, count in connection.execute(
                f'SELECT copyKind, bytes, COUNT(*) FROM "{memcpy_table}" '
                "GROUP BY copyKind, bytes ORDER BY copyKind, bytes"
            )
        ]
        result = {
            "schema": "memra.sigrouter.nsys-memcpy.v1",
            "database": args.database.name,
            "memcpy_columns": columns,
            "memcpy_groups": groups,
            "memcpy_total": sum(int(group["count"]) for group in groups),
            "enum_tables": enum_rows,
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
