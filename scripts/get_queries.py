#!/usr/bin/env python3

import sqlite3
from pathlib import Path

import fire  # type: ignore


def get_all_queries(db_file: str = "queries.db") -> None:
    db_path = Path(db_file)
    if not db_path.exists():
        print(f"Database file not found: {db_file}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            query,
            COUNT(*) as count,
            AVG(results_count) as avg_results,
            MAX(timestamp) as last_query,
            SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
        FROM queries
        GROUP BY query
        ORDER BY count DESC
        """
    )

    rows = cursor.fetchall()

    if not rows:
        print("No queries found in database")
        conn.close()
        return

    print(f"\nTotal unique queries: {len(rows)}\n")
    print(f"{'Query':<50} {'Count':<8} {'Avg Results':<12} {'Errors':<8} {'Last Query':<20}")
    print("=" * 120)

    for row in rows:
        query = row["query"]
        if len(query) > 47:
            query = query[:44] + "..."

        avg_results = f"{row['avg_results']:.1f}" if row["avg_results"] else "N/A"

        print(f"{query:<50} {row['count']:<8} {avg_results:<12} {row['error_count']:<8} {row['last_query']:<20}")

    conn.close()


def get_recent_queries(limit: int = 20, db_file: str = "queries.db") -> None:
    db_path = Path(db_file)
    if not db_path.exists():
        print(f"Database file not found: {db_file}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            query,
            top_k,
            timestamp,
            results_count,
            error
        FROM queries
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()

    if not rows:
        print("No queries found in database")
        conn.close()
        return

    print(f"\nLast {limit} queries:\n")
    print(f"{'ID':<6} {'Query':<40} {'Top K':<8} {'Results':<10} {'Timestamp':<20} {'Error':<15}")
    print("=" * 120)

    for row in rows:
        query = row["query"]
        if len(query) > 37:
            query = query[:34] + "..."

        error = "Yes" if row["error"] else "No"
        results = str(row["results_count"]) if row["results_count"] is not None else "N/A"

        print(f"{row['id']:<6} {query:<40} {row['top_k']:<8} {results:<10} {row['timestamp']:<20} {error:<15}")

    conn.close()


def get_stats(db_file: str = "queries.db") -> None:
    db_path = Path(db_file)
    if not db_path.exists():
        print(f"Database file not found: {db_file}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM queries")
    total_queries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM queries WHERE error IS NOT NULL")
    error_queries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT query) FROM queries")
    unique_queries = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(results_count) FROM queries WHERE results_count IS NOT NULL")
    avg_results = cursor.fetchone()[0]

    print("\n=== Query Statistics ===")
    print(f"Total queries: {total_queries}")
    print(f"Unique queries: {unique_queries}")
    print(f"Failed queries: {error_queries}")
    print(f"Average results per query: {avg_results:.1f}" if avg_results else "Average results: N/A")

    cursor.execute(
        """
        SELECT query, COUNT(*) as count
        FROM queries
        GROUP BY query
        ORDER BY count DESC
        LIMIT 5
        """
    )

    top_queries = cursor.fetchall()
    if top_queries:
        print("\nTop 5 most common queries:")
        for i, (query, count) in enumerate(top_queries, 1):
            display_query = query[:60] + "..." if len(query) > 60 else query
            print(f"  {i}. ({count}x) {display_query}")

    conn.close()


if __name__ == "__main__":
    fire.Fire(
        {
            "all": get_all_queries,
            "recent": get_recent_queries,
            "stats": get_stats,
        }
    )
