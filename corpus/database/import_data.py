import json
import sqlite3
from pathlib import Path


JSONL_FILE = Path(__file__).parent.parent / "data" / "SP 7_2026.chunks.jsonl"
DB_FILE = Path(__file__).parent / "standards.db"


def import_data():
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()

    with open(JSONL_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)

            cursor.execute("""
                INSERT OR IGNORE INTO chunks (
                    chunk_id,
                    page_start,
                    page_end,
                    section_number,
                    section_title,
                    text
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk.get("chunk_id"),
                chunk.get("page_start"),
                chunk.get("page_end"),
                chunk.get("section_number"),
                chunk.get("section_title"),
                chunk.get("text")
            ))

    connection.commit()
    connection.close()

    print("JSONL data imported successfully!")


if __name__ == "__main__":
    import_data()