import sqlite3
from pathlib import Path


DB_FILE = Path(__file__).parent / "standards.db"


def create_database():
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()

    # Store information about each standard
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS standards (
            id INTEGER PRIMARY KEY,
            standard_number TEXT,
            title TEXT,
            year INTEGER,
            edition TEXT,
            revision TEXT,
            status TEXT,
            source_file TEXT,
            page_count INTEGER
        )
    """)

    # Store the actual chunks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            page_start INTEGER,
            page_end INTEGER,
            section_number TEXT,
            section_title TEXT,
            text TEXT
        )
    """)

    # If an older chunks table exists without the text column, add it
    try:
        cursor.execute("ALTER TABLE chunks ADD COLUMN text TEXT")
    except sqlite3.OperationalError:
        pass

    connection.commit()
    connection.close()

    print("Database created successfully!")


if __name__ == "__main__":
    create_database()