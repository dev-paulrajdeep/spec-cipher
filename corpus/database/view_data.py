import sqlite3

DB_FILE = "corpus/database/standards.db"

connection = sqlite3.connect(DB_FILE)
cursor = connection.cursor()

cursor.execute("""
    SELECT id, chunk_id, page_start, page_end, section_title, text
    FROM chunks
    LIMIT 3
""")

rows = cursor.fetchall()

for row in rows:
    print("\nID:", row[0])
    print("Chunk:", row[1])
    print("Pages:", row[2], "-", row[3])
    print("Section:", row[4])
    print("Text:", (row[5] or "")[:500][:500])

connection.close()

