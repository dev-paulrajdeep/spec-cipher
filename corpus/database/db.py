import sqlite3

def create_database():
    connection = sqlite3.connect("standards.db")

    cursor = connection.cursor()

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

    connection.commit()
    connection.close()

    print("Database created successfully!")


if __name__ == "__main__":
    create_database()