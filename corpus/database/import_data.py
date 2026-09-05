import sqlite3


connection = sqlite3.connect("standards.db")

cursor = connection.cursor()


cursor.execute("""
    INSERT INTO standards
    (standard_number, title, year, edition, revision, status, source_file, page_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    "IS 13252",
    "Information technology equipment",
    2010,
    None,
    None,
    "current",
    "IS_13252.pdf",
    50
))


connection.commit()
connection.close()

print("Data added successfully!")