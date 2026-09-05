import sqlite3

connection = sqlite3.connect("standards.db")
cursor = connection.cursor()

cursor.execute("SELECT * FROM standards")

rows = cursor.fetchall()

for row in rows:
    print(row)

connection.close()