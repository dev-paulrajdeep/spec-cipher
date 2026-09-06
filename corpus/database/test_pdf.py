import json
from dataclasses import asdict

from corpus.ingestion.pdf_reader import PDFReader


pdf_path = "national building construction standards 2026 v1.pdf"

reader = PDFReader()

document = reader.read(pdf_path)

data = asdict(document)

with open("extracted.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

print("PDF extracted successfully!")
print("File:", document.source_file)
print("Total pages:", document.page_count)
print("JSON saved as extracted.json")