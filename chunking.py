import os
import re
import docx
import pandas as pd
from PyPDF2 import PdfReader

# -------------------------
# CONFIGURATION
# -------------------------
FILE_PATH = [
    "BillGates.txt",
    "AjantaCaves.txt",
    "BiographyofRudyardKipling.txt",
    "CarSafetyManual.txt",
    "MoneyHeist.txt",
    "News.txt",
    "Recipies.txt",
    "Sholay.txt",
    "SkateboardingClub.txt"
             ]  

CHUNK_SIZE = 300  # Target chunk size in words
CHUNK_OVERLAP = 50  # Overlap between chunks in words

# -------------------------
# HELPERS
# -------------------------

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    # Remove extra spaces and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move start with overlap
    return chunks

# -------------------------
# MAIN LOGIC
# -------------------------


data = []

for file_path in FILE_PATH:
    ext = file_path.lower().split(".")[-1]

    # Read file
    if ext == "txt":
        text = read_txt(file_path)
    elif ext == "pdf":
        text = read_pdf(file_path)
    elif ext == "docx":
        text = read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    text = clean_text(text)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for i, chunk in enumerate(chunks, start=1):
        data.append({
            "chunk_id": f"{file_path}_chunk{i}",
            "chunk_text": chunk,
            "document_title": file_path.rsplit(".", 1)[0],
            "source_file": file_path,
            "tags": ""  # You can fill manually later if needed
        })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("chunkAll.csv", index=False, encoding="utf-8")
print(f"✅ Chunking complete! Saved {len(df)} chunks from {FILE_PATH} to chunkAll.csv")


