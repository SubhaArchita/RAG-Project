import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

CSV = "chunkAll.csv"

client = chromadb.PersistentClient(path="./chroma_db")

# Fresh start so you don't get duplicate-id errors when reindexing
try:
    client.delete_collection("rag_collection")
except Exception:
    pass

collection = client.create_collection(
    name="rag_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

df = pd.read_csv(CSV, encoding="utf-8")

collection.add(
    ids=df["chunk_id"].astype(str).tolist(),
    documents=df["chunk_text"].astype(str).tolist(),
    metadatas=df[["document_title","source_file","tags"]].to_dict("records")
)

print(f"✅ Added {len(df)} chunks to Chroma collection 'rag_collection' at ./chroma_db")
