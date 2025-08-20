import streamlit as st
import re, docx
import chromadb
import requests
import pandas as pd
from PyPDF2 import PdfReader
from chromadb.utils import embedding_functions

# -----------------
# CONFIG
# -----------------
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Chroma setup (persistent so docs remain across runs)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# -----------------
# HELPERS
# -----------------
def read_txt(file): return file.read().decode("utf-8")
def read_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(t): return re.sub(r'\s+', ' ', t).strip()

def chunk_text(t, size, overlap):
    words, chunks, start = t.split(), [], 0
    while start < len(words):
        chunk = " ".join(words[start:start+size])
        chunks.append(chunk)
        start += size - overlap
    return chunks

def add_to_db(uploaded_files):
    data = []
    for file in uploaded_files:
        ext = file.name.lower().split(".")[-1]
        if ext=="txt": text = read_txt(file)
        elif ext=="pdf": text = read_pdf(file)
        elif ext=="docx": text = read_docx(file)
        else: 
            st.error(f"Unsupported file type: {ext}")
            continue

        text = clean_text(text)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks, start=1):
            doc_id = f"{file.name}_chunk{i}"
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[{
                    "document_title": file.name.rsplit(".",1)[0],
                    "source_file": file.name,
                    "tags": ""
                }]
            )
            data.append({"id": doc_id, "chunk": chunk})
    return pd.DataFrame(data)

def retrieve_chunks(question, k=3):
    results = collection.query(query_texts=[question], n_results=k)
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "chunk_text": doc,
            "source": meta.get("source_file","Unknown")
        })
    return retrieved

def build_prompt(question, chunks):
    context_text = "\n\n".join(
        [f"{i+1}. {chunk['chunk_text']}" for i, chunk in enumerate(chunks)]
    )
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the user's question and provide the response.

Guidelines:
- Read the context carefully but do not copy-paste large parts of it.
- If multiple parts of the context are relevant, summarize them in your own words, but don't go out of the context provided in the text.
- Only include a short quote if absolutely necessary to support your answer.
- Ignore unrelated details or repeated Q&A.
- If the answer cannot be found, say "I don't know based on the provided information."


Context:
{context_text}

Question: {question}
Answer:
"""
    return prompt

def get_llm_answer(prompt, model="llama3.2:1b"):
    url = "http://localhost:11434/api/generate"  # Ollama
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 2048}
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response","").strip()
    else:
        return f"Error {response.status_code}: {response.text}"

# -----------------
# STREAMLIT UI
# -----------------
st.title("📚 RAG App (Upload → ChromaDB → Ask Questions)")

tab1, tab2 = st.tabs(["📂 Upload Documents", "💬 Ask Questions"])

with tab1:
    st.header("Upload and Index Docs")
    uploaded_files = st.file_uploader("Upload files", type=["txt","pdf","docx"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process & Add to Database"):
            df = add_to_db(uploaded_files)
            st.success(f"✅ Added {len(df)} chunks from {len(uploaded_files)} files")
            st.dataframe(df)

with tab2:
    st.header("Ask Questions")
    user_query = st.text_input("Enter your question:")
    if user_query:
        chunks = retrieve_chunks(user_query, k=3)
        if chunks:
            prompt = build_prompt(user_query, chunks)
            answer = get_llm_answer(prompt)
            st.subheader("Answer:")
            st.write(answer)

            with st.expander("View Retrieved Chunks"):
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**{i}.** {chunk['chunk_text']}\n*Source:* {chunk['source']}")
        else:
            st.warning("No relevant chunks found. Try uploading docs first.")
