import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests

# -----------------
# 1. Connect to ChromaDB
# -----------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(
    name="rag_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # embedding model
    )
)

# -----------------
# 2. Retrieval (Step 7)
# -----------------
def retrieve_chunks(question, k=3):
    results = collection.query(query_texts=[question], n_results=k)
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
    "chunk_text": doc,
    "source": meta.get("source_file", "Unknown")  # safer
})
    return retrieved

# -----------------
# 3. Prompt design (Step 8)
# -----------------
def build_prompt(question, chunks):
    context_text = "\n\n".join([f"{i+1}. {chunk['chunk_text']}" for i, chunk in enumerate(chunks)])
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the user's Q question and provide the response under A.

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

# -----------------
# 4. Ask Ollama LLM (Step 8 continued)
# -----------------
def get_llm_answer(prompt, model="llama3.2:1b"):
    url = "http://localhost:11434/api/generate"  # Ollama local API endpoint
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
        "num_predict": 2048  # increase response length
    }
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        return f"Error from Ollama: {response.status_code} - {response.text}"

# -----------------
# 5. UI (Step 9)
# -----------------
st.title("📚 Mini RAG App (Chroma + Ollama)")
st.write("Ask a question based on your documents.")

user_query = st.text_input("Enter your question:")

if user_query:
    chunks = retrieve_chunks(user_query, k=3)
    prompt = build_prompt(user_query, chunks)
    answer = get_llm_answer(prompt, model="llama3.2:1b")

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("View Retrieved Chunks"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**{i}.** {chunk['chunk_text']}\n*Source:* {chunk['source']}")