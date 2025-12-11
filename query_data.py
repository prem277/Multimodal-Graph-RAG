# query_data.py
import numpy as np
import ast
from math import log
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from gremlin_python.driver.client import Client
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
import json
# ---------------- CONFIG ----------------
TEXT_EMBED_MODEL = "phi3:mini"
LLM_MODEL = "phi3:mini"
TOP_K = 5

HYBRID_ALPHA = 0.7  # Weight for embedding similarity
HYBRID_BETA = 0.3   # Weight for BM25 score

# ---------------- CLIENT ----------------
client = Client(
    "ws://localhost:8182/gremlin",
    "g",
    message_serializer=GraphSONSerializersV3d0()
)

# ---------------- EMBEDDINGS & LLM ----------------
text_embedder = OllamaEmbeddings(model=TEXT_EMBED_MODEL)
llm = OllamaLLM(model=LLM_MODEL, temperature=0)

# ---------------- PROMPTS ----------------
CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Given the chat history and a follow-up question, rewrite the question to be a standalone question.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:""")

QA_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about documents.

Use ONLY the following context. If you don't know, say: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

condense_chain = CONDENSE_PROMPT | llm
qa_chain = QA_PROMPT | llm

TARGET_DIM = 3072

def resize_embedding(vec):
    arr = np.array(vec, dtype=np.float32)
    if arr.shape[0] >= TARGET_DIM:
        return arr[:TARGET_DIM].tolist()
    pad_len = TARGET_DIM - arr.shape[0]
    padded = np.pad(arr, (0, pad_len), mode="constant")
    return padded.tolist()

# ---------------- COSINE SIM ----------------
def cosine_sim(a: list, b: list) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.size != b.size:
        # best-effort: resize the smaller one to match the larger
        if a.size < b.size:
            a = np.pad(a, (0, b.size - a.size), mode="constant")
        else:
            b = np.pad(b, (0, a.size - b.size), mode="constant")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# ---------------- SAFE & WORKING PARSING (2025 version) ----------------
def parse_embedding(embedding_prop):
    if not embedding_prop:
        return []
    s = embedding_prop[0] if isinstance(embedding_prop, list) else embedding_prop
    
    # We now store proper JSON → try this first
    try:
        return json.loads(s)
    except:
        pass
    
    # Fallback: old str([1.2, 3.4, ...]) format
    try:
        return ast.literal_eval(s)
    except:
        pass
    
    # Last resort: manual regex (very rare)
    import re
    nums = re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', s)
    return [float(x) for x in nums]

# ---------------------- BM25 -------------------------
def tokenize(text):
    return text.lower().split()

def compute_bm25_score(query_terms, doc_terms, df, N, avgdl, k1=1.5, b=0.75):
    score = 0.0
    dl = len(doc_terms)
    for term in query_terms:
        if term not in doc_terms:
            continue
        tf = doc_terms.count(term)
        idf = log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
        denom = tf + k1 * (1 - b + b * (dl / avgdl))
        score += (idf * tf * (k1 + 1)) / denom
    return float(score)

# ---------------- HYBRID SEARCH ----------------
def hybrid_search(query: str):
    q_emb = text_embedder.embed_query(query)
    query_terms = tokenize(query)

    # fetch chunks + images
    chunks = client.submit("g.V().hasLabel('CHUNK').valueMap(true)").all().result()
    images = client.submit("g.V().hasLabel('IMAGE').valueMap(true)").all().result()

    corpus_terms = []
    df = {}
    for vertex in chunks:
        text = vertex.get('text', [''])[0]
        terms = tokenize(text)
        corpus_terms.append(terms)
        for t in set(terms):
            df[t] = df.get(t, 0) + 1

    N = len(corpus_terms)
    avgdl = (sum(len(t) for t in corpus_terms) / N) if N > 0 else 1

    results = []
    idx = 0
    for vertex in chunks:
        text = vertex.get('text', [''])[0]
        emb_raw = vertex.get('embedding')
        if not emb_raw:
            idx += 1
            continue
        emb = parse_embedding(emb_raw)
        if not emb:
            idx += 1
            continue

        # ensure both vectors are same dim (resize if needed)
        emb = resize_embedding(emb)
        embed_score = cosine_sim(q_emb, emb)
        bm25_score = compute_bm25_score(query_terms, corpus_terms[idx], df, N, avgdl)
        hybrid = HYBRID_ALPHA * embed_score + HYBRID_BETA * bm25_score

        source = f"[Document] {vertex.get('file',[''])[0]} - chunk {vertex.get('chunk_index',[0])[0]}"
        results.append((hybrid, source, text))
        idx += 1

    # images: parse embedding, resize to TARGET_DIM (if not already), compare with q_emb
    for vertex in images:
        emb_raw = vertex.get('embedding')
        if not emb_raw:
            continue
        emb = parse_embedding(emb_raw)
        if not emb:
            continue
        emb = resize_embedding(emb)
        embed_score = cosine_sim(q_emb, emb)
        hybrid = HYBRID_ALPHA * embed_score  # no BM25 for images
        file = vertex.get('file', ['unknown'])[0]
        results.append((
            hybrid,
            f"[Chart Image] {file}",
            "This is a scientific chart/image. A detailed textual description generated by LLaVA is stored and will appear in the context below."
        ))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:TOP_K]

# ---------------- RAG PIPELINE ----------------
def answer_question(question: str, chat_history: str = "") -> str:
    condensed = condense_chain.invoke({
        "chat_history": chat_history,
        "question": question
    })
    standalone_q = condensed.content if hasattr(condensed, "content") else str(condensed)

    retrieved = hybrid_search(standalone_q)

    context_parts = []
    for score, source, text in retrieved:
        context_parts.append(f"{source}\n{text}\n(relevance: {score:.4f})")
    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    resp = qa_chain.invoke({
        "context": context,
        "question": standalone_q
    })
    answer = resp.content if hasattr(resp, "content") else str(resp)

    return answer

# ---------------- CLI ----------------
if __name__ == "__main__":
    print("Multimodal RAG + Hybrid Search (BM25 + Embedding) Ready!\n")
    history = ""

    try:
        while True:
            q = input("Ask: ").strip()
            if q.lower() in {"quit", "exit"}:
                break
            if not q:
                continue

            print("\nThinking...")
            answer = answer_question(q, history)
            print("\nAnswer:\n")
            print(answer)

            history += f"Human: {q}\nAssistant: {answer}\n\n"
    finally:
        client.close()
