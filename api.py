import sys
import subprocess

def install_if_needed():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except:
        print("dependências sendo instaladas")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                               "fastapi", "uvicorn", "requests", "beautifulsoup4", 
                               "numpy", "faiss-cpu", "sentence-transformers"])
install_if_needed()

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware   
import uvicorn
import requests, re, numpy as np, faiss, pickle, os, time
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
INDEX_FILE = 'faiss_index.faiss'
CHUNKS_FILE = 'chunks.pkl'

URLS = [
    "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial",
    "https://pt.wikipedia.org/wiki/Aprendizado_de_m%C3%A1quina",
    "https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural",
    "https://pt.wikipedia.org/wiki/Redes_neurais_artificiais",
    "https://pt.wikipedia.org/wiki/Transformador_(arquitetura_de_aprendizagem_profunda)",
]

app = FastAPI(title="Busca Semântica")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def scrape_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            content = soup.find("div", id="mw-content-text") or soup
            text = re.sub(r'\s+', ' ', content.get_text()).strip()
            text = re.sub(r'\[.*?\]', '', text)
            return text
        except:
            time.sleep(1)
    return ""

def chunk_text(text: str, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

index = None
chunks = None
model = None

def build_or_load():
    global index, chunks, model
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Índices sendo carregados")
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f: chunks = pickle.load(f)
        model = SentenceTransformer(MODEL_NAME)
    else:
        print("índices sendo construindos")
        model = SentenceTransformer(MODEL_NAME)
        all_chunks = []
        for url in URLS:
            text = scrape_page(url)
            all_chunks.extend(chunk_text(text))
            time.sleep(0.8)
        
        embeddings = model.encode(all_chunks)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        
        faiss.write_index(index, INDEX_FILE)
        with open(CHUNKS_FILE, "wb") as f: pickle.dump(all_chunks, f)
        chunks = all_chunks
    print(f"{len(chunks)} chunks carregados!")

@app.get("/")
def root():
    return {"status": "OK", "use": "/search?q=Sua pergunta aqui"}

@app.get("/search")
def search(q: str, top_k: int = 5):
    if not q: return JSONResponse({"erro": "Use ?q=texto"}, status_code=400)
    
    global index, chunks, model
    if index is None: build_or_load()
    
    q_emb = model.encode([q])[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    q_emb = np.array([q_emb]).astype(np.float32)
    
    distances, indices = index.search(q_emb, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        sim = 1 - (distances[0][i] ** 2) / 2
        results.append({
            "rank": i+1,
            "similarity": round(float(sim), 4),
            "text": chunks[idx][:420] + "..."
        })
    
    return {"query": q, "results": results}

if __name__ == "__main__":
    build_or_load()
    print("\nAPI rodando!")
    print("   Teste: http://127.0.0.1:8000/search?q=Transformer")
    uvicorn.run(app, host="0.0.0.0", port=8000)