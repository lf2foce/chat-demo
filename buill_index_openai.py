import os
import time
import tiktoken
import faiss
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Cấu hình embedding model
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    embed_batch_size=1  # mặc định 1, ta sẽ tự batch chung sau
)

tokenizer = tiktoken.get_encoding("cl100k_base")  # phù hợp với embedding

def chunk_text(text, max_tokens=4000, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
        if i + max_tokens >= len(tokens):
            break
    return chunks

def safe_embed_batch(texts, max_retry=5, delay=60):
    for attempt in range(max_retry):
        try:
            return [Settings.embed_model.get_text_embedding(t) for t in texts]
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg or "RESOURCE_EXHAUSTED" in msg:
                print(f"[{attempt+1}/{max_retry}] Rate limit. Waiting {delay}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Embedding batch failed after retries")

# Load documents
docs = SimpleDirectoryReader("./docs").load_data()

all_chunks = []  # chứa tuple (Document, chunk_text)
for doc in docs:
    for chunk in chunk_text(doc.text, max_tokens=4000):
        d = Document(text=chunk, metadata=dict(doc.metadata))
        all_chunks.append(d)

print(f"Total chunks: {len(all_chunks)}")

# Batch embedding
embedded = []
batch_size = 50  # điều chỉnh nếu cần
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i+batch_size]
    texts = [d.text for d in batch]
    embeddings = safe_embed_batch(texts)
    for d, vec in zip(batch, embeddings):
        d.embedding = vec
        embedded.append(d)
    print(f"Embedded {len(embedded)}/{len(all_chunks)} chunks")

# Build FAISS index
if embedded:
    dim = len(embedded[0].embedding)
    index_store = FaissVectorStore(faiss.IndexFlatL2(dim))
    storage = StorageContext.from_defaults(vector_store=index_store)
    idx = VectorStoreIndex.from_documents(documents=embedded, storage_context=storage)
    storage.persist(persist_dir="./storage/csv-openai-small")
    print("✅ Index đã build và lưu tại ./storage/csv-openai-small")
else:
    print("❌ No chunks embedded.")
