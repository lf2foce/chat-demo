import os
import time
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

import faiss
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ Cấu hình embedding model
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-exp-03-07",
    embed_batch_size=1,  # Siêu an toàn, 1 doc/lần
    # dimensions=1536,
)

# 2️⃣ Đọc tài liệu
docs = SimpleDirectoryReader("./docs").load_data()

# 3️⃣ Hàm embed an toàn với retry 429
def safe_embed(text, max_retry=5, delay=60):
    for attempt in range(max_retry):
        try:
            return Settings.embed_model.get_text_embedding(text)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"[{attempt+1}/{max_retry}] Quota exceeded. Waiting {delay}s before retrying...")
                time.sleep(delay)
            else:
                print(f"Other error: {e}")
                raise
    print("Failed after retries, skipping this doc.")
    return None

# 4️⃣ Tạo embedding cho từng doc, lưu lại kết quả thành list
embedded_docs = []
for i, doc in enumerate(docs):
    print(f"\nEmbedding doc {i+1}/{len(docs)}: {doc.metadata.get('file_path', '') or doc.metadata.get('filename', '') or ''}")
    vec = safe_embed(doc.text)
    if vec is not None:
        # Gán embedding cho doc, rồi thêm vào list
        doc.embedding = vec
        embedded_docs.append(doc)
    else:
        print(f"Skipped doc {i+1} due to repeated errors.")

print(f"✅ Đã embed xong {len(embedded_docs)}/{len(docs)} documents.")

# 5️⃣ Lấy dimension và build FAISS index
if embedded_docs:
    dim = len(embedded_docs[0].embedding)
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=embedded_docs,
        storage_context=storage_context,
    )
    storage_context.persist(persist_dir="./storage3")
    print("✅ Index đã build và lưu tại ./storage3")
else:
    print("❌ Không có tài liệu nào được embed thành công.")
