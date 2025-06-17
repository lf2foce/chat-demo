# build_index.py
import pandas as pd
from mistralai import Client
import faiss, numpy as np
import pickle

client = Client(api_key="YOUR_MISTRAL_KEY")

# Đọc file CSV
df = pd.read_csv("notebook/university_texts.csv")
texts = df["text"].dropna().tolist()

# Tách chunks nhỏ nếu cần (ví dụ: theo câu hoặc đoạn)
chunks = texts  # đơn giản giữ nguyên

# Tạo embeddings
def get_emb(t): return client.embeddings.create(model="mistral-embed", inputs=[t]).data[0].embedding
embs = np.array([get_emb(c) for c in chunks], dtype="float32")

# Build FAISS index
d = embs.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embs)

# Lưu vào disk
faiss.write_index(index, "/data/index.faiss")
with open("/data/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Index đã build vào /data")
