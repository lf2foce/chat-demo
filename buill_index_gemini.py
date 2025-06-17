import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import faiss

from dotenv import load_dotenv
load_dotenv()

# 1️⃣ Global config via Settings
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-exp-03-07",
    embed_batch_size=10
)

# 2️⃣ Load docs
docs = SimpleDirectoryReader("./docs").load_data()

# 3️⃣ Lấy dimension embedding
sample_vec = Settings.embed_model.get_text_embedding("Hello world")
d = len(sample_vec)

# 4️⃣ Build FAISS index
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 5️⃣ Storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 6️⃣ Tạo và persist index
index = VectorStoreIndex.from_documents(
    documents=docs,
    storage_context=storage_context,
)
storage_context.persist(persist_dir="./storage1")
print("✅ Index built và lưu tại ./storage1")
