# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# 1️⃣ Load index
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-exp-03-07",
    embed_batch_size=64
)
Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir="./storage"
)
index = load_index_from_storage(storage_context)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query(req: QueryRequest):
    resp = index.as_query_engine().query(req.query)
    return {"response": str(resp)}
