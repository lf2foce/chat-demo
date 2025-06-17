from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env
load_dotenv()

# Import routers sau khi đã load environment variables
from chat_router import chat_router
from chat_basic_router import chat_router as chat_basic_router
# from chat_faiss_router import chat_router as chat_faiss_router
from chat_engine_cloud_router import chat_router as chat_faiss_router
from chat_demand_router import chat_router as chat_demand_router

# from chat_engine_router import chat_router as chat_faiss_router
# from chat_reactagent_router import chat_router as chat_faiss_router
from dsdaihoc_router import chat_router  as dsdaihoc_router
from grade_router import grade_router

# FastAPI app
app = FastAPI(title="RAG Chat API", version="1.0.0")

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://www.chamdiem.com",
    "https://chamdiem.com",
    "https://dsdaihoc.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router với prefix riêng
app.include_router(chat_router, tags=["Chat"])
app.include_router(chat_basic_router, tags=["Chat"])
app.include_router(chat_faiss_router, tags=["Chat"])
app.include_router(chat_faiss_router, tags=["Chat"])
app.include_router(chat_demand_router, prefix="/api/py", tags=["Chat"])
app.include_router(dsdaihoc_router, prefix="/dsdaihoc", tags=["dsdahoc"])
app.include_router(grade_router, prefix="/api/py", tags=["Grade"])
