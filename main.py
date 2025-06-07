from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env
load_dotenv()

# Import routers sau khi đã load environment variables
# from chat_router import chat_router
from grade_router import grade_router

# FastAPI app
app = FastAPI(title="RAG Chat API", version="1.0.0")

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://chamdiem.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router với prefix riêng
# app.include_router(chat_router, tags=["Chat"])
app.include_router(grade_router, prefix="/api/py", tags=["Grade"])