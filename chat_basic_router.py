# worked
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi import APIRouter


# Tải biến môi trường từ .env
load_dotenv()


chat_router = APIRouter()

# Định nghĩa lớp dữ liệu đầu vào
class ChatInput(BaseModel):
    message: str

# Khởi tạo OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@chat_router.post("/chat-basic")
async def chat_endpoint(input: ChatInput):
    def generate():
        print(f"Input message: {input.message}")
        
        # Sử dụng chat completions API đúng cách
        stream = client.chat.completions.create(
            model="gpt-4o-mini",  # Sử dụng model name chính xác
            messages=[
                {"role": "user", "content": input.message}
            ],
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    return StreamingResponse(generate(), media_type="text/plain")
