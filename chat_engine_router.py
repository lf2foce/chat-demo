# Enhanced chat router with retriever approach
import os
from fastapi import FastAPI, HTTPException
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from dotenv import load_dotenv
load_dotenv()






chat_router = APIRouter()

# System prompt cho trường đại học
SYSTEM_PROMPT = """
Bạn là trợ lý AI chuyên nghiệp về giáo dục và các trường đại học tại Việt Nam.
        
        NHIỆM VỤ:
        - Tư vấn thông tin về trường đại học
        - Trả lời về điểm chuẩn, ngành học, tuyển sinh
        - Thống kê và so sánh giữa các trường
        - Tìm kiếm trường phù hợp với nhu cầu
        - Hỗ trợ ra quyết định cho học sinh
        - Nếu người dùng dùng từ university có nghĩa là đại học (để tìm trong truy vấn dữ liệu nội bộ nhé), và chú ý các từ song ngữ ở Việt Nam
        - Nếu người dùng nói thông tin bị sai, hãy bảo người dùng liên hệ với chúng tôi (email: dudupython@gmail.com) để dữ liệu được cập nhật mới nhất
        - Khi bạn sử dụng thông tin từ "Dữ liệu nội bộ" được cung cấp, hãy đề cập rõ ràng rằng thông tin đó đến từ "Dữ liệu nội bộ của chúng tôi" hoặc một cách diễn đạt tương tự để người dùng biết nguồn gốc thông tin.

        LƯU Ý VỀ TÊN TRƯỜNG:
        - "FPT University" và "Đại học FPT" là cùng một trường.
        - "VinUniversity" và "Đại học VinUni" là cùng một trường.
        - "UIT" là viết tắt của "Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM".
        - "PNTU" (nếu có) cần làm rõ hoặc cung cấp tên đầy đủ.
        - Luôn ưu tiên sử dụng tên đầy đủ và chính thức của trường khi có thể, nhưng cần nhận diện được các biến thể và tên viết tắt phổ biến.
        
        KHẢ NĂNG ĐẶC BIỆT:
        - Structured queries: thống kê, so sánh, ranking
        - Text search: tìm kiếm theo từ khóa tự nhiên
        - Hybrid approach: kết hợp cả hai phương pháp
        
        PHONG CÁCH:
        - Tiếng Việt tự nhiên, thân thiện
        - Thông tin chính xác, cập nhật
        - Lời khuyên thiết thực
        - Giải thích rõ ràng, xúc tích
        
        EXAMPLES:
        - "So sánh điểm chuẩn UIT và PNTU" → Structured query
        - "Tìm trường y khoa ở TP.HCM" → Text search  
        - "Top 5 trường điểm chuẩn cao nhất" → Structured query
"""

# 1️⃣ Load index
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-exp-03-07",
    embed_batch_size=64
)
Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

vector_store = FaissVectorStore.from_persist_dir("./storage1")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir="./storage1"
)
index = load_index_from_storage(storage_context)

# Tạo retriever với similarity_top_k=3
retriever = index.as_retriever(similarity_top_k=3)

# Chat memory để lưu lịch sử hội thoại
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=5000)

# Tạo chat engine với chế độ context + streaming
chat_engine = index.as_chat_engine(
    # chat_mode="context",
    chat_mode="context",
    memory=chat_memory,
    system_prompt=SYSTEM_PROMPT,
    verbose=True,
    streaming=True
)

class ChatRequest(BaseModel):
    message: str



@chat_router.post("/chat-faiss-stream")
def chat_stream(req: ChatRequest):
    """Enhanced chat với streaming response qua LlamaIndex chat engine."""
    def event_generator():
        # Lấy kết quả streaming từ chat engine
        response = chat_engine.stream_chat(req.message)
        full_text = ""
        
        # Duyệt token trả về
        for tok in response.response_gen:
            chunk = str(tok)
            full_text += chunk
            import json
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        
        # Lưu vào memory (nếu cần)
        # memory tự động được cập nhật nếu dùng chat_engine
        
        # Kết thúc stream
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@chat_router.post("/reset-chat")
def reset_chat():
    """Reset chat memory"""
    chat_memory.reset()
    return {"message": "Chat history reset successfully"}
