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
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from dotenv import load_dotenv
load_dotenv()
import json






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


query_engine = index.as_query_engine(similarity_top_k=3)
query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="retrieval_tool",
        description="Tìm kiếm thông tin các trường đại học tại Việt Nam",
    )
)

# Chat memory để lưu lịch sử hội thoại
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=5000)

# Tạo ReActAgent với memory và hỗ trợ streaming
react_agent = ReActAgent.from_tools(
    tools=[query_tool],
    llm=Settings.llm,  # giả sử bạn đã set llm = GoogleGenAI(...)
    memory=ChatMemoryBuffer.from_defaults(token_limit=5000),
    system_prompt=SYSTEM_PROMPT, # Added system prompt
    verbose=True,
)

class ChatRequest(BaseModel):
    message: str



@chat_router.post("/chat-faiss-stream")
async def chat_faiss_react_stream(req: ChatRequest):
    """Endpoint dùng ReActAgent kèm streaming output"""
    
    # Use astream_chat for asynchronous streaming
    streaming_agent_response = await react_agent.astream_chat(req.message)

    async def event_gen():
        full_response = ""
        
        # Iterate over the async generator from the streaming response
        async for chunk_str in streaming_agent_response.async_response_gen():
            if not chunk_str:
                continue
                
            full_response += chunk_str
            
            # Stream all content without filtering
            token_data = json.dumps({"content": chunk_str}, ensure_ascii=False)
            yield f"data: {token_data}\n\n"

        completion_data = json.dumps({"done": True, "final_response_text": full_response.strip()}, ensure_ascii=False)
        yield f"data: {completion_data}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream", # Correct media type
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*" # Added CORS
        }
    )

@chat_router.post("/reset-chat")
def reset_chat():
    """Reset chat memory"""
    chat_memory.reset()
    return {"message": "Chat history reset successfully"}
