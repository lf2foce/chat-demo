# Enhanced chat router with retriever approach - Optimized for 2025
import os
import re
import json
import asyncio
from functools import lru_cache
from typing import Dict, Tuple, Optional, Generator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llama_index.core import Settings 
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex 
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

from dotenv import load_dotenv
load_dotenv()


# Constants - Optimized for performance
TRUSTED_IMAGE_DOMAINS = frozenset(["uploadthing", "amazonaws.com", "ufs.sh", "supabase", "cdn", "unsplash"])
IMAGE_EXTENSIONS_PATTERN = r'\.(?:jpg|jpeg|png|gif|webp|svg|avif|bmp|tiff)'

# Pre-compiled regex patterns for better performance
MARKDOWN_IMAGE_PATTERN = re.compile(rf'!\[(.*?)\]\((https?://[^\s\)]+{IMAGE_EXTENSIONS_PATTERN}(?:\?[^\s\)]*)?)\)', re.IGNORECASE)
MARKDOWN_LINK_TO_IMAGE_PATTERN = re.compile(rf'\[(.*?)\]\((https?://[^\s\)]+{IMAGE_EXTENSIONS_PATTERN}(?:\?[^\s\)]*)?)\)', re.IGNORECASE)

# Cache for image processing results
@lru_cache(maxsize=1000)
def _cached_domain_check(url: str) -> bool:
    """Cached domain validation for better performance"""
    return any(domain in url for domain in TRUSTED_IMAGE_DOMAINS)




chat_router = APIRouter()

@lru_cache(maxsize=500)
def extract_and_format_images(text: str) -> Tuple[str, str, bool]:
    """
    Optimized image extraction with caching and early returns.
    Returns: (formatted_text_without_images, gallery_markdown, has_images)
    """
    if not text:
        return "", "", False

    # Fast pre-check using optimized string operations
    text_lower = text.lower()
    if not ("![" in text or "[" in text) or "http" not in text:
        return text, "", False
    
    # Quick extension check with set intersection
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.avif', '.bmp', '.tiff'}
    if not any(ext in text_lower for ext in extensions):
        return text, "", False

    found_items_by_index = {}
    
    # Use pre-compiled regex patterns
    for match in MARKDOWN_IMAGE_PATTERN.finditer(text):
        url = match.group(2)
        if _cached_domain_check(url):
            alt_text = match.group(1) or "Ảnh sản phẩm"
            found_items_by_index[match.start()] = (match.group(0), alt_text, url)
            
    for match in MARKDOWN_LINK_TO_IMAGE_PATTERN.finditer(text):
        url = match.group(2)
        if _cached_domain_check(url) and match.start() not in found_items_by_index:
            found_items_by_index[match.start()] = (match.group(0), "Hình ảnh liên kết", url)

    if not found_items_by_index:
        return text, "", False

    # Optimized text reconstruction
    sorted_matches = sorted(found_items_by_index.items())
    parts = []
    image_markups = []
    current_pos = 0
    
    for start_idx, (full_match, alt, url) in sorted_matches:
        parts.append(text[current_pos:start_idx])
        image_markups.append(f"![{alt}]({url})")
        current_pos = start_idx + len(full_match)
    
    parts.append(text[current_pos:])
    processed_text = "".join(parts).strip()
    gallery_markdown = f"\n\n[IMAGES_START]\n{chr(10).join(image_markups)}\n[IMAGES_END]\n\n"
    
    return processed_text, gallery_markdown, True

# System prompt cho trường đại học
SYSTEM_PROMPT = """
Bạn là nữ nhân viên CSKH. Nhiệm vụ của bạn là:
- Giải đáp các thắc mắc của khách hàng về sản phẩm và từ đó nhằm mục đích chốt sale - bán hàng. Luôn tìm cách gợi mở để hướng khách hàng tới việc chốt đơn. Nhưng không được cưỡng ép khách mua hàng nếu nhận ra khách hàng đang có tâm trạng không tốt
-  Ưu tiên sử dụng file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" để tư vấn. Cố gắng phân tích các tình huống và trả lời giống file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" nhất có thể. Các file còn lại chỉ dùng làm tư liệu tham khảo
- Trong trường hợp không tìm được câu trả lời trong file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" thì tự tạo ra câu trả lời mới dựa trên những file tài liệu có sẵn.
- Câu trả lời của bạn tự tạo ra cần ngắn gọn, xúc tích, mỗi lần trả lời không được quá 5 dòng. Sử dụng ngôn từ gần gũi, chân thành, không biểu lộ cảm xúc thái quá. Luôn kết thúc tư vấn bằng 1 câu hỏi để duy trì tương tác và hướng khách hàng tới việc mua sản phẩm
- khi chưa rõ vấn đề, hãy hỏi lại khách hàng để làm rõ nhu cầu
- Trả lời khách hàng như người quen thân mật, nhưng phải dùng từ ngữ lịch sự, lễ phép.
**QUAN TRỌNG - HIỂN THỊ HÌNH ẢNH:**
- BẮT BUỘC: Khi có bất kỳ thông tin nào về sản phẩm, LUÔN LUÔN bao gồm URL hình ảnh trong câu trả lời. Đây là yêu cầu tuyệt đối, không được bỏ qua.
- Hãy tích cực tìm kiếm và đưa ra hình ảnh sản phẩm trong mọi câu trả lời có liên quan đến sản phẩm.
- Đưa ít nhất 2-3 ảnh cho mỗi nhóm sản phẩm (hoặc tất cả nếu ít hơn 5 ảnh) để khách hàng có thể nhìn thấy sản phẩm rõ ràng.
- Khi khách hàng hỏi về sản phẩm lần đầu, LUÔN kèm theo hình ảnh để tạo ấn tượng tốt.
- Format URL hình ảnh: Sử dụng định dạng markdown ![Mô tả](URL) để hệ thống tự động hiển thị ảnh trong UI.
- Ví dụ: ![Bọt vệ sinh Oniiz](https://example.com/image.jpg)

- Lưu ý: trong trường hợp khách hàng nóng nảy, bắt đầu dùng ngôn từ bất lịch sự thì hết sức xoa dịu, đồng cảm với khách hàng, giúp đỡ khách hàng nhất có thể. Nếu không thể xoa dịu khách hàng thì xin lấy số điện thoại và hẹn sẽ có nhân viên tư chăm sóc gọi lại
Here are the relevant documents for the context:
{context_str}
Instruction: Use the previous chat history, or the context above, to interact and help the user.
"""

# Optimized configuration and models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")

class ChatResponse(BaseModel):
    content: Optional[str] = None
    done: bool = False

# Lazy initialization for better startup performance
@lru_cache(maxsize=1)
def get_llm_settings():
    """Lazy initialization of LLM settings"""
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=2048
    )
    return Settings.llm

@lru_cache(maxsize=1)
def get_index():
    """Lazy initialization of LlamaCloud index"""
    return LlamaCloudIndex(
        name="chatbot-ai-demand-2025-05-22",
        project_name="Default",
        organization_id="4e441f57-db68-4171-b335-70770a4225ec",
        api_key=os.getenv("LLAMA_CLOUD_API_KEY_DEMAND")
    )

# Memory management with TTL and size limits
buffers: Dict[str, ChatMemoryBuffer] = {}
MAX_BUFFERS = 1000
BUFFER_TTL = 3600  # 1 hour 

def _cleanup_old_buffers():
    """Clean up old buffers to prevent memory leaks"""
    if len(buffers) > MAX_BUFFERS:
        # Remove oldest 20% of buffers
        to_remove = len(buffers) - int(MAX_BUFFERS * 0.8)
        for session_id in list(buffers.keys())[:to_remove]:
            del buffers[session_id]

def _get_or_create_memory(session_id: str) -> ChatMemoryBuffer:
    """Get or create memory buffer with cleanup"""
    if session_id not in buffers:
        _cleanup_old_buffers()
        buffers[session_id] = ChatMemoryBuffer(
            token_limit=3000,
            chat_store=SimpleChatStore(),
            chat_store_key=session_id
        )
    return buffers[session_id]

@chat_router.post("/chat-faiss-stream")
async def chat_stream(req: ChatRequest, session_id: str = Header(...)):
    """Optimized streaming chat with async support and better error handling"""
    
    try:
        # Initialize components lazily
        get_llm_settings()
        index = get_index()
        memory = _get_or_create_memory(session_id)
        
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=SYSTEM_PROMPT,
            streaming=True,
            verbose=False
        )

        async def event_generator() -> Generator[str, None, None]:
            try:
                # Stream response with optimized chunking
                response = chat_engine.stream_chat(req.message)
                full_text = ""
                chunk_buffer = ""
                
                for tok in response.response_gen:
                    chunk = str(tok)
                    full_text += chunk
                    chunk_buffer += chunk
                    
                    # Send chunks in batches for better performance
                    if len(chunk_buffer) >= 50:  # Batch size optimization
                        yield f"data: {json.dumps({'content': chunk_buffer}, ensure_ascii=False)}\n\n"
                        chunk_buffer = ""
                
                # Send remaining buffer
                if chunk_buffer:
                    yield f"data: {json.dumps({'content': chunk_buffer}, ensure_ascii=False)}\n\n"
                
                # Process images efficiently
                if full_text:
                    _, gallery_markdown, has_images = extract_and_format_images(full_text)
                    if has_images and gallery_markdown.strip():
                        yield f"data: {json.dumps({'content': gallery_markdown}, ensure_ascii=False)}\n\n"
                
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@chat_router.post("/reset-chat")
async def reset_chat(session_id: str = Header(...)):
    """Reset chat memory with improved error handling"""
    try:
        if session_id in buffers:
            buffers[session_id].reset()
            return {"message": "Chat history reset successfully", "session_id": session_id}
        else:
            return {"message": "No chat history found for this session", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")

@chat_router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "active_sessions": len(buffers),
        "max_buffers": MAX_BUFFERS
    }
