# Enhanced chat router with retriever approach
import os
import re
import asyncio
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Header
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from llama_index.core import Settings 
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex 
from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.llms.groq import Groq
# from llama_index.llms.cerebras import Cerebras
# from llama_index.llms.together import TogetherLLM
# from llama_index.llms.openai import OpenAI



from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

import json
from dotenv import load_dotenv
from image_data import IMAGE_DATA, get_product_image, get_all_product_images, search_images_by_tags
load_dotenv()



chat_router = APIRouter()

# Cache and cleanup configuration
MAX_BUFFER_AGE = 3600  # 1 hour
CLEANUP_INTERVAL = 300  # 5 minutes
last_cleanup = time.time()

# System prompt cho chatbot bán hàng
SYSTEM_PROMPT = f"""
## NHIỆM VỤ CHÍNH
Bạn là nữ nhân viên CSKH chuyên nghiệp. Nhiệm vụ:
- Tư vấn sản phẩm và hướng khách hàng đến việc mua hàng một cách tự nhiên
- Ưu tiên sử dụng file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" làm kịch bản chính
- Trả lời ngắn gọn (tối đa 5 dòng), thân thiện, luôn kết thúc bằng câu hỏi tương tác
- Khi chưa rõ vấn đề, hỏi lại để làm rõ nhu cầu

## KHO ẢNH SẢN PHẨM - THÔNG TIN CHI TIẾT
{IMAGE_DATA}

## QUY TẮC HIỂN THỊ HÌNH ẢNH - BẮT BUỘC

### 1. HÌNH ẢNH ĐƠN LẺ - SỬ DỤNG MARKDOWN
**MẶC ĐỊNH:** Luôn hiển thị ảnh đơn lẻ bằng Markdown
- CHỈ sử dụng URL từ KHO ẢNH SẢN PHẨM ở trên
- Format: `![Tên sản phẩm](URL)` 
- Không được bịa link, chỉ dùng URL có sẵn trong kho ảnh
- KHÔNG được thay đổi format
- Áp dụng cho: giới thiệu sản phẩm, hướng dẫn sử dụng, feedback

**Ví dụ đúng:**
```
Dạ, đây là hình ảnh sản phẩm anh nhé:
![Bọt vệ sinh Oniiz](https://goxfkfz6v6.ufs.sh/f/MqdJQVTIGDYB10VKZUeodzDAUuLaXTc3lNqmw62rI5WVQ7iR)
Anh có muốn biết thêm gì về sản phẩm này không?
```

### 2. KHI TRẢ RA BẢNG  - CHỈ SỬ DỤNG HTML
**Khi nào dùng:** Khách hàng yêu cầu so sánh nhiều sản phẩm
- Từ khóa: "so sánh", "khác nhau gì", "bảng so sánh", "tất cả sản phẩm"
- Sử dụng <img src="url" alt="text"> cho hình ảnh trong bảng
- CHỈ dùng URL từ KHO ẢNH SẢN PHẨM

**Ví dụ đúng:**
```
Dạ vâng, em gửi anh bảng so sánh các dòng sữa tắm Oniiz:

<table>
<tr>
    <th>Sản phẩm</th>
    <th>Hình ảnh</th>
    <th>Đặc điểm</th>
</tr>
<tr>
    <td>Men In Black</td>
    <td><img src="https://goxfkfz6v6.ufs.sh/f/MqdJQVTIGDYBPz0W5yTU0tYM6uNDmBqwRp1hbSlOaKWnj52e" alt="Men In Black" style="max-width:100px;height:auto;"></td>
    <td>Trầm lắng, nam tính</td>
</tr>
</table>

Anh có muốn biết thêm thông tin gì không?
```

### 3. QUY TẮC NGHIÊM NGẶT
- **TUYỆT ĐỐI KHÔNG** tự tạo URL ảnh giả
- **CHỈ SỬ DỤNG** ảnh từ KHO ẢNH SẢN PHẨM ở trên
- **KHÔNG TRỘN LẪN** Markdown và HTML trong cùng một phản hồi
- Nếu không có ảnh phù hợp trong kho → mô tả bằng text

## THÔNG TIN SẢN PHẨM
**Sản phẩm có sẵn:**
- Bọt vệ sinh Oniiz (3 hương)
- Sữa tắm (Bel Homme, Men In Black)
- Nước hoa (Paris, Miami)
- Xịt thơm miệng (3 loại)
- Bao cao su V2Joy (4 hương)

**Quy tắc tên sản phẩm:**
- CHỈ sử dụng tên chính xác từ dữ liệu
- KHÔNG tự tạo hoặc rút gọn tên
- Kiểm tra kỹ trong KHO ẢNH SẢN PHẨM

## XỬ LÝ KHÁCH HÀNG KHÓ TÍNH
Khi khách hàng nóng nảy:
- Xoa dịu, đồng cảm
- Hỗ trợ tối đa có thể
- Nếu không giải quyết được → xin số điện thoại, hẹn nhân viên gọi lại
Here are the relevant documents for the context:
{{context_str}}
Instruction: Use the previous chat history, or the context above, to interact and help the user.
"""

# 1️⃣ Load index
# Settings.embed_model = GoogleGenAIEmbedding(
#     model_name="gemini-embedding-exp-03-07",
#     embed_batch_size=64
# )
# Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")

# Settings.llm=Groq(
#     model="llama3-70b-8192",
#     temperature=0.1,
#     max_tokens=2000,
#     request_timeout=30.0
# )

# Settings.llm = TogetherLLM(
#     model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# )
# Settings.llm = OpenAI(
#     model="gpt-4.1-mini",
# )

class ChatRequest(BaseModel):
    message: str

# Index global - chỉ khởi tạo 1 lần
index = LlamaCloudIndex(
    name="chatbot-ai-demand-2025-05-22",
    project_name="Default",
    organization_id="4e441f57-db68-4171-b335-70770a4225ec",
    api_key=os.getenv("LLAMA_CLOUD_API_KEY_DEMAND")
)

# Cache cho chat engines và buffers
buffers = {}
chat_engines = {}
buffer_timestamps = defaultdict(float)

def cleanup_old_sessions():
    """Cleanup sessions cũ hơn 1 giờ"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, timestamp in buffer_timestamps.items()
        if current_time - timestamp > MAX_BUFFER_AGE
    ]
    
    for session_id in expired_sessions:
        if session_id in buffers:
            del buffers[session_id]
        if session_id in chat_engines:
            del chat_engines[session_id]
        del buffer_timestamps[session_id]
    
    if expired_sessions:
        print(f"Cleaned up {len(expired_sessions)} expired sessions")

async def get_chat_engine(session_id: str):
    """Asynchronously get or create a chat engine for a session"""
    global last_cleanup
    current_time = time.time()

    # Perform cleanup periodically
    if current_time - last_cleanup > CLEANUP_INTERVAL:
        cleanup_old_sessions()
        last_cleanup = current_time

    if session_id not in chat_engines:
        if session_id not in buffers:
            buffers[session_id] = ChatMemoryBuffer(
                token_limit=3000,
                chat_store=SimpleChatStore(),
                chat_store_key=session_id
            )
        
        chat_engines[session_id] = index.as_chat_engine(
            chat_mode="context",
            memory=buffers[session_id],
            system_prompt=SYSTEM_PROMPT,
            streaming=True, # Enable streaming
            verbose=False
        )
    
    # Update timestamp for the current session
    buffer_timestamps[session_id] = current_time
    return chat_engines[session_id]

@chat_router.post("/chat-faiss-stream")
async def chat_stream(req: ChatRequest, session_id: str = Header(...)):
    message = req.message
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    chat_engine = await get_chat_engine(session_id)

    async def event_generator():
        try:
            # Add timeout for chat response
            response = await asyncio.wait_for(
                chat_engine.astream_chat(message),
                timeout=60.0  # 60 seconds timeout
            )
            
            # Buffer to accumulate tokens
            buffer = ""
            token_count = 0
            
            async for token in response.async_response_gen():
                try:
                    buffer += token
                    token_count += 1
                    
                    # Send data chunk as a JSON object
                    # This ensures that even if the token is a special character like a quote,
                    # it's safely encapsulated in a JSON string.
                    json_data = json.dumps({"content": token, "session_id": session_id, "success": True}, ensure_ascii=False)
                    yield f"data: {json_data}\n\n"
                    
                    # Add small delay every 10 tokens to prevent overwhelming
                    if token_count % 10 == 0:
                        await asyncio.sleep(0.01)
                        
                except Exception as token_error:
                    print(f"Error processing token: {token_error}")
                    # Continue with next token instead of breaking
                    continue

            # Send sources after streaming is complete
            sources = []
            try:
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    # Filter sources with score > 40% and limit to 5 sources
                    filtered_nodes = [node for node in response.source_nodes if getattr(node, 'score', 0.0) > 0.4]
                    # Limit to top 5 sources
                    filtered_nodes = filtered_nodes[:5]
                    
                    for i, node in enumerate(filtered_nodes):
                        try:
                            # Safely get metadata
                            metadata = getattr(node, 'metadata', {})
                            file_name = metadata.get('file_name', f"Tài liệu {i+1}")
                            
                            # Safely get node text
                            node_text = getattr(node, 'text', '')
                            if not node_text:
                                node_text = "Không có nội dung"
                            
                            source_info = {
                                "id": i + 1,
                                "title": file_name,
                                "content": node_text[:300] + "..." if len(node_text) > 300 else node_text,
                                "score": getattr(node, 'score', 0.0)
                            }
                            sources.append(source_info)
                        except Exception as source_error:
                            print(f"Error processing source {i}: {source_error}")
                            continue
                
                # Send sources as final message
                if sources:
                    sources_data = json.dumps({
                        "sources": sources, 
                        "session_id": session_id, 
                        "success": True,
                        "type": "sources"
                    }, ensure_ascii=False)
                    yield f"data: {sources_data}\n\n"
            except Exception as sources_error:
                print(f"Error processing sources: {sources_error}")
                # Continue without sources if there's an error

            print(f"[{session_id}] ===== FULL STREAMED RESPONSE =====")
            print(f"Response length: {len(buffer)}")
            print(f"Response content: {buffer}")
            print(f"Sources count: {len(sources)}")
            print(f"[{session_id}] ===== END STREAMED RESPONSE =====")

        except asyncio.TimeoutError:
            print(f"[{session_id}] Timeout error during streaming")
            error_message = {
                "content": "Xin lỗi, phản hồi mất quá nhiều thời gian. A/c vui lòng thử lại nhé.",
                "session_id": session_id,
                "success": False,
                "error": "timeout",
                "type": "error"
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
        except ConnectionError as e:
            print(f"[{session_id}] Connection error during streaming: {e}")
            error_message = {
                "content": "Xin lỗi, có lỗi kết nối. A/c vui lòng kiểm tra mạng và thử lại nhé.",
                "session_id": session_id,
                "success": False,
                "error": "connection_error",
                "type": "error"
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[{session_id}] Unexpected error during streaming: {e}")
            error_message = {
                "content": "Xin lỗi, em đang tìm dữ liệu. A/c vui lòng thử lại nhé.",
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "type": "error"
            }
            yield f"data: {json.dumps(error_message, ensure_ascii=False)}\n\n"
        finally:
            # Ensure stream ends properly
            try:
                yield f"data: {json.dumps({'type': 'stream_end', 'session_id': session_id}, ensure_ascii=False)}\n\n"
            except:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@chat_router.post("/reset-chat")
async def reset_chat(session_id: str = Header(...)):
    """Reset chat memory"""
    if session_id in buffers:
        buffers[session_id].reset()
    return {"message": "Chat history reset successfully"}
