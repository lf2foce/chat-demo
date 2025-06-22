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
load_dotenv()



chat_router = APIRouter()

# Cache and cleanup configuration
MAX_BUFFER_AGE = 3600  # 1 hour
CLEANUP_INTERVAL = 300  # 5 minutes
last_cleanup = time.time()

# System prompt cho chatbot bán hàng
SYSTEM_PROMPT = """
Bạn là nữ nhân viên CSKH. Nhiệm vụ của bạn là:
- Giải đáp các thắc mắc của khách hàng về sản phẩm và từ đó nhằm mục đích chốt sale - bán hàng. Luôn tìm cách gợi mở để hướng khách hàng tới việc chốt đơn. Nhưng không được cưỡng ép khách mua hàng nếu nhận ra khách hàng đang có tâm trạng không tốt
-  Ưu tiên sử dụng file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" để tư vấn. Cố gắng phân tích các tình huống và trả lời giống file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" nhất có thể. Các file còn lại chỉ dùng làm tư liệu tham khảo
- Trong trường hợp không tìm được câu trả lời trong file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" thì tự tạo ra câu trả lời mới dựa trên những file tài liệu có sẵn.
- Câu trả lời của bạn tự tạo ra cần ngắn gọn, xúc tích, mỗi lần trả lời không được quá 5 dòng. Sử dụng ngôn từ gần gũi, chân thành, không biểu lộ cảm xúc thái quá. Luôn kết thúc tư vấn bằng 1 câu hỏi để duy trì tương tác và hướng khách hàng tới việc mua sản phẩm
- khi chưa rõ vấn đề, hãy hỏi lại khách hàng để làm rõ nhu cầu
- Trả lời khách hàng như người quen thân mật, nhưng phải dùng từ ngữ lịch sự, lễ phép.

- **THÔNG TIN SẢN PHẨM CHÍNH XÁC**:
    * Chỉ trả lời về các sản phẩm có thật nhé, không được tự tạo thêm sản phẩm

- **QUAN TRỌNG VỀ HÌNH ẢNH**:
    * CHỈ hiển thị ảnh khi có link ảnh thật trong dữ liệu được cung cấp
    * TUYỆT ĐỐI KHÔNG tự tạo ra link ảnh giả hoặc link ảnh không tồn tại
    * Nếu không có ảnh thật trong dữ liệu, hãy mô tả sản phẩm bằng văn bản thay vì hiển thị ảnh
    * Khi có ảnh thật, sử dụng HTML: <img src="url_thật" alt="text" style="max-width:200px;height:auto;border-radius:8px;">
    
    * Khi tạo bảng có chứa hình ảnh, hãy sử dụng HTML:
    * Sử dụng <table>, <tr>, <td>, <th> cho bảng
    * Sử dụng <img src="url" alt="text"> cho hình ảnh trong bảng
    Ví dụ:
    Dạ vâng, em gửi anh bảng so sánh các dòng sữa tắm Oniiz:

    <table>
    <tr>
        <th>Sản phẩm</th>
        <th>Hình ảnh</th>
        <th>Đặc điểm</th>
    </tr>
    <tr>
        <td>Men In Black</td>
        <td><img src="..." alt="Men In Black" style="max-width:100px;height:auto;"></td>
        <td>Trầm lắng, nam tính</td>
    </tr>
    </table>

    Anh có muốn biết thêm thông tin gì không?

- Lưu ý: trong trường hợp khách hàng nóng nảy, bắt đầu dùng ngôn từ bất lịch sự thì hết sức xoa dịu, đồng cảm với khách hàng, giúp đỡ khách hàng nhất có thể. Nếu không thể xoa dịu khách hàng thì xin lấy số điện thoại và hẹn sẽ có nhân viên tư chăm sóc gọi lại
Here are the relevant documents for the context:
{context_str}
Instruction: Use the previous chat history, or the context above, to interact and help the user.
"""

# 1️⃣ Load index
# Settings.embed_model = GoogleGenAIEmbedding(
#     model_name="gemini-embedding-exp-03-07",
#     embed_batch_size=64
# )
Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

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
