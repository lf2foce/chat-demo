# Enhanced chat router with retriever approach
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from llama_index.core import Settings 
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex 
from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
import json
from dotenv import load_dotenv
load_dotenv()



chat_router = APIRouter()

def extract_and_format_images(text: str):
    """
    Tìm và format các URL hình ảnh trong text thành markdown truyền thống ![alt](url)
    Returns: (formatted_text, has_images)
    """
    # Regex tìm URLs
    url_pattern = r'https?://[^\s\)\]]+'

    formatted_text = text
    has_images = False
    found_urls = re.findall(url_pattern, text)
    image_counter = 1

    for url in found_urls:
        # Kiểm tra nếu là URL hình ảnh từ các hosting services
        if (any(domain in url for domain in ["uploadthing", "amazonaws.com", "ufs.sh", "supabase", "cdn", "unsplash"]) or 
            re.search(r'\.(jpg|jpeg|png|gif|webp|svg)(\?|$)', url, re.IGNORECASE)):
            
            # Tạo markdown image
            image_markdown = f"\n\n![Ảnh sản phẩm {image_counter}]({url})"
            
            # Thay thế URL gốc bằng markdown image
            formatted_text = formatted_text.replace(url, image_markdown)
            has_images = True
            image_counter += 1

    return formatted_text.strip(), has_images

# System prompt cho trường đại học
SYSTEM_PROMPT = """
Bạn là nữ nhân viên CSKH. Nhiệm vụ của bạn là:
- Giải đáp các thắc mắc của khách hàng về sản phẩm và từ đó nhằm mục đích chốt sale - bán hàng. Luôn tìm cách gợi mở để hướng khách hàng tới việc chốt đơn. Nhưng không được cưỡng ép khách mua hàng nếu nhận ra khách hàng đang có tâm trạng không tốt
-  Ưu tiên sử dụng file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" để tư vấn. Cố gắng phân tích các tình huống và trả lời giống file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" nhất có thể. Các file còn lại chỉ dùng làm tư liệu tham khảo
- Trong trường hợp không tìm được câu trả lời trong file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" thì tự tạo ra câu trả lời mới dựa trên những file tài liệu có sẵn.
- Câu trả lời của bạn tự tạo ra cần ngắn gọn, xúc tích, mỗi lần trả lời không được quá 5 dòng. Sử dụng ngôn từ gần gũi, chân thành, không biểu lộ cảm xúc thái quá. Luôn kết thúc tư vấn bằng 1 câu hỏi để duy trì tương tác và hướng khách hàng tới việc mua sản phẩm
- khi chưa rõ vấn đề, hãy hỏi lại khách hàng để làm rõ nhu cầu
- Trả lời khách hàng như người quen thân mật, nhưng phải dùng từ ngữ lịch sự, lễ phép.
- Khi có hình ảnh sản phẩm trong dữ liệu, HÃY LUÔN BAO GỒM URL hình ảnh trong câu trả lời để hệ thống tự động chuyển đổi thành markdown

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

index = LlamaCloudIndex(
  name="chatbot-ai-demand-2025-05-22",
  project_name="Default",
  organization_id="4e441f57-db68-4171-b335-70770a4225ec",
  api_key=os.getenv("LLAMA_CLOUD_API_KEY_DEMAND")
)

# Chat memory để lưu lịch sử hội thoại
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Tạo chat engine với chế độ context + streaming
chat_engine = index.as_chat_engine(
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
            
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        
        # Xử lý ảnh sau khi hoàn thành streaming
        formatted_text, has_images = extract_and_format_images(full_text)
        
        # Nếu có ảnh, gửi phần markdown images được thêm vào
        if has_images:
            # Tìm phần markdown images được thêm vào (phần sau full_text gốc)
            additional_content = formatted_text[len(full_text):]
            if additional_content.strip():
                yield f"data: {json.dumps({'content': additional_content}, ensure_ascii=False)}\n\n"
        
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
