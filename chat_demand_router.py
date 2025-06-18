# Enhanced chat router with retriever approach
import os
import re
from fastapi import FastAPI, HTTPException, Header
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from llama_index.core import Settings 
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex 
from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore




import json
from dotenv import load_dotenv
load_dotenv()



chat_router = APIRouter()

def extract_and_format_images(text: str):
    """
    Tìm và format các URL hình ảnh trong text thành markdown truyền thống ![alt](url)
    Nếu có nhiều ảnh, sẽ nhóm lại trong một gallery.
    Returns: (formatted_text, has_images)
    """
    # Regex to find markdown image/link patterns that contain image URLs
    # This pattern looks for ![alt](url) or [text](url) where url is an image URL
    markdown_image_pattern = r'!\\[.*?\\]\\((https?://[^\s\\)]+\.(?:jpg|jpeg|png|gif|webp|svg)(?:\?.*?)?)\\)'
    markdown_link_pattern = r'\\[.*?\\]\\((https?://[^\s\\)]+\.(?:jpg|jpeg|png|gif|webp|svg)(?:\?.*?)?)\\)'
    
    formatted_text = text
    has_images = False
    image_markups = []
    gallery_markdown = ""
    
    # Find all markdown image patterns and extract URLs
    found_image_urls = []
    for match in re.finditer(markdown_image_pattern, text, re.IGNORECASE):
        full_match = match.group(0) # The entire ![alt](url) string
        url = match.group(1) # The URL part
        if any(domain in url for domain in ["uploadthing", "amazonaws.com", "ufs.sh", "supabase", "cdn", "unsplash"]):
            found_image_urls.append((full_match, url))
            
    # Find all markdown link patterns that point to images and extract URLs
    found_link_urls = []
    for match in re.finditer(markdown_link_pattern, text, re.IGNORECASE):
        full_match = match.group(0) # The entire [text](url) string
        url = match.group(1) # The URL part
        if any(domain in url for domain in ["uploadthing", "amazonaws.com", "ufs.sh", "supabase", "cdn", "unsplash"]):
            found_link_urls.append((full_match, url))

    # Combine and remove duplicates (if a URL is both an image and a link, unlikely but safe)
    all_found_items = {} # {full_match: url}
    for full_match, url in found_image_urls + found_link_urls:
        all_found_items[full_match] = url

    # Loại bỏ các markdown image/link đã tìm thấy khỏi văn bản gốc
    for full_match in all_found_items.keys():
        formatted_text = formatted_text.replace(full_match, "").strip()

    # Tạo markdown cho từng ảnh và nhóm lại
    if all_found_items:
        has_images = True
        for i, (full_match, url) in enumerate(all_found_items.items()):
            # Use the original URL to create a new markdown image tag
            image_markups.append(f"![Ảnh sản phẩm {i+1}]({url})")
        
        gallery_markdown = "\n\n[IMAGES_START]\n" + "\n".join(image_markups) + "\n[IMAGES_END]\n\n"

    return formatted_text.strip(), gallery_markdown, has_images

# System prompt cho trường đại học
SYSTEM_PROMPT = """
Bạn là nữ nhân viên CSKH. Nhiệm vụ của bạn là:
- Giải đáp các thắc mắc của khách hàng về sản phẩm và từ đó nhằm mục đích chốt sale - bán hàng. Luôn tìm cách gợi mở để hướng khách hàng tới việc chốt đơn. Nhưng không được cưỡng ép khách mua hàng nếu nhận ra khách hàng đang có tâm trạng không tốt
-  Ưu tiên sử dụng file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" để tư vấn. Cố gắng phân tích các tình huống và trả lời giống file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" nhất có thể. Các file còn lại chỉ dùng làm tư liệu tham khảo
- Trong trường hợp không tìm được câu trả lời trong file "KỊCH BẢN CHĂM SÓC KHÁCH – TƯ VẤN BỌT VỆ SINH NAM ONIIZ TRÊN FB" thì tự tạo ra câu trả lời mới dựa trên những file tài liệu có sẵn.
- Câu trả lời của bạn tự tạo ra cần ngắn gọn, xúc tích, mỗi lần trả lời không được quá 5 dòng. Sử dụng ngôn từ gần gũi, chân thành, không biểu lộ cảm xúc thái quá. Luôn kết thúc tư vấn bằng 1 câu hỏi để duy trì tương tác và hướng khách hàng tới việc mua sản phẩm
- khi chưa rõ vấn đề, hãy hỏi lại khách hàng để làm rõ nhu cầu
- Trả lời khách hàng như người quen thân mật, nhưng phải dùng từ ngữ lịch sự, lễ phép.
- Khi có hình ảnh sản phẩm trong dữ liệu, HÃY LUÔN BAO GỒM URL hình ảnh trong câu trả lời để hệ thống tự động chuyển đổi thành markdown. (cách này có thể được thực hiện bằng cách sử dụng [markdown-to-image](https://github.com/tchiotludo/markdown-to-image)).
    - Đưa ít nhất 2 ảnh cho một nhóm sản phẩm (hoặc tất cả nếu ít hơn 5 ảnh) (ví dụ sữa tắm) (nếu có và khác nhau)
    - Khi khách chưa biết về đến sản phẩm mà hỏi thì nên trả ra cả ảnh sản phẩm nhé
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
Settings.llm = GoogleGenAI(model="gemini-2.5-flash-lite-preview-06-17")



class ChatRequest(BaseModel):
    message: str

index = LlamaCloudIndex(
  name="chatbot-ai-demand-2025-05-22",
  project_name="Default",
  organization_id="4e441f57-db68-4171-b335-70770a4225ec",
  api_key=os.getenv("LLAMA_CLOUD_API_KEY_DEMAND")
)


buffers = {} 

@chat_router.post("/chat-faiss-stream")
def chat_stream(req: ChatRequest, session_id: str = Header(...)):
    """Enhanced chat với streaming response qua LlamaIndex chat engine."""
    
    message = req.message
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")
    
    # Load hoặc tạo buffer cho phiên chat session_id
    if session_id not in buffers:
        buffers[session_id] = ChatMemoryBuffer(
            token_limit=3000,
            chat_store=SimpleChatStore(),
            chat_store_key=session_id
        )
    
    memory = buffers[session_id]
    
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        streaming=True,
        verbose=False
    )

    def event_generator():
        # Lấy kết quả streaming từ chat engine
        response = chat_engine.stream_chat(message)
        full_text = ""
        
        # Duyệt token trả về
        for tok in response.response_gen:
            chunk = str(tok)
            full_text += chunk
            
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        
        # Xử lý ảnh sau khi hoàn thành streaming
        original_text, gallery_markdown, has_images = extract_and_format_images(full_text)
        
        # Nếu có ảnh, gửi phần markdown images được thêm vào
        if has_images:
            if gallery_markdown.strip():
                yield f"data: {json.dumps({'content': gallery_markdown}, ensure_ascii=False)}\n\n"
        
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
def reset_chat(session_id: str = Header(...)):
    """Reset chat memory"""
    """Reset chat memory"""
    if session_id in buffers:
        buffers[session_id].reset()
    return {"message": "Chat history reset successfully"}
