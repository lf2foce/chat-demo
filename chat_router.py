# stream data tới html
import os
import re
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LlamaIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.selectors import LLMSingleSelector

# from llama_index.llms.deepseek import DeepSeek
# from llama_index.llms.groq import Groq
# from llama_index.llms.cerebras import Cerebras
# from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

from my_module import setup_pinecone_vector_index

# Embed model setup
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-exp-03-07",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Chat input model
class ChatInput(BaseModel):
    message: str


def extract_and_format_images(text: str):
    """
    Tìm và format các URL hình ảnh trong text dạng (https://...) thành [IMAGE:...]
    Ẩn link gốc và thay bằng placeholder ảnh cho frontend xử lý.
    Returns: (formatted_text, has_images)
    """
    # Regex tìm (https://...) trong ngoặc đơn hoặc ngoài
    url_pattern = r'\(?(https?://[^\s\)]+)\)?'

    formatted_text = text
    has_images = False
    found_urls = re.findall(url_pattern, text)

    for url in found_urls:
        # Nếu chưa phải ảnh, nhưng từ các dịch vụ hosting (UploadThing, S3, Cloudinary…)
        if any(domain in url for domain in ["uploadthing", "amazonaws.com", "ufs.sh", "supabase", "cdn", "unsplash"]):
            image_tag = f"[IMAGE:{url}]"
            # Replace đoạn "(https://...)" hoặc " https://... "
            formatted_text = formatted_text.replace(f"({url})", "")  # xóa trong ngoặc
            formatted_text = formatted_text.replace(url, "")         # xóa nếu không ngoặc
            formatted_text += f"\n\n{image_tag}"
            has_images = True
        elif re.search(r'\.(jpg|jpeg|png|gif|webp|svg)(\?|$)', url, re.IGNORECASE):
            image_tag = f"[IMAGE:{url}]"
            formatted_text = formatted_text.replace(f"({url})", "")
            formatted_text = formatted_text.replace(url, "")
            formatted_text += f"\n\n{image_tag}"
            has_images = True

    return formatted_text.strip(), has_images

# Context prompt builder
def build_full_prompt(system_prompt, memory, message):
    context = "\n".join(f"{msg.role.value}: {msg.content}" for msg in memory.get_all()[-6:])
    return f"""{system_prompt}

LỊCH SỬ TRÒ CHUYỆN:
{context}

CÂU HỎI HIỆN TẠI: {message}

Hãy trả lời câu hỏi dựa trên ngữ cảnh cuộc trò chuyện và thông tin từ cơ sở dữ liệu.
""" if context else f"{system_prompt}\n\nCÂU HỎI: {message}"

# Multi index engine
class MultiIndexChatEngine:
    def __init__(self, indices_config, llm):
        self.llm = llm
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        tools = []

        for config in indices_config:
            query_engine = config['index'].as_query_engine(
                llm=llm,
                similarity_top_k=3  # Limit to top 3 most similar results
            )
            tools.append(QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(name=config['name'], description=config['description'])
            ))

        self.router = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(llm=llm),
            query_engine_tools=tools,
            verbose=True
        )

        self.system_prompt = """
        Bạn là trợ lý AI chuyên nghiệp về sản phẩm của CÔNG TY CỔ PHẦN CÔNG NGHỆ SINH HỌC HOÀ BÌNH 
        
        NHIỆM VỤ:
        - Tư vấn sản phẩm một cách chi tiết và hữu ích
        - Trả lời các câu hỏi về thành phần, công dụng, cách sử dụng
        - Cung cấp thông tin về giá cả và nơi mua nếu có
        - Duy trì cuộc trò chuyện tự nhiên và thân thiện
        - Khi có hình ảnh sản phẩm trong dữ liệu, HÃY LUÔN BAO GỒM trong câu trả lời
        - Định dạng hình ảnh theo chuẩn Ảnh:URL ở cuối câu trả lời
        
        QUY TẮC VỀ HÌNH ẢNH:
        - Luôn kiểm tra dữ liệu có chứa hình ảnh sản phẩm không
        - Nếu có hình ảnh phù hợp với câu hỏi, hãy thêm vào câu trả lời
        - Mỗi hình ảnh phải có mô tả ngắn gọn trước khi hiển thị
        - Giới hạn tối đa 3 hình ảnh mỗi câu trả lời
        
        PHONG CÁCH:
        - Sử dụng tiếng Việt tự nhiên, không cứng nhắc
        - Đưa ra lời khuyên cụ thể và thực tế
        - Hỏi lại nếu cần làm rõ yêu cầu của khách hàng
        """

    def chat_stream(self, message: str):
        prompt = build_full_prompt(self.system_prompt, self.memory, message)
        response = str(self.router.query(prompt))
        formatted, _ = extract_and_format_images(response)

        self.memory.put(ChatMessage(role=MessageRole.USER, content=message))
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response))

        for i in range(0, len(formatted), 5):
            yield formatted[i:i+5]
            time.sleep(0.02)

    def reset(self):
        self.memory.reset()

# RAG setup
def setup_rag_system():
    llm = GoogleGenAI(
        model="gemini-2.0-flash",
    )

    # llm = DeepSeek(model="deepseek-chat")
    
    # llm=Groq(
    #     # model="meta-llama/llama-4-maverick-17b-128e-instruct", # k ổn
    #     model="llama-3.3-70b-versatile",
    #     # model="llama3-70b-8192", # ổn nhất
    #     temperature=0.1,
    #     # max_tokens=4000
    # )

    # llm = Cerebras(
    #     # model="llama-3.1-8b", 
    #     model="llama-3.3-70b",  
    #     # model="qwen-3-32b", # think
    #     # model="llama-4-scout-17b-16e-instruct",  # không ổn
    #     api_key=os.environ["CEREBRAS_API_KEY"])



    # index_oniz = LlamaCloudIndex(
    #     name="subsequent-stork-2025-05-22",
    #     project_name="Default", 
    #     organization_id="1bcf5fb2-bd7d-4c76-b6d5-bb793486e1b3",
    #     api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    # )

    # index_excel = LlamaCloudIndex(
    #     name="funny-clownfish-2025-05-28",
    #     project_name="Default", 
    #     organization_id="1bcf5fb2-bd7d-4c76-b6d5-bb793486e1b3",
    #     api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    # )

    index_pinecone = setup_pinecone_vector_index()
    
    indices_config = [
        {
            'name': 'Thông tin về CÔNG TY CỔ PHẦN CÔNG NGHỆ SINH HỌC HOÀ BÌNH',
            'index': index_pinecone,
            'description': "CÔNG TY CỔ PHẦN CÔNG NGHỆ SINH HỌC HOÀ BÌNH"
        }
    ]

    return MultiIndexChatEngine(indices_config, llm)

# Khởi tạo chat engine
chat_engine = setup_rag_system()

# Tạo router - không đặt prefix ở đây, sẽ đặt trong main_test.py
chat_router = APIRouter()

@chat_router.post("/chat")
async def chat_endpoint(input: ChatInput):
    def generate():
        try:
            yield from chat_engine.chat_stream(input.message)
        except Exception as e:
            yield f"❌ Lỗi: {str(e)}"
    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

@chat_router.post("/reset")
async def reset_chat():
    try:
        chat_engine.reset()
        return {"message": "Đã reset lịch sử trò chuyện thành công"}
    except Exception as e:
        return {"error": str(e)}

@chat_router.get("/")
async def root():
    return {
        "message": "🤖 Oniiz RAG Chat Server đang hoạt động",
        "features": ["Image Support", "Streaming Chat", "Memory Management"],
        "endpoints": {
            "/chat": "POST - Chat với streaming response",
            "/reset": "POST - Reset lịch sử",
            "/health": "GET - Kiểm tra trạng thái"
        }
    }

@chat_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": "pinecone",
        "features": ["streaming", "memory", "images"],
    }