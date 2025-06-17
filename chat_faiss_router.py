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
from llama_index.embeddings.openai import OpenAIEmbedding

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
# Settings.embed_model = GoogleGenAIEmbedding(
#     model_name="gemini-embedding-exp-03-07",
#     embed_batch_size=64
# )

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

vector_store = FaissVectorStore.from_persist_dir("./storage/csv-openai-small")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir="./storage/csv-openai-small"
)
index = load_index_from_storage(storage_context)

# Tạo retriever với similarity_top_k=3
retriever = index.as_retriever(similarity_top_k=3)

# Chat memory để lưu lịch sử hội thoại
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

class QueryRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    message: str

@chat_router.post("/chat-faiss")
def query(req: QueryRequest):
    """Endpoint cũ để tương thích ngược"""
    resp = index.as_query_engine().query(req.query)
    return {"response": str(resp)}

@chat_router.post("/chat-faiss-enhanced")
def chat_enhanced(req: ChatRequest):
    """Enhanced chat với retriever approach"""
    try:
        # Lấy context từ retriever
        retrieved_nodes = retriever.retrieve(req.message)
        
        retrieved_data_context = ""
        if retrieved_nodes and len(retrieved_nodes) > 0:
            retrieved_data_context = "\n\n".join([
                node.node.text for node in retrieved_nodes
            ])[:5000]  # Giới hạn 5000 ký tự
            
            print(f"Retrieved data length: {len(retrieved_data_context)}")
        else:
            print("No data found")
            retrieved_data_context = "Không tìm thấy thông tin cụ thể về câu hỏi này trong tài liệu."
            
    except Exception as error:
        print(f"Error retrieving data: {error}")
        retrieved_data_context = "Đã xảy ra lỗi khi tìm kiếm thông tin. Tôi sẽ trả lời dựa trên kiến thức chung."
    
    # Tạo enhanced system prompt
    enhanced_system_prompt = f"""{SYSTEM_PROMPT}

Dữ liệu nội bộ:
{retrieved_data_context}

Sử dụng thông tin trong "Kho Dữ Liệu" (Dữ liệu của chúng tôi) để trả lời câu hỏi của người dùng. Nếu thông tin không có trong "Kho Dữ Liệu", hãy trả lời dựa trên kiến thức chung của bạn về các trường Đại Học tại Việt Nam và nói rõ là bạn đang dùng kiến thức chung."""
    
    # Tạo messages với enhanced system prompt
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=enhanced_system_prompt),
        ChatMessage(role=MessageRole.USER, content=req.message)
    ]
    
    # Thêm lịch sử chat nếu có
    chat_history = chat_memory.get_all()
    if chat_history:
        # Chèn lịch sử giữa system prompt và user message hiện tại
        messages = [messages[0]] + chat_history + [messages[1]]
    
    # Gọi LLM
    llm = Settings.llm
    response = llm.chat(messages)
    
    # Lưu vào memory
    chat_memory.put(ChatMessage(role=MessageRole.USER, content=req.message))
    chat_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=str(response)))
    
    return {
        "response": str(response),
        "context_length": len(retrieved_data_context),
        "nodes_found": len(retrieved_nodes) if retrieved_nodes else 0
    }

@chat_router.post("/chat-faiss-stream")
def chat_stream(req: ChatRequest):
    """Enhanced chat với streaming response"""
    def generate_response():
        try:
            # Lấy context từ retriever
            retrieved_nodes = retriever.retrieve(req.message)
            
            retrieved_data_context = ""
            if retrieved_nodes and len(retrieved_nodes) > 0:
                retrieved_data_context = "\n\n".join([
                    node.node.text for node in retrieved_nodes
                ])[:5000]
                
                print(f"Retrieved context length: {len(retrieved_data_context)}")
            else:
                print("No context found")
                retrieved_data_context = "Không tìm thấy thông tin cụ thể về câu hỏi này trong tài liệu."
                
        except Exception as error:
            print(f"Error retrieving context: {error}")
            retrieved_data_context = "Đã xảy ra lỗi khi tìm kiếm thông tin. Tôi sẽ trả lời dựa trên kiến thức chung."
        
        # Tạo enhanced system prompt - loại bỏ từ "data" và cải thiện prompt
        enhanced_system_prompt = f"""{SYSTEM_PROMPT}

Thông tin tham khảo:
{retrieved_data_context}

Hãy sử dụng thông tin tham khảo ở trên để trả lời câu hỏi của người dùng một cách tự nhiên và thân thiện. Nếu thông tin không đủ, hãy bổ sung từ kiến thức chung về giáo dục đại học Việt Nam."""
        
        # Tạo messages
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=enhanced_system_prompt),
            ChatMessage(role=MessageRole.USER, content=req.message)
        ]
        
        # Thêm lịch sử chat
        chat_history = chat_memory.get_all()
        if chat_history:
            messages = [messages[0]] + chat_history + [messages[1]]
        
        # Stream response
        llm = Settings.llm
        response_stream = llm.stream_chat(messages)
        
        full_response = ""
        for token in response_stream:
            chunk = str(token.delta)
            full_response += chunk
            # Cải thiện format - loại bỏ "data:" prefix và sử dụng JSON format
            import json
            chunk_data = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"data: {chunk_data}\n\n"
        
        # Lưu vào memory
        chat_memory.put(ChatMessage(role=MessageRole.USER, content=req.message))
        chat_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=full_response))
        
        # Signal completion
        completion_data = json.dumps({"done": True}, ensure_ascii=False)
        yield f"data: {completion_data}\n\n"
    
    return StreamingResponse(
        generate_response(),
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
