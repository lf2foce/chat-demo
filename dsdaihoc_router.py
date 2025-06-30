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

def generate_smart_suggestions(response_content: str, session_id: str = None) -> list:
    """Generate dynamic smart suggestions using fast LLM with chat history context"""
    if not response_content:
        return []
    
    try:
        # Use faster model for suggestions generation
        
        fast_llm = GoogleGenAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.3)
        
        # Get chat history if session_id is provided
        chat_history = ""
        if session_id and session_id in buffers:
            try:
                # Get recent chat messages from memory buffer
                memory_buffer = buffers[session_id]
                chat_messages = memory_buffer.get_all()
                
                # Extract last few exchanges (limit to avoid token overflow)
                recent_messages = []
                for msg in chat_messages[-6:]:  # Last 3 exchanges (user + assistant)
                    if hasattr(msg, 'content'):
                        role = "User" if msg.role.value == "user" else "AI"
                        recent_messages.append(f"{role}: {msg.content[:200]}")
                
                if recent_messages:
                    chat_history = "\n".join(recent_messages)
            except Exception as e:
                print(f"Error getting chat history: {e}")
                chat_history = ""
        
        # Let LLM analyze the full context without pre-filtering products
        
        # Enhanced prompt with chat history context
        history_context = f"\n\nLịch sử trò chuyện gần đây:\n{chat_history}" if chat_history else ""
        
        prompt = f"""Phân tích cuộc trò chuyện và tạo 4 câu hỏi thông minh (dưới 35 ký tự) để khách hàng tiếp tục hỏi:

Phản hồi mới nhất: "{response_content[:500]}"{history_context}

Sản phẩm Oniiz:
- Bọt vệ sinh: Classical (bạc hà), Perfume (nước hoa), Amorous (quyến rũ)
- Sữa tắm: Bel Homme (thanh lịch), Men In Black (nam tính)
- Nước hoa: Paris (ngọt ngào), Miami (tươi mát)
- Xịt thơm miệng: 3 loại khác nhau
- Bao cao su V2Joy: 4 hương đặc biệt

Yêu cầu:
- Tạo câu hỏi tự nhiên, phù hợp ngữ cảnh cuộc trò chuyện
- Khuyến khích khách hàng tìm hiểu sâu hơn về sản phẩm
- Hướng đến việc mua hàng một cách tự nhiên
- Mỗi câu dưới 35 ký tự, dễ hiểu

Chỉ trả về 4 câu hỏi, mỗi dòng một câu, không đánh số:
Bọt vệ sinh nào phù hợp nhất?
Giá sữa tắm Bel Homme bao nhiêu?
Nước hoa Paris có mùi như thế nào?
Có khuyến mãi gì không em?"""
        
        response = fast_llm.complete(prompt)
        
        # Parse and clean suggestions
        suggestions_text = response.text.strip()
        suggestions = []
        
        for line in suggestions_text.split('\n'):
            clean_line = line.strip('- •123456789. ').strip()
            if clean_line and len(clean_line) <= 35 and clean_line not in suggestions:
                suggestions.append(clean_line)
        
        # Enhanced fallback suggestions - natural and contextual
        if len(suggestions) < 4:
            fallback_suggestions = [
                "Sản phẩm nào phù hợp với anh?",
                "Có gì mới không em?",
                "Chất lượng thế nào vậy?",
                "Giá bao nhiêu vậy em?",
                "Có combo ưu đãi nào không?",
                "Có khuyến mãi gì không em?",
                "Sản phẩm nào phù hợp nhất?",
                "Mua ở đâu được không em?"
            ]
            
            # Add fallback suggestions up to 4 total
            for suggestion in fallback_suggestions:
                if len(suggestions) < 4 and len(suggestion) <= 35:
                    suggestions.append(suggestion)
        
        return suggestions[:4]
        
    except Exception as e:
        print(f"Error generating smart suggestions: {e}")
        # Product-focused fallbacks
        return [
            "Bọt vệ sinh có mấy hương vậy?",
            "Sữa tắm nào phù hợp với anh?", 
            "Giá cả như thế nào?",
            "Em tư vấn sản phẩm cho anh nhé"
        ]



chat_router = APIRouter()

# Cache and cleanup configuration
MAX_BUFFER_AGE = 3600  # 1 hour
CLEANUP_INTERVAL = 300  # 5 minutes
last_cleanup = time.time()

# System prompt cho chatbot bán hàng
SYSTEM_PROMPT = """
    Bạn là trợ lý AI chuyên nghiệp về giáo dục và các trường đại học tại Việt Nam.
        
    ### 1. NHIỆM VỤ:
    - Tư vấn thông tin về trường đại học
    - Trả lời về điểm chuẩn, ngành học, tuyển sinh
    - Thống kê và so sánh giữa các trường
    - Tìm kiếm trường phù hợp với nhu cầu
    - Hỗ trợ decision making cho học sinh
    
    KHẢ NĂNG ĐẶC BIỆT:
    - Structured queries: thống kê, so sánh, ranking
    - Text search: tìm kiếm theo từ khóa tự nhiên
    - Hybrid approach: kết hợp cả hai phương pháp
    
    PHONG CÁCH:
    - Tiếng Việt tự nhiên, thân thiện
    - Thông tin chính xác, cập nhật
    - Lời khuyên thiết thực
    - Giải thích rõ ràng
    
    EXAMPLES:
    - "So sánh điểm chuẩn UIT và PNTU" → Structured query
    - "Tìm trường y khoa ở TP.HCM" → Text search  
    - "Top 5 trường điểm chuẩn cao nhất" → Structured query

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
index_dsdaihoc = LlamaCloudIndex(
        name="dsdaihoc",
        project_name="Default",
        organization_id="1bcf5fb2-bd7d-4c76-b6d5-bb793486e1b3",
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        )

from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context


# chưa ổn nhé
# 2️⃣ Setup tools
qe = index_dsdaihoc.as_query_engine(similarity_top_k=5)
rag_tool = QueryEngineTool.from_defaults(
    query_engine=qe,
    name="RAG",
    description="Retrieve university info."
)

def score_program(
    user_gpa: float,
    desired_program: str,
    budget: float,
    location_pref: str,
    candidate_program: dict
) -> dict:
    """Score a candidate program for a student based on their preferences."""
    score = 0
    reasons = []
    
    if user_gpa >= candidate_program.get("cutoff_gpa", 0):
        score += 5
        reasons.append("GPA đạt yêu cầu")
    
    if candidate_program.get("tuition_per_month", 0)/1e6 <= budget:
        score += 3
        reasons.append("Trong ngân sách")
    
    if location_pref.lower() in candidate_program.get("location", "").lower():
        score += 2
        reasons.append("Gần khu vực")
    
    if desired_program.lower() in [p.lower() for p in candidate_program.get("programs", [])]:
        score += 4
        reasons.append("Có ngành bạn muốn")
    
    return {
        "program_name": candidate_program.get("name"), 
        "score": score, 
        "reasons": reasons
    }

score_tool = FunctionTool.from_defaults(
    fn=score_program,
    name="ScoreProgram",
    description="Score a candidate program for a student based on GPA, program, budget, and location preferences"
)

# 6️⃣ Tạo agent ReAct với cả 2 tools
agent = ReActAgent(
    tools=[rag_tool, score_tool],
    llm=Settings.llm,
    verbose=True
)
ctx = Context(agent)

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
        
        # chat_engines[session_id] = index_dsdaihoc.as_chat_engine(
        #     chat_mode="context",
        #     memory=buffers[session_id],
        #     system_prompt=SYSTEM_PROMPT,
        #     streaming=True, # Enable streaming
        #     verbose=False
        # )

        chat_engines[session_id] = index_dsdaihoc.as_chat_engine(
            chat_mode="react",
            tools=[rag_tool, score_tool],
            # llm=OpenAI(model="gpt-4", temperature=0),
            streaming=True
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

    def event_generator():
        try:
            # Get sync response instead of async
            response = chat_engine.stream_chat(message)
            
            # Buffer to accumulate tokens
            buffer = ""
            token_count = 0
            
            for token in response.response_gen:
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
                        time.sleep(0.01)
                        
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

            # Generate smart suggestions using LLM based on response content and chat history
            smart_suggestions = generate_smart_suggestions(buffer, session_id)
            print(f"🧠 LLM generated {len(smart_suggestions)} smart suggestions: {smart_suggestions}")
            
            if smart_suggestions:
                suggestions_data = {
                    "type": "smart_suggestions",
                    "suggestions": smart_suggestions
                }
                yield f"data: {json.dumps(suggestions_data)}\n\n"
            # end suggestion
            print(f"[{session_id}] ===== FULL STREAMED RESPONSE =====")
            print(f"Response length: {len(buffer)}")
            print(f"Response content: {buffer}")
            print(f"Sources count: {len(sources)}")
            print(f"Smart suggestions count: {len(smart_suggestions) if smart_suggestions else 0}")
            print(f"[{session_id}] ===== END STREAMED RESPONSE =====")
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
