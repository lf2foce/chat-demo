# from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex 

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import os

from dotenv import load_dotenv
load_dotenv()

# Cấu hình LLM
llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.llm = llm  # Thêm dòng này

# Using the LlamaDebugHandler to print the trace of the sub questions
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

try:
    # Index global - chỉ khởi tạo 1 lần
    index_dsdaihoc = LlamaCloudIndex(
            name="dsdaihoc",
            project_name="Default",
            organization_id="1bcf5fb2-bd7d-4c76-b6d5-bb793486e1b3",
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            )

    # Setup query engine
    qe = index_dsdaihoc.as_query_engine(similarity_top_k=5)

    # Setup base query engine as tool
    query_engine_tools = [
        QueryEngineTool(
            query_engine=qe,
            metadata=ToolMetadata(
                name="dsdaihoc",
                description="Danh sách các trường đại học ở việt nam",
            ),
        ),
    ]

    # Tạo SubQuestionQueryEngine để breakdown câu hỏi
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True,
    )

    # Thực hiện query và in kết quả
    print("🔍 Đang phân tích câu hỏi và tìm kiếm thông tin...\n")
    
    response = query_engine.query(
        "Học trường kinh tế nào ở Hà Nội, học phí dưới 2 triệu/tháng, ngành tài chính ngân hàng có triển vọng khi ra trường"
    )
    
    print("📋 Kết quả:")
    print("=" * 50)
    print(response)
    print("=" * 50)
    
except Exception as e:
    print(f"❌ Lỗi: {e}")