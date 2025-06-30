from llama_index.llms.google_genai import GoogleGenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step, Context
from typing import List, Dict, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ===== EVENTS DEFINITION =====
class SearchEvent(Event):
    """Event chứa kết quả tìm kiếm"""
    query: str
    results: List[str]
    success: bool

# ===== OPTIMIZED RAG WORKFLOW =====
class OptimizedRAGWorkflow(Workflow):
    """Tối ưu hóa RAG Workflow để tránh timeout"""
    
    def __init__(self, query_engine_tool: QueryEngineTool):
        super().__init__(timeout=30.0)  # Giảm timeout
        self.query_engine_tool = query_engine_tool
        self.llm = Settings.llm
    
    @step
    async def search_and_synthesize(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Tìm kiếm và tổng hợp trực tiếp để tránh timeout"""
        query = ev.query
        start_time = datetime.now()
        
        print(f"🔍 Tìm kiếm: {query}")
        
        try:
            # Tìm kiếm trực tiếp
            response = await self.query_engine_tool.query_engine.aquery(query)
            print(response)
            # Tổng hợp ngắn gọn
            synthesis_prompt = f"""
            Dựa trên thông tin sau, hãy trả lời  câu hỏi: {query}
            
            Thông tin: {str(response)[:1000]}
            
            Trả lời theo format:
            🎯 **Trả lời :**
            [Câu trả lời chính]
            
            🏫 **Trường phù hợp:**
            [Danh sách trường]
            
            💰 **Học phí:**
            [Thông tin học phí]
            """
            
            final_response = await self.llm.acomplete(synthesis_prompt)
            
            total_time = (datetime.now() - start_time).total_seconds()
            result = f"{str(final_response)}\n\n⏱️ Thời gian: {total_time:.1f}s"
            
            print(f"✅ Hoàn thành trong {total_time:.1f}s")
            return StopEvent(result=result)
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return StopEvent(result=f"Lỗi xử lý: {str(e)}")
    


# ===== MAIN EXECUTION =====
async def main():
    """Hàm chính để chạy Optimized RAG Workflow"""
    
    # Cấu hình LLM
    llm = GoogleGenAI(model="gemini-2.0-flash")
    Settings.llm = llm
    
    try:
        # Khởi tạo index
        print("🚀 Khởi tạo LlamaCloud Index...")
        index_dsdaihoc = LlamaCloudIndex(
            name="dsdaihoc",
            project_name="Default",
            organization_id="1bcf5fb2-bd7d-4c76-b6d5-bb793486e1b3",
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        )
        
        # Tạo query engine tool
        qe = index_dsdaihoc.as_query_engine(similarity_top_k=3)  # Giảm số lượng
        query_engine_tool = QueryEngineTool(
            query_engine=qe,
            metadata=ToolMetadata(
                name="dsdaihoc",
                description="Danh sách các trường đại học ở việt nam",
            ),
        )
        
        # Khởi tạo Optimized RAG Workflow
        print("🤖 Khởi tạo Optimized RAG Workflow...")
        workflow = OptimizedRAGWorkflow(query_engine_tool)
        
        # Chạy workflow
        print("\n🎯 BẮT ĐẦU WORKFLOW")
        
        query = "Học trường kinh tế nào ở Cầu Giấy, học phí dưới 2 triệu/tháng, ngành tài chính ngân hàng và có triển vọng khi ra trường"
        
        result = await workflow.run(query=query)
        
        print("\n📋 KẾT QUẢ:")
        print(result)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())