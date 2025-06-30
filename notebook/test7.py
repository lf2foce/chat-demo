from llama_index.llms.google_genai import GoogleGenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.storage.cache import SimpleCache

import os
import logging
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class University:
    name: str
    location: str
    fee_per_month: float
    programs: List[str]
    ranking: Optional[int] = None

class OptimizedAgenticRAG:
    def __init__(self):
        self.setup_llm()
        self.setup_debugging()
        self.setup_cache()
        self.setup_index()
        self.setup_tools()
        self.setup_agent()
    
    def setup_llm(self):
        """Cấu hình LLM với các tham số tối ưu"""
        self.llm = GoogleGenAI(
            model="gemini-2.0-flash",
            temperature=0.1,  # Giảm randomness cho kết quả ổn định
            max_tokens=4096
        )
        Settings.llm = self.llm
        logger.info("✅ LLM configured successfully")
    
    def setup_debugging(self):
        """Setup debug và callback manager"""
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        self.callback_manager = CallbackManager([llama_debug])
        Settings.callback_manager = self.callback_manager
        logger.info("✅ Debug handler configured")
    
    def setup_cache(self):
        """Setup caching để tối ưu performance"""
        cache = SimpleCache()
        Settings.cache = cache
        logger.info("✅ Cache configured")
    
    def setup_index(self):
        """Khởi tạo LlamaCloud index với error handling"""
        try:
            self.index = LlamaCloudIndex(
                name="dsdaihoc",
                project_name="Default", 
                organization_id="1bcf5fb2-bd7d-4c76-b6d5-bb793486e1b3",
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            )
            logger.info("✅ LlamaCloud index initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize index: {e}")
            raise
    
    def create_advanced_query_engine(self):
        """Tạo query engine với retrieval strategy nâng cao"""
        # Retriever với top_k cao hơn
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
        
        # Postprocessor để lọc kết quả có độ tương đồng thấp
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        
        # Response synthesizer tối ưu
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            use_async=True
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            response_synthesizer=response_synthesizer
        )
    
    def search_by_criteria(self, location: str = None, max_fee: float = None, 
                          program: str = None) -> str:
        """Tool tìm kiếm theo tiêu chí cụ thể"""
        query_parts = []
        if location:
            query_parts.append(f"trường đại học ở {location}")
        if max_fee:
            query_parts.append(f"học phí dưới {max_fee} triệu")
        if program:
            query_parts.append(f"ngành {program}")
        
        query = " ".join(query_parts)
        logger.info(f"🔍 Searching with criteria: {query}")
        
        try:
            qe = self.create_advanced_query_engine()
            response = qe.query(query)
            return str(response)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Không thể tìm kiếm với tiêu chí: {query}"
    
    def analyze_career_prospects(self, program: str) -> str:
        """Tool phân tích triển vọng nghề nghiệp"""
        query = f"triển vọng nghề nghiệp ngành {program} tại Việt Nam"
        logger.info(f"📊 Analyzing career prospects for: {program}")
        
        try:
            qe = self.create_advanced_query_engine()
            response = qe.query(query)
            return str(response)
        except Exception as e:
            logger.error(f"Career analysis error: {e}")
            return f"Không thể phân tích triển vọng cho ngành: {program}"
    
    def compare_universities(self, university_names: List[str]) -> str:
        """Tool so sánh các trường đại học"""
        query = f"so sánh các trường đại học: {', '.join(university_names)}"
        logger.info(f"⚖️ Comparing universities: {university_names}")
        
        try:
            qe = self.create_advanced_query_engine()
            response = qe.query(query)
            return str(response)
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return f"Không thể so sánh các trường: {', '.join(university_names)}"
    
    def setup_tools(self):
        """Setup các tools chuyên biệt cho agent"""
        # Tool chính cho query tổng quát
        main_qe = self.create_advanced_query_engine()
        main_tool = QueryEngineTool(
            query_engine=main_qe,
            metadata=ToolMetadata(
                name="university_search",
                description="Tìm kiếm thông tin tổng quát về các trường đại học Việt Nam"
            ),
        )
        
        # Tool tìm kiếm theo tiêu chí
        criteria_tool = FunctionTool.from_defaults(
            fn=self.search_by_criteria,
            name="search_by_criteria",
            description="Tìm kiếm trường đại học theo tiêu chí cụ thể (địa điểm, học phí, ngành học)"
        )
        
        # Tool phân tích triển vọng
        career_tool = FunctionTool.from_defaults(
            fn=self.analyze_career_prospects,
            name="career_analysis", 
            description="Phân tích triển vọng nghề nghiệp của một ngành học"
        )
        
        # Tool so sánh trường
        compare_tool = FunctionTool.from_defaults(
            fn=self.compare_universities,
            name="university_comparison",
            description="So sánh nhiều trường đại học với nhau"
        )
        
        self.tools = [main_tool, criteria_tool, career_tool, compare_tool]
        logger.info(f"✅ {len(self.tools)} tools configured")
    
    def setup_agent(self):
        """Setup ReAct agent với memory"""
        # Memory để nhớ context cuộc hội thoại
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        # Tạo agent
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            memory=memory,
            verbose=True,
            max_iterations=10,
            system_prompt="""
            Bạn là một AI consultant chuyên tư vấn giáo dục đại học tại Việt Nam.
            
            Nhiệm vụ của bạn:
            1. Phân tích câu hỏi của user một cách chi tiết
            2. Sử dụng các tools phù hợp để tìm kiếm thông tin
            3. Tổng hợp và đưa ra lời khuyên cụ thể, thực tế
            4. Luôn cung cấp thông tin chính xác và cập nhật
            
            Hãy trả lời một cách thân thiện, chuyên nghiệp và hữu ích.
            """
        )
        logger.info("✅ ReAct agent configured with memory")
    
    async def query(self, question: str) -> Optional[str]:
        """Thực hiện query với error handling"""
        try:
            logger.info(f"🤖 Processing question: {question}")
            response = await self.agent.achat(question)
            return str(response)
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"
    
    def chat(self, question: str) -> Optional[str]:
        """Sync version của query method"""
        try:
            logger.info(f"🤖 Processing question: {question}")
            response = self.agent.chat(question)
            return str(response)
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

# Usage example
async def main():
    """Main function để test agent"""
    try:
        # Khởi tạo agent
        print("🚀 Initializing Optimized Agentic RAG...")
        rag_agent = OptimizedAgenticRAG()
        
        # Test query
        question = "Học trường kinh tế nào ở Hà Nội, học phí dưới 2 triệu/tháng, ngành tài chính ngân hàng có triển vọng khi ra trường"
        
        print("\n🔍 Đang phân tích câu hỏi và tìm kiếm thông tin...")
        print("=" * 70)
        
        # Sử dụng async version
        response = await rag_agent.query(question)
        
        print("\n📋 Kết quả từ Agentic RAG:")
        print("=" * 70)
        print(response)
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

# Sync version cho testing
def main_sync():
    """Sync version of main for easier testing"""
    try:
        print("🚀 Initializing Optimized Agentic RAG...")
        rag_agent = OptimizedAgenticRAG()
        
        question = "Học trường kinh tế nào ở Hà Nội, học phí dưới 2 triệu/tháng, ngành tài chính ngân hàng có triển vọng khi ra trường"
        
        print("\n🔍 Đang phân tích câu hỏi và tìm kiếm thông tin...")
        print("=" * 70)
        
        response = rag_agent.chat(question)
        
        print("\n📋 Kết quả từ Agentic RAG:")
        print("=" * 70) 
        print(response)
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    # Chạy sync version
    main_sync()
    
    # Hoặc chạy async version
    # asyncio.run(main())