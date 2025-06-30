import os
import asyncio
from typing import List, Dict, Optional
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool, QueryEngineTool
import logging

from dotenv import load_dotenv

load_dotenv()

# Khởi tạo RAG index
def create_rag_index():
    """Tạo LlamaCloudIndex"""
    index = LlamaCloudIndex(
        name="dsdaihoc",
        project_name="Default", 
        organization_id="1bcf5fb2-bd7d-4c76-b6d5-bb793486e1b3",
        api_key=os.getenv("LLAMA_CLOUD_API_KEY")
    )
    return index

# Structured data cho tooling
UNIVERSITIES_DB = {
    "bach_khoa_hn": {
        "name": "Đại học Bách Khoa Hà Nội",
        "cutoff_gpa": 8.5,
        "tuition_per_month": 2000000,
        "location": "Hà Nội",
        "programs": {
            "Công nghệ thông tin": {"cutoff": 8.7, "career": "Lập trình viên, Data Scientist"},
            "Cơ khí": {"cutoff": 8.3, "career": "Kỹ sư cơ khí, Thiết kế sản phẩm"},
            "Điện tử viễn thông": {"cutoff": 8.5, "career": "Kỹ sư điện tử, IoT Developer"}
        },
        "facilities": "Lab hiện đại, thư viện lớn",
        "ranking": "Top 1 kỹ thuật Việt Nam"
    },
    "kinh_te_qd": {
        "name": "Đại học Kinh tế Quốc dân",
        "cutoff_gpa": 7.8,
        "tuition_per_month": 1500000,
        "location": "Hà Nội",
        "programs": {
            "Kinh tế": {"cutoff": 8.0, "career": "Nhà kinh tế, Phân tích chính sách"},
            "Tài chính Ngân hàng": {"cutoff": 8.2, "career": "Banker, Financial Analyst"},
            "Marketing": {"cutoff": 7.5, "career": "Marketing Manager, Digital Marketer"}
        },
        "facilities": "Mô phỏng giao dịch chứng khoán",
        "ranking": "Top 1 kinh tế Việt Nam"
    },
    "thuong_mai": {
        "name": "Đại học Thương mại",
        "cutoff_gpa": 7.5,
        "tuition_per_month": 1200000,
        "location": "Hà Nội",
        "programs": {
            "Kinh tế": {"cutoff": 7.8, "career": "Chuyên viên kinh tế, Tư vấn"},
            "Quản trị kinh doanh": {"cutoff": 7.6, "career": "Manager, Business Analyst"},
            "Logistics": {"cutoff": 7.2, "career": "Supply Chain Manager"}
        },
        "facilities": "Trung tâm thực hành kinh doanh",
        "ranking": "Top 3 kinh tế Việt Nam"
    }
}

CAREER_INFO = {
    "kinh tế": {
        "prospects": "Ngành Kinh tế có triển vọng tốt với mức lương khởi điểm 8-12 triệu/tháng. Cơ hội việc làm ở ngân hàng, tập đoàn, tư vấn tài chính.",
        "skills": "Phân tích dữ liệu, Excel nâng cao, tư duy logic, tiếng Anh"
    },
    "công nghệ thông tin": {
        "prospects": "IT có nhu cầu cao với lương 12-20 triệu/tháng. Nhiều cơ hội remote work và phát triển sự nghiệp.",
        "skills": "Lập trình, tư duy thuật toán, học hỏi liên tục"
    }
}

# Hàm tìm trường đại học với RAG
async def find_universities(area: str, major: str, max_fee: float = 2.0) -> str:
    """Tìm trường theo khu vực và học phí sử dụng RAG"""
    print(f"🔍 Đang tìm trường ngành {major} ở {area} với học phí dưới {max_fee} triệu...")
    
    index = create_rag_index()
    query_engine = index.as_query_engine(similarity_top_k=3)
    query = f"Tìm trường đại học ngành {major} ở {area} với học phí dưới {max_fee} triệu/tháng"
    response = await query_engine.aquery(query)
    return str(response)

# Hàm tư vấn nghề nghiệp với RAG
async def career_advice(major: str) -> str:
    """Tư vấn triển vọng nghề nghiệp sử dụng RAG"""
    print(f"💼 Đang phân tích triển vọng ngành {major}...")
    
    index = create_rag_index()
    query_engine = index.as_query_engine(similarity_top_k=3)
    query = f"Triển vọng nghề nghiệp ngành {major}, cơ hội việc làm và mức lương"
    response = await query_engine.aquery(query)
    return str(response)

# Hàm tìm học phí ngành kinh tế rẻ nhất
def find_cheapest_economics_program() -> str:
    """Tìm học phí ngành kinh tế rẻ nhất từ structured data"""
    economics_programs = []
    
    for uni_id, uni_data in UNIVERSITIES_DB.items():
        for program_name, program_info in uni_data["programs"].items():
            if "kinh tế" in program_name.lower():
                economics_programs.append({
                    "university": uni_data["name"],
                    "program": program_name,
                    "tuition": uni_data["tuition_per_month"],
                    "cutoff": program_info["cutoff"],
                    "career": program_info["career"]
                })
    
    # Sắp xếp theo học phí
    economics_programs.sort(key=lambda x: x["tuition"])
    
    if economics_programs:
        cheapest = economics_programs[0]
        return f"Học phí ngành kinh tế rẻ nhất:\n- Trường: {cheapest['university']}\n- Ngành: {cheapest['program']}\n- Học phí: {cheapest['tuition']:,} VND/tháng\n- Điểm chuẩn: {cheapest['cutoff']}\n- Triển vọng: {cheapest['career']}"
    
    return "Không tìm thấy ngành kinh tế nào."

# Tạo tools
uni_tool = FunctionTool.from_defaults(
    fn=find_universities,
    name="find_universities",
    description="Tìm trường đại học theo khu vực, ngành học và học phí tối đa"
)

career_tool = FunctionTool.from_defaults(
    fn=career_advice, 
    name="career_advice",
    description="Tư vấn triển vọng nghề nghiệp của một ngành học"
)

cheapest_economics_tool = FunctionTool.from_defaults(
    fn=find_cheapest_economics_program,
    name="find_cheapest_economics",
    description="Tìm học phí ngành kinh tế rẻ nhất từ cơ sở dữ liệu"
)

# System prompt tiếng Việt
SYSTEM_PROMPT = """
🎓 XIN CHÀO! Tôi là trợ lý tư vấn giáo dục của bạn!

Tôi có thể giúp bạn:
✨ Tìm trường đại học phù hợp theo khu vực và ngân sách
💼 Tư vấn triển vọng nghề nghiệp các ngành học
📊 So sánh học phí và chất lượng đào tạo

Khi bạn hỏi về trường đại học, tôi sẽ dùng find_universities
Khi bạn hỏi về triển vọng nghề nghiệp, tôi sẽ dùng career_advice

Tôi sẽ trả lời bằng tiếng Việt, thân thiện và chi tiết! 🌟
"""

# Khởi tạo LLM với fallback
def create_llm():
    """Tạo LLM với error handling"""
    try:
        # Thử Google AI trước
        if os.getenv("GOOGLE_API_KEY"):
            print("🚀 Sử dụng Google AI...")
            return GoogleGenAI(
                model="gemini-2.0-flash",
            )
    except Exception as e:
        print(f"⚠️ Không thể kết nối Google AI: {e}")
    
    
    # Nếu không có API key nào hoạt động
    print("❌ Không có API key hợp lệ. Vui lòng cấu hình GOOGLE_API_KEY hoặc OPENAI_API_KEY")
    return None

# Hàm main đơn giản
async def main():
    """Chạy chương trình chính"""
    print("🎓 KHỞI ĐỘNG HỆ THỐNG TƯ VẤN GIÁO DỤC")
    print("=" * 50)
    
    # Tạo LLM
    llm = create_llm()
    if not llm:
        print("❌ Không thể khởi tạo LLM. Vui lòng kiểm tra API keys!")
        return
    
    try:
        # Tạo agent
        agent = FunctionAgent(
            tools=[uni_tool, career_tool, cheapest_economics_tool],
            llm=llm,
            system_prompt=SYSTEM_PROMPT,
            verbose=True
        )
        
        # Tạo workflow
        workflow = AgentWorkflow(
            agents=[agent],
            root_agent=agent
        )
        
        print("✅ Khởi tạo thành công!")
        
        # Test query
        user_query = (
            "Cho tôi học phí ngành Kinh tế ở khu vực Cầu Giấy, "
            "học phí dưới 2 triệu/tháng, ngành nào có triển vọng khi ra trường"
        )
        
        print(f"\n💬 Câu hỏi: {user_query}")
        print("\n🤖 Đang xử lý...")
        
        response = await workflow.run(user_msg=user_query)
        
        print("\n📋 KẾT QUẢ:")
        print("=" * 50)
        print(response)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("\n🔧 TEST STRUCTURED DATA:")
        
        # Test structured data function
        cheapest = find_cheapest_economics_program()
        print("\n💰 HỌC PHÍ KINH TẾ RẺ NHẤT:")
        print(cheapest)

# Cấu hình API keys (thêm vào file .env hoặc export)
if __name__ == "__main__":
    print("💡 HƯỚNG DẪN CẤU HÌNH:")
    print("1. Tạo file .env với:")
    print("   GOOGLE_API_KEY=your_google_api_key")
    print("   # hoặc")  
    print("   OPENAI_API_KEY=your_openai_api_key")
    print("\n2. Hoặc export trong terminal:")
    print("   export GOOGLE_API_KEY='your_key'")
    print("\n" + "=" * 50)
    
    asyncio.run(main())