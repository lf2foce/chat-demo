"""
Enhanced Agentic RAG với Tools và Memory - University Advisor
"""

import os
import json
from typing import List, Dict, Optional
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from dotenv import load_dotenv

# Load env
load_dotenv()
# Setup LLM và Embeddings
Settings.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# Database các trường đại học chi tiết
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
    "fpt": {
        "name": "Đại học FPT",
        "cutoff_gpa": 6.5,
        "tuition_per_month": 3000000,
        "location": "TP.HCM, Hà Nội, Đà Nẵng",
        "programs": {
            "Công nghệ thông tin": {"cutoff": 6.8, "career": "Software Engineer, DevOps"},
            "Kinh doanh Quốc tế": {"cutoff": 6.3, "career": "Business Analyst, Project Manager"},
            "Thiết kế Đồ họa": {"cutoff": 6.0, "career": "UI/UX Designer, Graphic Designer"}
        },
        "facilities": "Campus hiện đại, lab Apple, thực tập tại FPT",
        "ranking": "Top CNTT tư thục"
    },
    "rmit": {
        "name": "RMIT Việt Nam",
        "cutoff_gpa": 7.0,
        "tuition_per_month": 8000000,
        "location": "TP.HCM",
        "programs": {
            "Computer Science": {"cutoff": 7.2, "career": "Tech Lead, Research Scientist"},
            "Business Administration": {"cutoff": 6.8, "career": "CEO, Consultant"},
            "Media & Communication": {"cutoff": 6.5, "career": "Content Creator, PR Manager"}
        },
        "facilities": "Chuẩn quốc tế, bằng cấp Australia",
        "ranking": "Top 250 thế giới"
    }
}

# Tạo documents chi tiết cho RAG
docs = []
for uni_id, uni_data in UNIVERSITIES_DB.items():
    # Document tổng quan
    overview = f"""
    {uni_data['name']}:
    - Điểm chuẩn chung: {uni_data['cutoff_gpa']}/10
    - Học phí: {uni_data['tuition_per_month']:,} VNĐ/tháng
    - Địa điểm: {uni_data['location']}
    - Cơ sở vật chất: {uni_data['facilities']}
    - Xếp hạng: {uni_data['ranking']}
    """
    
    # Document chi tiết từng ngành
    for program, details in uni_data['programs'].items():
        program_info = f"""
        {uni_data['name']} - Ngành {program}:
        - Điểm chuẩn: {details['cutoff']}/10
        - Cơ hội nghề nghiệp: {details['career']}
        - Trường: {uni_data['name']}
        - Học phí: {uni_data['tuition_per_month']:,} VNĐ/tháng
        - Địa điểm: {uni_data['location']}
        """
        docs.append(Document(text=program_info))
    
    docs.append(Document(text=overview))

# Tạo Vector Index
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=5)

# Tool 1: Tìm kiếm trường phù hợp
def search_universities(
    desired_program: Optional[str] = None,
    max_tuition: Optional[float] = None,
    location: Optional[str] = None,
    min_gpa: Optional[float] = None
) -> str:
    """Tìm kiếm các trường đại học phù hợp với tiêu chí"""
    
    results = []
    
    for uni_id, uni_data in UNIVERSITIES_DB.items():
        matches = []
        
        # Check location
        if location and location.lower() in uni_data['location'].lower():
            matches.append(f"✅ Có cơ sở tại {location}")
        
        # Check budget
        if max_tuition and uni_data['tuition_per_month'] <= max_tuition * 1000000:
            matches.append(f"✅ Trong ngân sách ({uni_data['tuition_per_month']:,} VNĐ/tháng)")
        
        # Check programs
        if desired_program:
            for prog, details in uni_data['programs'].items():
                if desired_program.lower() in prog.lower():
                    gpa_status = ""
                    if min_gpa:
                        if min_gpa >= details['cutoff']:
                            gpa_status = f" ✅ GPA đủ điều kiện"
                        else:
                            gpa_status = f" ⚠️ Cần GPA {details['cutoff']} (bạn có {min_gpa})"
                    
                    matches.append(f"✅ Có ngành {prog} - Điểm chuẩn: {details['cutoff']}{gpa_status}")
                    matches.append(f"   💼 Cơ hội việc làm: {details['career']}")
        
        if matches:
            result = f"\n🏫 **{uni_data['name']}**\n"
            result += f"📍 {uni_data['location']} | 💰 {uni_data['tuition_per_month']:,} VNĐ/tháng\n"
            result += f"🏆 {uni_data['ranking']}\n"
            for match in matches:
                result += f"{match}\n"
            results.append(result)
    
    if not results:
        return "Không tìm thấy trường nào phù hợp với tiêu chí. Hãy thử mở rộng điều kiện tìm kiếm."
    
    return "\n".join(results)

# Tool 2: Chấm điểm chi tiết
def score_program(
    user_gpa: Optional[float] = None, 
    desired_program: Optional[str] = None, 
    budget: Optional[float] = None, 
    location_pref: Optional[str] = None, 
    university_name: str = ""
) -> str:
    """Chấm điểm chi tiết một trường đại học cụ thể"""
    
    # Tìm trường trong database
    target_uni = None
    for uni_id, uni_data in UNIVERSITIES_DB.items():
        if university_name.lower() in uni_data['name'].lower():
            target_uni = uni_data
            break
    
    if not target_uni:
        return f"Không tìm thấy thông tin về {university_name}"
    
    score = 0
    max_score = 0
    details = []
    
    # Chấm điểm GPA
    if user_gpa is not None:
        max_score += 5
        if user_gpa >= target_uni['cutoff_gpa']:
            score += 5
            details.append(f"✅ GPA phù hợp ({user_gpa} >= {target_uni['cutoff_gpa']})")
        else:
            gap = target_uni['cutoff_gpa'] - user_gpa
            details.append(f"⚠️ GPA chưa đủ (thiếu {gap:.1f} điểm)")
    
    # Chấm điểm ngân sách
    if budget is not None:
        max_score += 3
        monthly_cost = target_uni['tuition_per_month'] / 1000000
        if monthly_cost <= budget:
            score += 3
            details.append(f"✅ Trong ngân sách ({monthly_cost:.1f}M <= {budget}M)")
        else:
            over_budget = monthly_cost - budget
            details.append(f"⚠️ Vượt ngân sách {over_budget:.1f}M VNĐ/tháng")
    
    # Chấm điểm địa điểm
    if location_pref:
        max_score += 2
        if location_pref.lower() in target_uni['location'].lower():
            score += 2
            details.append(f"✅ Có cơ sở tại {location_pref}")
        else:
            details.append(f"⚠️ Không có cơ sở tại {location_pref}")
    
    # Chấm điểm ngành học
    if desired_program:
        max_score += 4
        program_found = False
        for prog, prog_details in target_uni['programs'].items():
            if desired_program.lower() in prog.lower():
                program_found = True
                score += 4
                details.append(f"✅ Có ngành {prog}")
                details.append(f"   📋 Điểm chuẩn ngành: {prog_details['cutoff']}")
                details.append(f"   💼 Cơ hội việc làm: {prog_details['career']}")
                break
        
        if not program_found:
            details.append(f"⚠️ Không có ngành {desired_program}")
    
    # Thông tin bổ sung
    details.extend([
        f"🏗️ Cơ sở vật chất: {target_uni['facilities']}",
        f"🏆 Xếp hạng: {target_uni['ranking']}"
    ])
    
    result = f"\n📊 **Đánh giá {target_uni['name']}**\n"
    if max_score > 0:
        percentage = (score / max_score) * 100
        result += f"🎯 Điểm phù hợp: {score}/{max_score} ({percentage:.0f}%)\n\n"
    
    result += "\n".join(details)
    
    return result

# Tool 3: Gợi ý dựa trên profile
def suggest_programs(
    user_gpa: Optional[float] = None,
    interests: Optional[str] = None,
    budget: Optional[float] = None
) -> str:
    """Gợi ý các ngành học và trường phù hợp với profile"""
    
    suggestions = []
    
    for uni_id, uni_data in UNIVERSITIES_DB.items():
        for program, details in uni_data['programs'].items():
            suitable = True
            reasons = []
            
            # Check GPA
            if user_gpa is not None:
                if user_gpa >= details['cutoff']:
                    reasons.append(f"✅ GPA đủ điều kiện ({user_gpa} >= {details['cutoff']})")
                else:
                    suitable = False
                    continue
            
            # Check budget
            if budget is not None:
                monthly_cost = uni_data['tuition_per_month'] / 1000000
                if monthly_cost <= budget:
                    reasons.append(f"✅ Trong ngân sách ({monthly_cost:.1f}M/tháng)")
                else:
                    suitable = False
                    continue
            
            # Check interests
            if interests:
                interest_keywords = interests.lower().split()
                program_text = f"{program} {details['career']}".lower()
                
                if any(keyword in program_text for keyword in interest_keywords):
                    reasons.append(f"✅ Phù hợp với sở thích ({interests})")
            
            if suitable and reasons:
                suggestion = f"\n🎓 **{program}** - {uni_data['name']}\n"
                suggestion += f"📍 {uni_data['location']} | 💰 {uni_data['tuition_per_month']:,} VNĐ/tháng\n"
                suggestion += f"📊 Điểm chuẩn: {details['cutoff']}/10\n"
                suggestion += f"💼 Cơ hội việc làm: {details['career']}\n"
                suggestion += "\n".join(reasons)
                
                suggestions.append((details['cutoff'], suggestion))
    
    if not suggestions:
        return "Không tìm thấy gợi ý phù hợp. Hãy cung cấp thêm thông tin hoặc mở rộng tiêu chí."
    
    # Sắp xếp theo điểm chuẩn (thấp đến cao để dễ đỗ hơn)
    suggestions.sort(key=lambda x: x[0])
    
    result = "🔍 **Gợi ý các ngành học phù hợp:**\n"
    result += "\n".join([s[1] for s in suggestions[:5]])  # Top 5 suggestions
    
    return result

# Setup Tools
rag_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="knowledge_search",
        description="Tìm kiếm thông tin chi tiết về trường đại học, ngành học từ knowledge base"
    )
)

search_tool = FunctionTool.from_defaults(
    fn=search_universities,
    name="search_universities",
    description="Tìm kiếm trường đại học theo tiêu chí: ngành học, học phí, địa điểm, GPA"
)

score_tool = FunctionTool.from_defaults(
    fn=score_program,
    name="score_program", 
    description="Chấm điểm chi tiết một trường đại học cụ thể dựa trên GPA, ngành, ngân sách, địa điểm"
)

suggest_tool = FunctionTool.from_defaults(
    fn=suggest_programs,
    name="suggest_programs",
    description="Gợi ý các ngành học và trường phù hợp dựa trên GPA, sở thích, ngân sách"
)

tools = [rag_tool, search_tool, score_tool, suggest_tool]

# Setup Memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Tạo Agent với system prompt chi tiết
agent = ReActAgent.from_tools(
    tools=tools,
    memory=memory,
    verbose=True,
    system_prompt="""
Bạn là trợ lý tư vấn tuyển sinh đại học thông minh và chu đáo.

NHIỆM VỤ:
1. Thu thập thông tin từ học sinh: GPA, ngành quan tâm, ngân sách, địa điểm
2. Sử dụng tools để tìm kiếm, chấm điểm, gợi ý
3. Đưa ra lời khuyên cụ thể và thực tế

CÁCH XỬ LÝ KHI THIẾU THÔNG TIN:
- Nếu thiếu GPA: Hỏi lịch sự và giải thích tầm quan trọng
- Nếu thiếu ngành: Gợi ý dựa trên sở thích, khả năng
- Nếu thiếu ngân sách: Đưa ra các mức tham khảo
- Nếu thiếu địa điểm: Hỏi về sở thích và gia đình

TOOLS USAGE:
- knowledge_search: Tìm thông tin chi tiết cụ thể
- search_universities: Lọc trường theo nhiều tiêu chí
- score_program: Đánh giá chi tiết một trường cụ thể  
- suggest_programs: Gợi ý dựa trên profile tổng thể

PHONG CÁCH:
- Thân thiện, chuyên nghiệp
- Đưa ra lời khuyên thực tế
- Giải thích rõ ràng các tiêu chí
- Khuyến khích học sinh cung cấp thêm thông tin
""".strip()
)

# Chat function với hướng dẫn
def chat():
    print("🎓 Trợ lý tư vấn tuyển sinh đại học")
    print("💡 Hãy chia sẻ: GPA, ngành quan tâm, ngân sách, nơi ở để được tư vấn tốt nhất!")
    print("❓ Ví dụ: 'Em có GPA 8.2, thích CNTT, ngân sách 2.5 triệu/tháng, ở Hà Nội'")
    print("📝 Gõ 'quit' để thoát\n")
    
    while True:
        user_input = input("\n👤 Bạn: ")
        if user_input.lower() == 'quit':
            break
        
        response = agent.chat(user_input)
        print(f"\n🤖 Trợ lý: {response}")

# Run chat
if __name__ == "__main__":
    chat()