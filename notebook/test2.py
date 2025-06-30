# pip install llama-index llama-index-utils-workflow openai python-dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
Settings.llm = OpenAI(model="gpt-4")

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
# 1. Load documents (university data)
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=5)


# 2. Tool: retrieval
rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="university_rag",
    description="Retrieve info about university programs and admission requirements"
)

# 3. Tool: filter logic
def filter_unis(gpa: float, budget: float, location: str) -> list:
    # giả định có dict UNIVERSITIES_DB tương tự
    results = []
    for uni in UNIVERSITIES_DB.values():
        for prog, d in uni["programs"].items():
            if d["cutoff"] <= gpa and uni["tuition_per_month"]/1e6 <= budget and location in uni["location"]:
                results.append({"uni": uni["name"], "program": prog})
    return results

filter_tool = FunctionTool.from_defaults(fn=filter_unis)

# 4. Agent chuyên gia: lọc
filter_agent = FunctionAgent(
    name="FilterAgent",
    description="Filter university programs by GPA, budget, location",
    tools=[filter_tool],
    system_prompt="You receive structured user data, return list of matching programs."
)

# 5. Agent tổng hợp: RAG + heat up answer
rag_agent = FunctionAgent(
    name="RAGAgent",
    description="Retrieve details about specified program from knowledge base",
    tools=[rag_tool],
    system_prompt="You get a list of program names. Retrieve details about each."
)




async def main():
    workflow = AgentWorkflow(
        agents=[filter_agent, rag_agent],
        root_agent=filter_agent.name,
    )
    response = await workflow.run(
        user_msg="GPA 8.2, thích CNTT, ngân sách 2.5 triệu/tháng, ở Hà Nội. Hỏi ngành phù hợp và lý do."
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())