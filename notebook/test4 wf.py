# ✅ Cách hiện tại: AgentWorkflow
# Gồm agent linearly: Filter → Score → RAG → Summarizer.

# Đơn giản, dễ hiểu, dùng can_handoff_to để định luồng.

# Limit: thiếu khả năng loop, parallelism, hoặc branching conditional phức tạp.


# pip install llama-index llama-index-utils-workflow openai python-dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event

from typing import List


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
# 2. Custom Event types
class FilteredEvent(Event):
    filtered: List[dict]  # danh sách {uni, program, cutoff}

class ScoredEvent(Event):
    scored: List[dict]  # thêm trường score

class DetailedEvent(Event):
    details: List[dict]  # thêm info từ retrieved text

# 3. Workflow definition
class UniWorkflow(Workflow):

    @step
    async def step_filter(self, ev: StartEvent) -> FilteredEvent:
        gpa = ev.gpa
        budget = ev.budget
        location = ev.location
        filtered = [
            {"uni": u["name"], "program": p, "cutoff": d["cutoff"]}
            for u in UNIVERSITIES_DB.values()
            for p, d in u["programs"].items()
            if d["cutoff"] <= gpa
            and u["tuition_per_month"] / 1e6 <= budget
            and location in u["location"]
        ]
        return FilteredEvent(filtered=filtered)

    @step
    async def step_score(self, ev: FilteredEvent) -> ScoredEvent:
        scored = sorted(
            [{"uni": f["uni"], "program": f["program"],
              "cutoff": f["cutoff"], 
              "score": (10 - f["cutoff"]) * 10}
             for f in ev.filtered],
            key=lambda x: -x["score"]
        )
        return ScoredEvent(scored=scored)

    @step
    async def step_retrieve(self, ev: ScoredEvent) -> DetailedEvent:
        results = []
        for item in ev.scored[:3]:  # top 3
            query = f"{item['program']} tại {item['uni']}"
            resp = index.as_query_engine().query(query)
            results.append({
                **item,
                "rag": resp.response
            })
        return DetailedEvent(details=results)

    @step
    async def step_summarize(self, ev: DetailedEvent) -> StopEvent:
        s = ""
        for d in ev.details:
            s += f"🏫 **{d['uni']}** – {d['program']} (score: {d['score']:.0f})\n"
            s += f"{d['rag']}\n\n"
        return StopEvent(result=s)

# 4. Chạy workflow
async def main():
    w = UniWorkflow(timeout=60, verbose=True)
    res = await w.run(gpa=8.2, budget=2.5, location="Hà Nội")
    print(str(res))

if __name__ == "__main__":
    asyncio.run(main())