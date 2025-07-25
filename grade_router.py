from fastapi import APIRouter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from markitdown import MarkItDown
from dotenv import load_dotenv
import json
import io
import asyncio
from google import genai
from google.genai import types
from fastapi.responses import StreamingResponse, JSONResponse
import uuid

# Load environment variables
load_dotenv()
gemini_client = genai.Client()

# Pydantic models
class RubricCriterion(BaseModel):
    criterion: str
    max_score: float
    guide: List[str]

class Rubric(BaseModel):
    criteria: List[RubricCriterion]

class CriterionResult(BaseModel):
    criterion: str
    score: float
    max_score: float
    comment: Optional[str] = ""

class ExamResult(BaseModel):
    student_name: str
    student_id: str
    overall_comment: Optional[str] = ""
    criteria: List[CriterionResult]
    total_score: float = 0
    max_total_score: float = 10
    
    def __init__(self, **data):
        super().__init__(**data)
        self.total_score = sum(c.score for c in self.criteria)

class ExamResultList(BaseModel):
    results: List[ExamResult]

# Tạo router - không đặt prefix ở đây, sẽ đặt trong main_test.py
grade_router = APIRouter()

# Utilities
def convert_docx_to_markdown(docx_file) -> str:
    converter = MarkItDown()
    result = converter.convert_stream(docx_file)
    return result.text_content

def extract_rubric_from_markdown(markdown_text: str) -> Rubric:
    prompt = f"""
    Bạn là một chuyên gia phân tích tài liệu giáo dục. Hãy trích xuất thông tin rubric chấm điểm từ văn bản sau.
    
    Rubric thường bao gồm:
    - Các tiêu chí đánh giá (criterion)
    - Điểm tối đa cho mỗi tiêu chí (max_score)
    - Hướng dẫn chi tiết cho từng tiêu chí (guide)
    
    Văn bản cần phân tích:
    {markdown_text}
    
    Trả về kết quả dưới dạng JSON với cấu trúc: criteria (danh sách các tiêu chí), mỗi tiêu chí có criterion (tên), max_score (điểm tối đa), và guide (danh sách hướng dẫn).
    """
    
    system_instruction = """Bạn là chuyên gia phân tích tài liệu giáo dục. 
    Hãy trích xuất chính xác thông tin rubric từ văn bản được cung cấp.
    Trả về kết quả dưới dạng JSON theo đúng cấu trúc được yêu cầu.
    """
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': Rubric.model_json_schema(),
                'system_instruction': types.Part.from_text(text=system_instruction),
                'temperature': 0.0,
            },
        )
        
        output = json.loads(response.text)
        return Rubric(**output)
    except Exception as e:
        return Rubric(criteria=[])

async def call_llm_grading_async(assignment_prompt: str, rubric_criteria: List[RubricCriterion], file) -> ExamResult:
    essay_text = convert_docx_to_markdown(file)

    criteria_descriptions = []
    for idx, criterion in enumerate(rubric_criteria, 1):
        criteria_descriptions.append(f"{idx}. {criterion.criterion}: Điểm tối đa {criterion.max_score}")
        for guide_item in criterion.guide:
            criteria_descriptions.append(f"   - {guide_item}")
    
    criteria_text = "\n".join(criteria_descriptions)
    
    prompt = f"""
        Bạn là một giáo viên đang chấm bài luận của học sinh. Dưới đây là bài luận của học sinh, hướng dẫn chấm điểm chi tiết với các điểm cụ thể cho từng tiêu chí, và đề bài.

        Đối với mỗi tiêu chí trong hướng dẫn chấm điểm:
        - Kiểm tra xem bài luận có bao gồm từng điểm yêu cầu không.
        - Với mỗi điểm, nếu có trong bài, cho điểm tương ứng (như trong hướng dẫn), nếu thiếu, cho 0 điểm.
        - Với mỗi tiêu chí, tổng hợp các điểm tìm thấy, đưa ra nhận xét ngắn gọn bằng tiếng Việt giải thích kết quả.
        - Cuối cùng, tổng hợp tất cả các tiêu chí để có điểm tổng.
        - Trích xuất và trả về tên học sinh và mã số sinh viên từ nội dung bài luận.

        Đề bài:
        {assignment_prompt}

        Hướng dẫn chấm điểm và tiêu chuẩn đánh giá:
        {criteria_text}

        Chi tiết rubric (dạng JSON):
        {json.dumps([c.model_dump() for c in rubric_criteria], ensure_ascii=False, indent=2)}

        Bài luận:
        {essay_text}

        Trả về câu trả lời của bạn dưới dạng JSON, bao gồm các trường student_name và student_id.
    """
    
    try:
        system_instruction = """Bạn là một giáo viên chuyên nghiệp đang chấm bài. 
        Hãy phân tích bài luận của học sinh theo các tiêu chí đánh giá được cung cấp.
        Đối với mỗi tiêu chí:
        - Kiểm tra xem bài viết có đáp ứng các yêu cầu không
        - Cho điểm dựa trên mức độ đáp ứng của bài viết
        - Viết nhận xét ngắn gọn, rõ ràng bằng tiếng Việt
        
        Trả về kết quả dưới dạng JSON với các trường: student_name, student_id, overall_comment và criteria.
        Mỗi criterion cần có tên tiêu chí, điểm đạt được, điểm tối đa và nhận xét.
        
        QUAN TRỌNG: Đảm bảo tên của mỗi criterion trong kết quả phải CHÍNH XÁC với tên trong rubric được cung cấp.
        """
        
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': ExamResult.model_json_schema(),
                'system_instruction': types.Part.from_text(text=system_instruction),
                'temperature': 0.0,
                'seed': 42
            },
        )
        
        content = json.loads(response.text)
        return ExamResult(**content)
    except Exception as e:
        return ExamResult(
            student_name="Không rõ",
            student_id="Không rõ",
            overall_comment=f"Lỗi xử lý: {e}",
            criteria=[]
        )


# In-memory storage for grading sessions
grading_sessions: Dict[str, Dict[str, Any]] = {}

@grade_router.post("/extract-rubric")
async def extract_rubric(rubric_file: UploadFile = File(...)):
    try:
        file_content = await rubric_file.read()
        markdown_text = convert_docx_to_markdown(io.BytesIO(file_content))
        rubric = extract_rubric_from_markdown(markdown_text)
        return rubric
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@grade_router.post("/grade-exam")
async def grade_exam(
    assignment_prompt: str = Form(...),
    rubric_file: UploadFile = File(...),
    exam_files: List[UploadFile] = File(...)
):
    try:
        # Extract rubric
        rubric_content = await rubric_file.read()
        markdown_text = convert_docx_to_markdown(io.BytesIO(rubric_content))
        rubric = extract_rubric_from_markdown(markdown_text)
        
        # Process each exam file concurrently
        semaphore = asyncio.Semaphore(5)
        tasks = []

        async def process_file_with_semaphore(exam_file_upload: UploadFile):
            async with semaphore:
                file_content = await exam_file_upload.read()
                result = await call_llm_grading_async(
                    assignment_prompt,
                    rubric.criteria,
                    io.BytesIO(file_content)
                )
                return result

        for exam_file in exam_files:
            tasks.append(process_file_with_semaphore(exam_file))
        
        results = await asyncio.gather(*tasks)
        
        return ExamResultList(results=results)
    except Exception as e:
        return {"error": str(e)}

@grade_router.post("/upload-and-grade")
async def upload_and_grade(
    assignment_prompt: str = Form(...),
    rubric: str = Form(...),
    exam_files: List[UploadFile] = File(...)
):
    session_id = uuid.uuid4().hex
    try:
        rubric_data = json.loads(rubric)
        rubric_obj = Rubric(**rubric_data)
        
        file_contents = []
        for exam_file in exam_files:
            content = await exam_file.read()
            file_contents.append({'filename': exam_file.filename, 'content': content})
        
        grading_sessions[session_id] = {
            "assignment_prompt": assignment_prompt,
            "rubric_obj": rubric_obj,
            "file_contents": file_contents,
            "status": "pending",
            "results": []
        }
        return {"session_id": session_id}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid rubric JSON format.")
    except Exception as e:
        if session_id in grading_sessions:
            del grading_sessions[session_id]
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

async def stream_grading_results_for_session(session_id: str):
    session_data = grading_sessions.get(session_id)
    if not session_data:
        yield f"data: {json.dumps({'error': 'Invalid session ID or session expired'})}\n\n"
        return

    assignment_prompt = session_data["assignment_prompt"]
    rubric_obj = session_data["rubric_obj"]
    file_contents_with_filenames = session_data["file_contents"]
    
    session_data["status"] = "processing"

    semaphore = asyncio.Semaphore(5)
    total_files = len(file_contents_with_filenames)
    completed_count = 0
    
    async def process_file_content_with_semaphore(index, filename, file_content_bytes):
        async with semaphore:
            result = await call_llm_grading_async(
                assignment_prompt,
                rubric_obj.criteria,
                io.BytesIO(file_content_bytes)
            )
            return index, filename, result

    tasks = [
        process_file_content_with_semaphore(i, item['filename'], item['content'])
        for i, item in enumerate(file_contents_with_filenames)
    ]

    try:
        for completed_task in asyncio.as_completed(tasks):
            index, filename, result = await completed_task
            completed_count += 1
            session_data["results"].append({'filename': filename, 'result': result.model_dump()})
            yield f"data: {json.dumps({'index': index, 'filename': filename, 'completed': completed_count, 'total': total_files, 'result': result.model_dump()})}\n\n"
        session_data["status"] = "completed"
    except Exception as e:
        session_data["status"] = "error"
        session_data["error_message"] = str(e)
        yield f"data: {json.dumps({'error': f'Error during grading: {str(e)}'})}\n\n"

@grade_router.get("/stream-grading-progress/{session_id}")
async def stream_grading_progress(session_id: str):
    if session_id not in grading_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return StreamingResponse(stream_grading_results_for_session(session_id), media_type="text/event-stream")