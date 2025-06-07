from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from markitdown import MarkItDown
from dotenv import load_dotenv
import os
import json
# import together
# import pandas as pd
import io
import asyncio
from google import genai
from google.genai import types
from fastapi.responses import StreamingResponse
from datetime import datetime
import tempfile

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

def parse_exam_file(file) -> str:
    return convert_docx_to_markdown(file)

async def call_llm_grading_async(assignment_prompt: str, rubric_criteria: List[RubricCriterion], file) -> ExamResult:
    essay_text = parse_exam_file(file)

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

### Create FastAPI instance with custom docs and openapi url
app = FastAPI(docs_url="/api/py/docs", openapi_url="/api/py/openapi.json")

@app.post("/api/py/extract-rubric")
async def extract_rubric(rubric_file: UploadFile = File(...)):
    try:
        file_content = await rubric_file.read()
        markdown_text = convert_docx_to_markdown(io.BytesIO(file_content))
        rubric = extract_rubric_from_markdown(markdown_text)
        return rubric
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/py/grade-exam")
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
        
        # Process each exam file
        results = []
        for exam_file in exam_files:
            file_content = await exam_file.read()
            result = await call_llm_grading_async(
                assignment_prompt,
                rubric.criteria,
                io.BytesIO(file_content)
            )
            results.append(result)
        
        return ExamResultList(results=results)
    except Exception as e:
        return {"error": str(e)}



@app.post("/api/py/grade-exam-with-rubric")
async def grade_exam_with_rubric(
    assignment_prompt: str = Form(...),
    rubric: str = Form(...),
    exam_files: List[UploadFile] = File(...)
):
    try:
        # Parse rubric from JSON string
        rubric_data = json.loads(rubric)
        rubric_obj = Rubric(**rubric_data)
        
        # Đọc tất cả file content trước để tránh lỗi file đã đóng
        file_contents = []
        for exam_file in exam_files:
            content = await exam_file.read()
            file_contents.append((exam_file.filename, content))
        
        # Hàm xử lý file content không có semaphore (để batch tự quản lý)
        async def process_file_content(filename, file_content):
            result = await call_llm_grading_async(
                assignment_prompt,
                rubric_obj.criteria,
                io.BytesIO(file_content)
            )
            return result
        
        # Nếu muốn trả về kết quả đồng bộ (đợi tất cả hoàn thành)
        if not rubric_data.get("stream", False):
            tasks = [process_file_content(filename, content) for filename, content in file_contents]
            results = await asyncio.gather(*tasks)
            return ExamResultList(results=results)
        
        # Nếu muốn streaming response (xử lý đồng thời tất cả file với semaphore)
        async def stream_results():
            semaphore = asyncio.Semaphore(5)  # Giới hạn 5 file đồng thời
            total_files = len(file_contents)
            all_results = []  # Thu thập tất cả kết quả
            completed_count = 0
            
            async def process_with_semaphore(index, filename, content):
                async with semaphore:
                    result = await process_file_content(filename, content)
                    return index, filename, result
            
            # Tạo tất cả tasks cùng lúc
            tasks = [
                process_with_semaphore(i, filename, content)
                for i, (filename, content) in enumerate(file_contents)
            ]
            
            # Sử dụng asyncio.as_completed để stream kết quả ngay khi có
            for completed_task in asyncio.as_completed(tasks):
                index, filename, result = await completed_task
                completed_count += 1
                
                # Lưu kết quả để xuất Excel sau này
                all_results.append({
                    'filename': filename,
                    'result': result
                })
                
                yield f"data: {json.dumps({'index': index, 'completed': completed_count, 'total': total_files, 'result': result.model_dump()})}\n\n"
            
            # Sau khi tất cả file được xử lý, tạo file Excel nếu được yêu cầu
            # if  all_results:
            #     try:
            #         # Tạo DataFrame từ kết quả
            #         excel_data = []
            #         for item in all_results:
            #             result_data = {
            #                 'Filename': item['filename'],
            #                 'Student Name': item['result'].student_name,
            #                 'Student ID': item['result'].student_id,
            #                 'Total Score': item['result'].total_score,
            #                 'Max Total Score': item['result'].max_total_score,
            #                 'Overall Comment': item['result'].overall_comment
            #             }
                        
            #             # Thêm điểm từng tiêu chí
            #             for criterion in item['result'].criteria:
            #                 result_data[f"{criterion.criterion} Score"] = criterion.score
            #                 result_data[f"{criterion.criterion} Max Score"] = criterion.max_score
            #                 result_data[f"{criterion.criterion} Comment"] = criterion.comment
                        
            #             excel_data.append(result_data)
                    
            #         df = pd.DataFrame(excel_data)
                    
            #         # Tạo file Excel trong thư mục api
            #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #         excel_filename = f"grading_results_{timestamp}.xlsx"
            #         excel_path = f"./api/{excel_filename}"
                    
            #         # Xuất ra Excel với định dạng
            #         with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            #             df.to_excel(writer, sheet_name='Grading Results', index=False)
                        
            #             # Tự động điều chỉnh độ rộng cột
            #             worksheet = writer.sheets['Grading Results']
            #             for column in worksheet.columns:
            #                 max_length = 0
            #                 column_letter = column[0].column_letter
            #                 for cell in column:
            #                     try:
            #                         if len(str(cell.value)) > max_length:
            #                             max_length = len(str(cell.value))
            #                     except:
            #                         pass
            #                 adjusted_width = min(max_length + 2, 50)
            #                 worksheet.column_dimensions[column_letter].width = adjusted_width
                    
               
            #         # Stream thông báo về file Excel đã tạo
            #         yield f"data: {json.dumps({'type': 'excel_created', 'filename': excel_filename, 'path': excel_path, 'download_url': f'/api/py/download-excel/{excel_filename}'})}\n\n"
                    
            #     except Exception as e:
            #         yield f"data: {json.dumps({'type': 'excel_error', 'error': str(e)})}\n\n"

        
        return StreamingResponse(stream_results(), media_type="text/event-stream")
    except Exception as e:
        return {"error": str(e)}