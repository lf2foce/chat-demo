import pandas as pd
import os
from pathlib import Path

def csv_to_structured_markdown(csv_path, output_dir="./docs_processed"):
    """
    Convert university CSV data to structured markdown for better LLM understanding
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Method 1: Single comprehensive document
    create_comprehensive_markdown(df, output_dir)
    
    # Method 2: Individual university documents (better for retrieval)
    create_individual_documents(df, output_dir)
    
    print(f"✅ Converted {len(df)} universities to markdown format")

def create_comprehensive_markdown(df, output_dir):
    """Create one comprehensive markdown file with all universities"""
    
    content = """# Danh sách các Trường Đại học Việt Nam

Đây là thông tin chi tiết về các trường đại học tại Việt Nam, bao gồm thông tin cơ bản, chương trình đào tạo, và tiêu chí tuyển sinh.

"""
    
    for idx, row in df.iterrows():
        university_name = row.get('Tên trường', 'N/A')
        description = row.get('Mô tả trường', 'N/A')
        changes_2025 = row.get('Thay đổi 2025', 'N/A')
        admission_info = row.get('Thông tin tuyển sinh', 'N/A')
        programs = row.get('Chương trình', 'N/A')
        standards = row.get('Điểm chuẩn', 'N/A')
        tags = row.get('Tags', 'N/A')
        
        content += f"""
## {university_name}

### Thông tin cơ bản
{description}

### Thay đổi năm 2025
{changes_2025}

### Thông tin tuyển sinh
{admission_info}

### Chương trình đào tạo
{programs}

### Điểm chuẩn
{standards}

### Từ khóa liên quan
{tags}

---

"""
    
    # Save comprehensive file
    with open(f"{output_dir}/universities_comprehensive.md", "w", encoding="utf-8") as f:
        f.write(content)

def create_individual_documents(df, output_dir):
    """Create individual markdown files for each university (better for vector search)"""
    
    individual_dir = Path(output_dir) / "individual_universities"
    individual_dir.mkdir(exist_ok=True)
    
    for idx, row in df.iterrows():
        university_name = row.get('Tên trường', 'N/A')
        description = row.get('Mô tả trường', 'N/A')
        changes_2025 = row.get('Thay đổi 2025', 'N/A')
        admission_info = row.get('Thông tin tuyển sinh', 'N/A')
        programs = row.get('Chương trình', 'N/A')
        standards = row.get('Điểm chuẩn', 'N/A')
        tags = row.get('Tags', 'N/A')
        
        # Create safe filename
        safe_name = "".join(c for c in university_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{idx+1:03d}_{safe_name.replace(' ', '_')}.md"
        
        content = f"""# {university_name}

## Thông tin tổng quan
{description}

## Những thay đổi mới năm 2025
{changes_2025}

## Thông tin tuyển sinh và xét tuyển
{admission_info}

## Các chương trình đào tạo
{programs}

## Điểm chuẩn tham khảo
{standards}

## Các từ khóa và ngành học liên quan
{tags}

## Metadata
- **Tên trường**: {university_name}
- **Loại thông tin**: Thông tin tuyển sinh đại học
- **Năm**: 2025
- **Nguồn**: Dữ liệu tuyển sinh chính thức
"""
        
        # Save individual file
        with open(individual_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)

def create_json_structured(df, output_dir):
    """Alternative: Create structured JSON for more complex queries"""
    
    universities_data = []
    
    for idx, row in df.iterrows():
        university = {
            "id": idx + 1,
            "name": row.get('Tên trường', ''),
            "description": row.get('Mô tả trường', ''),
            "changes_2025": row.get('Thay đổi 2025', ''),
            "admission_info": row.get('Thông tin tuyển sinh', ''),
            "programs": row.get('Chương trình', ''),
            "admission_scores": row.get('Điểm chuẩn', ''),
            "tags": row.get('Tags', '').split(', ') if row.get('Tags') else [],
            "metadata": {
                "year": 2025,
                "type": "university",
                "country": "Vietnam"
            }
        }
        universities_data.append(university)
    
    # Convert to markdown with JSON structure
    content = "# Dữ liệu Trường Đại học Việt Nam (Structured)\n\n"
    content += "Dữ liệu được cấu trúc dưới dạng JSON cho các truy vấn phức tạp:\n\n"
    
    import json
    content += "```json\n"
    content += json.dumps(universities_data, ensure_ascii=False, indent=2)
    content += "\n```"
    
    with open(f"{output_dir}/universities_structured.md", "w", encoding="utf-8") as f:
        f.write(content)

# Usage example
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "../notebook/Trường-Grid view (2).csv"  # Adjust path as needed
    
    if os.path.exists(csv_file):
        csv_to_structured_markdown(csv_file)
    else:
        print(f"CSV file not found: {csv_file}")
        print("Please update the csv_file path to match your file location")