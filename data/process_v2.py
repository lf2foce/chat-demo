import pandas as pd
import os
import re
from pathlib import Path
import json

class UniversityDataProcessor:
    def __init__(self, csv_path, output_dir="./docs_processed"):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def clean_text(self, text):
        """Clean and format text content"""
        if pd.isna(text) or text == '':
            return "Không có thông tin"
        
        # Convert to string and clean
        text = str(text).strip()
        
        # Fix common formatting issues
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        
        return text
    
    def parse_tags(self, tags_str):
        """Parse tags string into list"""
        if pd.isna(tags_str) or tags_str == '':
            return []
        
        # Remove brackets and quotes, split by comma
        tags_str = str(tags_str).strip('[]"\'')
        tags = [tag.strip().strip('"\'') for tag in tags_str.split(',')]
        return [tag for tag in tags if tag]
    
    def create_safe_filename(self, university_name, index):
        """Create safe filename from university name"""
        # Remove/replace special characters
        safe_name = re.sub(r'[^\w\s-]', '', university_name)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        return f"{index+1:03d}_{safe_name}.md"
    
    def process_csv(self):
        """Main processing function"""
        print(f"🔄 Đang xử lý file CSV: {self.csv_path}")
        
        # Read CSV with proper encoding
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        
        print(f"📊 Tìm thấy {len(df)} trường đại học")
        
        # Create different output formats
        self.create_individual_files(df)
        self.create_comprehensive_file(df)
        self.create_search_optimized_files(df)
        
        print("✅ Hoàn thành chuyển đổi dữ liệu!")
        return df
    
    def create_individual_files(self, df):
        """Create individual markdown files for each university"""
        individual_dir = self.output_dir / "universities"
        individual_dir.mkdir(exist_ok=True)
        
        for idx, row in df.iterrows():
            university_name = self.clean_text(row.get('Tên trường', ''))
            description = self.clean_text(row.get('Mô tả trường', ''))
            changes_2025 = self.clean_text(row.get('Thay đổi 2025', ''))
            admission_info = self.clean_text(row.get('Thông tin trường', ''))
            programs = self.clean_text(row.get('Chương trình', ''))
            admission_scores = self.clean_text(row.get('Điểm chuẩn ', ''))  # Note the space
            tags = self.parse_tags(row.get('Tags', ''))
            
            filename = self.create_safe_filename(university_name, idx)
            
            content = f"""# {university_name}

## Thông tin tổng quan
{description}

## Những thay đổi và cập nhật năm 2025
{changes_2025}

## Thông tin tuyển sinh và xét tuyển
{admission_info}

## Các chương trình đào tạo và học phí
{programs}

## Điểm chuẩn tham khảo
{admission_scores}

## Từ khóa và đặc điểm nổi bật
{', '.join(tags) if tags else 'Không có thông tin'}

---

**Metadata:**
- **Loại trường:** {'Công lập' if 'Công lập' in tags else 'Tư thục' if 'Tư thục' in tags else 'Không xác định'}
- **Khu vực:** {self.extract_location(tags, university_name)}
- **Lĩnh vực:** {self.extract_field(tags, university_name)}
- **Năm cập nhật:** 2025
- **Nguồn:** Dữ liệu tuyển sinh chính thức

**Các từ khóa tìm kiếm:**
{', '.join(tags + [university_name.split()[0], university_name.split()[-1]] if len(university_name.split()) > 1 else tags + [university_name])}
"""
            
            with open(individual_dir / filename, "w", encoding="utf-8") as f:
                f.write(content)
    
    def create_comprehensive_file(self, df):
        """Create one comprehensive file with all universities"""
        content = """# Cẩm nang Tuyển sinh Đại học Việt Nam 2025

Tài liệu này tổng hợp thông tin chi tiết về các trường đại học tại Việt Nam, bao gồm thông tin tuyển sinh, chương trình đào tạo, điểm chuẩn và những thay đổi mới nhất năm 2025.

---

"""
        
        for idx, row in df.iterrows():
            university_name = self.clean_text(row.get('Tên trường', ''))
            description = self.clean_text(row.get('Mô tả trường', ''))
            changes_2025 = self.clean_text(row.get('Thay đổi 2025', ''))
            admission_info = self.clean_text(row.get('Thông tin trường', ''))
            programs = self.clean_text(row.get('Chương trình', ''))
            admission_scores = self.clean_text(row.get('Điểm chuẩn ', ''))
            tags = self.parse_tags(row.get('Tags', ''))
            
            content += f"""
## {idx + 1}. {university_name}

### 📋 Thông tin cơ bản
{description}

### 🔄 Cập nhật năm 2025
{changes_2025}

### 🎓 Tuyển sinh và xét tuyển
{admission_info}

### 📚 Chương trình đào tạo
{programs}

### 📊 Điểm chuẩn
{admission_scores}

### 🏷️ Đặc điểm nổi bật
- **Loại hình:** {'Công lập' if 'Công lập' in tags else 'Tư thục' if 'Tư thục' in tags else 'Không xác định'}
- **Lĩnh vực mạnh:** {', '.join([tag for tag in tags if tag not in ['Công lập', 'Tư thục']])}

---

"""
        
        with open(self.output_dir / "cam_nang_tuyen_sinh_2025.md", "w", encoding="utf-8") as f:
            f.write(content)
    
    def create_search_optimized_files(self, df):
        """Create search-optimized files by category"""
        search_dir = self.output_dir / "by_category"
        search_dir.mkdir(exist_ok=True)
        
        # Group by location
        self.create_location_files(df, search_dir)
        
        # Group by field
        self.create_field_files(df, search_dir)
        
        # Create admission scores summary
        self.create_scores_summary(df, search_dir)
    
    def create_location_files(self, df, search_dir):
        """Create files grouped by location"""
        location_groups = {}
        
        for idx, row in df.iterrows():
            university_name = self.clean_text(row.get('Tên trường', ''))
            tags = self.parse_tags(row.get('Tags', ''))
            location = self.extract_location(tags, university_name)
            
            if location not in location_groups:
                location_groups[location] = []
            location_groups[location].append((idx, row))
        
        for location, universities in location_groups.items():
            content = f"""# Các trường đại học tại {location}

Danh sách các trường đại học và thông tin tuyển sinh tại khu vực {location} năm 2025.

"""
            
            for idx, row in universities:
                university_name = self.clean_text(row.get('Tên trường', ''))
                description = self.clean_text(row.get('Mô tả trường', ''))
                admission_scores = self.clean_text(row.get('Điểm chuẩn ', ''))
                
                content += f"""
## {university_name}

{description[:300]}...

**Điểm chuẩn 2024:** {admission_scores.split('.')[0] if admission_scores != 'Không có thông tin' else 'Chưa công bố'}

---
"""
            
            safe_location = re.sub(r'[^\w\s-]', '', location).replace(' ', '_')
            with open(search_dir / f"truong_dai_hoc_{safe_location.lower()}.md", "w", encoding="utf-8") as f:
                f.write(content)
    
    def create_field_files(self, df, search_dir):
        """Create files grouped by field of study"""
        field_groups = {}
        
        for idx, row in df.iterrows():
            university_name = self.clean_text(row.get('Tên trường', ''))
            tags = self.parse_tags(row.get('Tags', ''))
            field = self.extract_field(tags, university_name)
            
            if field not in field_groups:
                field_groups[field] = []
            field_groups[field].append((idx, row))
        
        for field, universities in field_groups.items():
            content = f"""# Các trường đại học {field}

Danh sách các trường đại học chuyên về {field.lower()} và thông tin tuyển sinh năm 2025.

"""
            
            for idx, row in universities:
                university_name = self.clean_text(row.get('Tên trường', ''))
                programs = self.clean_text(row.get('Chương trình', ''))
                admission_scores = self.clean_text(row.get('Điểm chuẩn ', ''))
                
                content += f"""
## {university_name}

**Chương trình nổi bật:** {programs[:200]}...

**Điểm chuẩn:** {admission_scores.split('\n')[0] if admission_scores != 'Không có thông tin' else 'Chưa công bố'}

---
"""
            
            safe_field = re.sub(r'[^\w\s-]', '', field).replace(' ', '_')
            with open(search_dir / f"nganh_{safe_field.lower()}.md", "w", encoding="utf-8") as f:
                f.write(content)
    
    def create_scores_summary(self, df, search_dir):
        """Create admission scores summary"""
        content = """# Tổng hợp Điểm chuẩn Đại học 2024

Bảng tổng hợp điểm chuẩn các trường đại học năm 2024 để tham khảo cho kỳ tuyển sinh 2025.

"""
        
        scores_data = []
        
        for idx, row in df.iterrows():
            university_name = self.clean_text(row.get('Tên trường', ''))
            admission_scores = self.clean_text(row.get('Điểm chuẩn ', ''))
            tags = self.parse_tags(row.get('Tags', ''))
            
            # Extract highest score for sorting
            score_numbers = re.findall(r'(\d+[,.]?\d*)', admission_scores)
            highest_score = 0
            if score_numbers:
                try:
                    highest_score = max([float(s.replace(',', '.')) for s in score_numbers])
                except:
                    highest_score = 0
            
            scores_data.append({
                'name': university_name,
                'scores': admission_scores,
                'highest': highest_score,
                'field': self.extract_field(tags, university_name)
            })
        
        # Sort by highest score
        scores_data.sort(key=lambda x: x['highest'], reverse=True)
        
        for data in scores_data:
            content += f"""
## {data['name']}
- **Lĩnh vực:** {data['field']}
- **Điểm chuẩn cao nhất:** {data['highest']} điểm
- **Chi tiết:** {data['scores'][:150]}...

---
"""
        
        with open(search_dir / "diem_chuan_2024.md", "w", encoding="utf-8") as f:
            f.write(content)
    
    def extract_location(self, tags, university_name):
        """Extract location from tags or university name"""
        location_keywords = {
            'Hồ Chí Minh': ['Hồ Chí Minh', 'TP.HCM', 'TPHCM', 'Sài Gòn'],
            'Hà Nội': ['Hà Nội', 'Hanoi'],
            'Đà Nẵng': ['Đà Nẵng', 'Da Nang'],
            'Cần Thơ': ['Cần Thơ', 'Can Tho'],
            'Huế': ['Huế', 'Hue'],
        }
        
        # Check tags first
        for location, keywords in location_keywords.items():
            for keyword in keywords:
                if any(keyword.lower() in tag.lower() for tag in tags):
                    return location
        
        # Check university name
        for location, keywords in location_keywords.items():
            for keyword in keywords:
                if keyword.lower() in university_name.lower():
                    return location
        
        return 'Các tỉnh thành khác'
    
    def extract_field(self, tags, university_name):
        """Extract field of study from tags or university name"""
        field_keywords = {
            'Công nghệ - Kỹ thuật': ['Công nghệ', 'Kỹ thuật', 'Công nghiệp', 'Điện', 'Cơ khí'],
            'Y - Dược': ['Y khoa', 'Y tế', 'Dược', 'Y học'],
            'Kinh tế - Quản trị': ['Kinh tế', 'Quản trị', 'Tài chính', 'Ngân hàng'],
            'Sư phạm - Giáo dục': ['Sư phạm', 'Giáo dục', 'Đào tạo'],
            'Nông - Lâm - Ngư': ['Nông nghiệp', 'Lâm nghiệp', 'Thủy sản'],
        }
        
        # Check tags first
        for field, keywords in field_keywords.items():
            for keyword in keywords:
                if any(keyword.lower() in tag.lower() for tag in tags):
                    return field
        
        # Check university name
        for field, keywords in field_keywords.items():
            for keyword in keywords:
                if keyword.lower() in university_name.lower():
                    return field
        
        return 'Đa ngành'

# Usage
if __name__ == "__main__":
    # Initialize processor
    processor = UniversityDataProcessor("../notebook/Trường-Grid view (2).csv")  # Update path as needed
    
    # Process the CSV
    df = processor.process_csv()
    
    print(f"\n📁 Các file đã được tạo trong thư mục: {processor.output_dir}")
    print("📂 Cấu trúc thư mục:")
    print("   ├── universities/           # File riêng cho từng trường")
    print("   ├── by_category/           # File theo danh mục")
    print("   └── cam_nang_tuyen_sinh_2025.md  # File tổng hợp")