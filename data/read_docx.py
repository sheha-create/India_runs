import os
import zipfile
import xml.etree.ElementTree as ET

def docx_to_txt(docx_path, txt_path):
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} does not exist")
        return
    
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Namespaces
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(paragraphs))
            print(f"Success: Wrote {len(paragraphs)} paragraphs from {os.path.basename(docx_path)} to {os.path.basename(txt_path)}")
    except Exception as e:
        print(f"Error reading {docx_path}: {e}")

# Paths
base_dir = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
scratch_dir = r"c:\Users\Home\redrob\data"

docx_to_txt(os.path.join(base_dir, "submission_spec.docx"), os.path.join(scratch_dir, "submission_spec.txt"))
docx_to_txt(os.path.join(base_dir, "README.docx"), os.path.join(scratch_dir, "README.txt"))
docx_to_txt(os.path.join(base_dir, "job_description.docx"), os.path.join(scratch_dir, "job_description.txt"))
docx_to_txt(os.path.join(base_dir, "redrob_signals_doc.docx"), os.path.join(scratch_dir, "redrob_signals_doc.txt"))
