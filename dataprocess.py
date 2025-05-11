# 1. Convert each PDF page to an image
# 2. Runs OCR(detecting text from the image) on each image
# 3. Saves the extracted data in .jsonl format.
from pdf2image import convert_from_path
import pytesseract
import os
import json

pdf_paths = ["1.pdf","2.pdf"]
output_dir = 'pdf_output'
image_dir = os.path.join(output_dir, "images") # pdf_output/images
jsonl_path = os.path.join(output_dir,"ocr_data.jsonl")
os.makedirs(image_dir, exist_ok=True)

# OCR extraction
def extract_ocr_data(pdf_path, prefix):
    print(f"Processing {pdf_path}")
    images = convert_from_path(pdf_path, dpi=200)
    data = []
    for i, img in enumerate(images):
        # Saving images extracted from pdf file
        page_num = i+1
        image_file = f"{prefix}_page_{page_num}.png"
        image_path = os.path.join(image_dir, image_file)
        img.save(image_path)

        text = pytesseract.image_to_string(img).strip()
        data.append({
            "page": page_num,
            "source_pdf": f"{prefix}.pdf",
            "image_path": image_file,
            "text":text
        })
    return data

# Process PDFs
all_data = []
for path in pdf_paths:
    prefix = os.path.splitext(os.path.basename(path))[0]
    all_data.extend(extract_ocr_data(path,prefix))

with open(jsonl_path, "w", encoding="utf-8") as f:
    for e in all_data:
        json.dump(e, f, ensure_ascii=False)
        f.write("\n")
print("Done")