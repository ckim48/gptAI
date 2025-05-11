import json

input_path = "pdf_output/ocr_data.jsonl"
output_path = "pdf_output/gpt_finetune.jsonl"

with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        e = json.loads(line)
        prompt = f"Tell a story like the one on page {e['page']} of {e['source_pdf']}"
        completion = e['text'].strip()

        if len(completion) < 20:
            continue

        formatted = {
            "messages": [
                {"role":"user", "content":prompt},
                {"role":"assistant","content":completion}
            ]
        }
        json.dump(formatted,f_out,ensure_ascii=False)
        f_out.write("\n")