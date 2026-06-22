# MEDPRS — Medical Paper Journal Recommendation System

Hệ thống gợi ý tạp chí khoa học phù hợp dựa trên Title + Abstract + Keywords của bài báo.

---

## Cấu trúc thư mục

```
MEDPRS_/
├── data/
│   ├── journal_category.csv          # Danh sách 1406 tạp chí (Journal, Aims, Label, Categories)
│   ├── journal_extract.jsonl         # Features đã trích xuất sẵn của từng tạp chí
│   └── Epoch_02_SIMCPRS_*.pth        # Checkpoint classifier model
├── src/
│   ├── inference.py                  # Bước 1: Classifier → top-20 journals
│   ├── aims_scope_sim.py             # Bước 2: Tính Aims_Scope_Sim (SPECTER2)
│   ├── llm_extract.py                # Bước 3: Qwen trích xuất features + coverage metrics
│   ├── prompt_builder_paper.py       # Prompt template cho Qwen extraction
│   ├── reasoning.py                  # Bước 4: Rerank + giải thích tiếng Việt (Qwen)
│   └── pipeline.py                   # Chạy toàn bộ 4 bước trong 1 lệnh
├── requirements.txt
└── setup_env.bat
```

---

## 1. Cài đặt

**Chạy một lần duy nhất:**

```bat
setup_env.bat
```

Lệnh này tạo virtual environment `.venv` và cài tất cả dependencies.

**Mỗi lần mở terminal mới:**

```powershell
.venv\Scripts\activate.bat
```

**Cấu hình VS Code** (nếu dùng): `Ctrl+Shift+P` → `Python: Select Interpreter` → chọn `.venv\Scripts\python.exe`

---

## 2. Chạy toàn bộ pipeline (khuyến nghị)

```powershell
python src\pipeline.py `
  --checkpoint_path "data\Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth" `
  --model_name "dmis-lab/biobert-v1.1" `
  --pooler_type "cls" `
  --data_path "data" `
  --aims_csv "journal_category.csv" `
  --journal_extract_jsonl "journal_extract.jsonl" `
  --encoder_model "allenai/specter2_base" `
  --qwen_model "Qwen/Qwen3.5-2B" `
  --title "YOUR PAPER TITLE" `
  --abstract "YOUR ABSTRACT TEXT" `
  --keywords "keyword1, keyword2, keyword3" `
  --use_aim `
  --features "TAK" `
  --topk 20 `
  --top_n_explain 20 `
  --output_json "result.json"
```

**Kết quả:** File `result.json` chứa top-20 tạp chí gợi ý với đầy đủ scores, metrics và giải thích tiếng Việt.

> **Lần đầu chạy:** Tự động download BioBERT (~440MB), SPECTER2 (~440MB), Qwen3.5-2B (~4.5GB) từ HuggingFace. Các lần sau dùng cache.

---

## 3. Chạy từng bước riêng lẻ

Dữ liệu truyền qua từng bước dưới dạng JSON file.

### Bước 1 — `inference.py` (Classifier → top-20)

```powershell
python src\inference.py `
  --checkpoint_path "data\Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth" `
  --model_name "dmis-lab/biobert-v1.1" `
  --pooler_type "cls" `
  --data_path "data" `
  --aims_csv "journal_category.csv" `
  --title "YOUR PAPER TITLE" `
  --abstract "YOUR ABSTRACT TEXT" `
  --keywords "keyword1, keyword2, keyword3" `
  --use_aim `
  --features "TAK" `
  --topk 20 `
  --output_json "step1_inference.json"
```

**Kết quả `step1_inference.json`:**
```json
{
  "paper": { "title": "...", "abstract": "...", "keywords": "..." },
  "top_journals": [
    { "journal_idx": 42, "Label": "1340", "Name": "...", "Aims": "...",
      "Rank": 1, "Base_Score": 0.85 },
    ...
  ]
}
```

---

### Bước 2 — `aims_scope_sim.py` (Tính độ tương đồng Aims/Scope)

```powershell
python src\aims_scope_sim.py `
  --inference_json "step1_inference.json" `
  --data_path "data" `
  --aims_csv "journal_category.csv" `
  --encoder_model "allenai/specter2_base" `
  --features "TAK" `
  --output_json "step2_sim.json"
```

**Kết quả `step2_sim.json`:** Thêm `Aims_Scope_Sim` vào mỗi journal (cosine similarity paper ↔ aims text, giá trị 0–1).

```json
{ "journal_idx": 42, "Name": "...", "Rank": 1, "Base_Score": 0.85,
  "Aims_Scope_Sim": 0.72 }
```

---

### Bước 3 — `llm_extract.py` (Qwen trích xuất features + coverage)

```powershell
python src\llm_extract.py `
  --inference_json "step2_sim.json" `
  --journal_extract_jsonl "data\journal_extract.jsonl" `
  --qwen_model "Qwen/Qwen3.5-2B" `
  --output_json "step3_coverage.json"
```

**Kết quả `step3_coverage.json`:** Thêm vào mỗi journal:
- `extracted_journal_features`: danh sách domains và research focuses của tạp chí
- `coverage_metrics`: 6 chỉ số đo mức độ bao phủ giữa bài báo và tạp chí, và `missing_coverage` (những khía cạnh của bài báo chưa được tạp chí bao phủ)

```json
{
  "extracted_journal_features": { "sci_evi": ["Neurology", "Stroke"], "research_evi": ["..."] },
  "coverage_metrics": {
    "scientific_domains_category_coverage": 0.85,
    "scientific_domains_coverage": 0.91,
    "scientific_domains_aimscope": 0.88,
    "research_focuses_category_coverage": 0.66,
    "research_focuses_coverage_aimscope": 0.75,
    "missing_coverage": ["Explainable AI"]
  }
}
```

---

### Bước 4 — `reasoning.py` (Rerank + Giải thích tiếng Việt)

```powershell
python src\reasoning.py `
  --inference_json "step3_coverage.json" `
  --qwen_model "Qwen/Qwen3.5-2B" `
  --top_n 20 `
  --output_json "step4_final.json"
```

**Kết quả `step4_final.json`:** File JSON hoàn chỉnh — các tạp chí được rerank theo `final_fit_score` và có thêm giải thích tiếng Việt:

```json
{
  "Rerank": { "final_fit_score": 92.5, "new_rank": 1 },
  "Explanation": {
    "main_reasoning": "Tạp chí phù hợp vì...",
    "reranking_reasons": "Vượt lên hạng 1 nhờ...",
    "weakness_warning": "Phương pháp X ít được đề cập..."
  }
}
```

---

## 4. Output JSON cuối cùng

```json
{
  "paper_id": "req_abc12345",
  "paper_information": {
    "inputs": { "T": "...", "A": "...", "K": ["..."] },
    "extracted_paper_features": {
      "sci_evidence":      { "1": ["Neurology", "Stroke"], "2": ["Rehabilitation"] },
      "research_evidence": { "1": ["Deep Learning", "Neural Networks"], "2": ["Clinical Prediction"] }
    }
  },
  "Top20_journals": [
    {
      "Label": "1340",
      "Name": "Neurorehabilitation and Neural Repair",
      "Aims": "...",
      "Categories": ["Neurology (clinical)", "Rehabilitation"],
      "Rank": 2,
      "Base_Score": 0.82,
      "Aims_Scope_Sim": 0.92,
      "extracted_journal_features": { "sci_evi": [...], "research_evi": [...] },
      "coverage_metrics": { "scientific_domains_coverage": 0.91, "missing_coverage": [] },
      "Rerank": { "final_fit_score": 92.5, "new_rank": 1 },
      "Explanation": {
        "main_reasoning": "...",
        "reranking_reasons": "...",
        "weakness_warning": "..."
      }
    }
  ]
}
```

---

## 5. Tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--model_name` | `roberta-base` | Base model của classifier — phải khớp với checkpoint |
| `--use_aim` | tắt | Bật classifier kiểu WithAim (dùng nếu checkpoint được train với aims) |
| `--features` | `TAK` | Kết hợp features: T=Title, A=Abstract, K=Keywords |
| `--topk` | `20` | Số journals trả về từ classifier |
| `--top_n_explain` | `20` | Số journals được sinh giải thích (ít hơn = nhanh hơn) |
| `--qwen_model` | `Qwen/Qwen3.5-2B` | Có thể dùng `Qwen/Qwen3.5-0.6B` để nhanh hơn trên CPU |

---

## 6. Lưu ý

- **Lần đầu chạy** sẽ download ~5.5GB model weights — cần kết nối internet ổn định
- **CPU**: Qwen 2B mất ~3-5 phút/paper; dùng `Qwen3.5-0.6B` để nhanh hơn
- **GPU**: Đặt `dtype=float16` trong code hoặc dùng `--qwen_model` nhỏ hơn
- Các `[ERROR]` từ transformers liên quan đến Qwen3.5 là **warning nội bộ**, không ảnh hưởng kết quả
