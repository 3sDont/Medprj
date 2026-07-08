# MedPRS — Medical Paper Journal Recommendation System

Hệ thống gợi ý tạp chí khoa học phù hợp dựa trên Title + Abstract + Keywords của bài báo, gồm 4 bước: classifier → similarity → LLM feature extraction/coverage → rerank + giải thích tiếng Anh.

Giao diện chính là một **Streamlit app** (`app.py`). Từng bước cũng có thể chạy độc lập qua dòng lệnh (xem mục 5).

---

## Cấu trúc thư mục

```text
Medprj/
├── app.py                       # Streamlit UI — điểm vào chính
├── src/
│   ├── inference.py              # Bước 1: Classifier (BioBERT) → top-N journals
│   ├── aims_scope_sim.py         # Bước 2: Aims_Scope_Sim (SPECTER2)
│   ├── llm_extract.py            # Bước 3: Qwen trích xuất features + coverage metrics
│   ├── prompt_builder_paper.py   # Prompt template cho Qwen extraction
│   ├── reasoning.py               # Bước 4: Rerank (LTR model) + giải thích (Qwen)
│   └── pipeline.py                # load_pipeline()/run_pipeline() — dùng bởi app.py và CLI
├── data/
│   ├── journal_category.csv      # Danh sách tạp chí (Journal, Aims, Categories, Best Quartile...)
│   └── journal_extract.json      # Features đã trích xuất sẵn của từng tạp chí
├── models/
│   ├── student_model.json        # Trọng số LTR đã train, dùng để tính final_fit_score
│   └── Epoch_02_SIMCPRS_*.pth    # Checkpoint classifier (không nằm trong git, xem mục 1)
├── outputs/
│   ├── result.json                # Kết quả lần chạy gần nhất từ app.py
│   └── cache/                     # Cache embeddings aims (BioBERT/SPECTER2) để load nhanh hơn
├── requirements.txt
└── setup_env.bat
```

---

## 1. Cài đặt

**Chạy một lần duy nhất:**

```bat
setup_env.bat
```

Script này tạo virtual environment `.venv` và cài dependencies từ `requirements.txt` (bản CPU của PyTorch). Nếu máy có GPU NVIDIA, cài `torch`/`torchvision` bản CUDA phù hợp **trước** theo hướng dẫn tại [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), rồi mới chạy `pip install -r requirements.txt`.

**Mỗi lần mở terminal mới:**

```powershell
.venv\Scripts\activate.bat
```

**Cấu hình VS Code** (nếu dùng): `Ctrl+Shift+P` → `Python: Select Interpreter` → chọn `.venv\Scripts\python.exe`

**Checkpoint classifier:** `models/Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth` (~1.3GB) quá lớn để đưa lên GitHub nên không nằm trong repo (`.gitignore`). Lấy file này từ nơi lưu trữ nội bộ/nhóm và đặt đúng đường dẫn trên trước khi chạy app.

---

## 2. Chạy ứng dụng (Streamlit UI — khuyến nghị)

```powershell
streamlit run app.py
```

Luồng sử dụng:

1. Ở sidebar, kiểm tra/điều chỉnh các đường dẫn model (checkpoint, data folder, SPECTER2, Qwen, LTR model) rồi bấm **Load Models** (lần đầu sẽ tải các model từ HuggingFace nếu chưa có trong cache — xem mục 6).
2. Nhập Title / Abstract / Keywords của bài báo.
3. Bấm **Analyze** — pipeline chạy qua 4 bước, kết quả được lưu vào `outputs/result.json` và hiển thị trực tiếp trên UI.
4. Có thể bấm **Load last result** ở sidebar để xem lại `outputs/result.json` mà không cần chạy lại pipeline.

---

## 3. Chạy toàn bộ pipeline qua dòng lệnh

Tương đương app.py nhưng không cần UI, dùng `src/pipeline.py`:

```powershell
python src\pipeline.py `
  --checkpoint_path "models\Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth" `
  --model_name "dmis-lab/biobert-v1.1" `
  --pooler_type "cls" `
  --data_path "data" `
  --aims_csv "journal_category.csv" `
  --journal_extract_jsonl "journal_extract.json" `
  --encoder_model "allenai/specter2_base" `
  --qwen_model "Qwen/Qwen3.5-2B" `
  --ltr_model_path "models\student_model.json" `
  --title "YOUR PAPER TITLE" `
  --abstract "YOUR ABSTRACT TEXT" `
  --keywords "keyword1, keyword2, keyword3" `
  --use_aim `
  --features "TAK" `
  --topk 10 `
  --top_n_explain 10 `
  --output_json "result.json"
```

**Kết quả:** File `result.json` chứa top journals gợi ý với đầy đủ scores, metrics và giải thích.

> **Lần đầu chạy:** Tự động download BioBERT (~440MB), SPECTER2 (~440MB), Qwen3.5-2B (~4.5GB) từ HuggingFace. Các lần sau dùng cache HuggingFace (`.cache/`) và cache embeddings (`outputs/cache/`).

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
  "Top10_journals": [
    {
      "Label": "1340",
      "Name": "Neurorehabilitation and Neural Repair",
      "Aims": "...",
      "Categories": ["Neurology (clinical)", "Rehabilitation"],
      "Best_Quartile": "Q1",
      "Rank": 2,
      "Base_Score": 0.82,
      "Aims_Scope_Sim": 0.92,
      "extracted_journal_features": { "sci_evi": ["..."], "research_evi": ["..."] },
      "coverage_metrics": { "scientific_domains_coverage": 0.91, "missing_coverage": [] },
      "Rerank": {
        "final_fit_score": 92.5,
        "new_rank": 1,
        "feature_contributions": [ { "feature": "Aims_Scope_Sim", "contribution": 0.31 } ]
      },
      "Explanation": {
        "header": "Neurorehabilitation and Neural Repair is considered a strong match for this paper because ...",
        "score_breakdown": [
          { "key": "aims_scope", "label": "...", "value_pct": 92, "explanation": "..." }
        ]
      }
    }
  ]
}
```

> Tên khóa `Top10_journals` là cố định trong code hiện tại kể cả khi `--topk`/`--top_n_explain` khác 10.

---

## 5. Chạy từng bước riêng lẻ (nâng cao)

Dữ liệu truyền qua từng bước dưới dạng JSON file.

### Bước 1 — `inference.py` (Classifier → top-K)

```powershell
python src\inference.py `
  --checkpoint_path "models\Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth" `
  --model_name "dmis-lab/biobert-v1.1" `
  --pooler_type "cls" `
  --data_path "data" `
  --aims_csv "journal_category.csv" `
  --title "YOUR PAPER TITLE" `
  --abstract "YOUR ABSTRACT TEXT" `
  --keywords "keyword1, keyword2, keyword3" `
  --use_aim `
  --features "TAK" `
  --topk 10 `
  --output_json "inference_result.json"
```

### Bước 2 — `aims_scope_sim.py` (Tính độ tương đồng Aims/Scope)

```powershell
python src\aims_scope_sim.py `
  --inference_json "inference_result.json" `
  --data_path "data" `
  --aims_csv "journal_category.csv" `
  --encoder_model "allenai/specter2_base" `
  --features "TAK" `
  --output_json "inference_with_sim.json"
```

Thêm `Aims_Scope_Sim` vào mỗi journal (cosine similarity paper ↔ aims text, giá trị 0–1).

### Bước 3 — `llm_extract.py` (Qwen trích xuất features + coverage)

```powershell
python src\llm_extract.py `
  --inference_json "inference_with_sim.json" `
  --journal_extract_jsonl "data\journal_extract.json" `
  --qwen_model "Qwen/Qwen3.5-2B" `
  --output_json "inference_with_coverage.json"
```

Thêm vào mỗi journal `extracted_journal_features` (domains/research focuses của tạp chí) và `coverage_metrics` (các chỉ số bao phủ giữa bài báo và tạp chí, cùng `missing_coverage`).

### Bước 4 — `reasoning.py` (Rerank + Giải thích)

```powershell
python src\reasoning.py `
  --inference_json "inference_with_coverage.json" `
  --qwen_model "Qwen/Qwen3.5-2B" `
  --ltr_model_path "models\student_model.json" `
  --top_n 20 `
  --output_json "final_result.json"
```

Rerank theo `final_fit_score` (tính từ model LTR `models/student_model.json`; nếu không tìm thấy file này sẽ fallback về `Base_Score` thô) và sinh giải thích cho top N journals.

---

## 6. Tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--model_name` | `roberta-base` | Base model của classifier — phải khớp với checkpoint |
| `--use_aim` | tắt | Bật classifier kiểu WithAim (dùng nếu checkpoint được train với aims) |
| `--features` | `TAK` | Kết hợp features: T=Title, A=Abstract, K=Keywords |
| `--topk` | `10` | Số journals trả về từ classifier |
| `--top_n_explain` (`pipeline.py`) / `--top_n` (`reasoning.py`) | `10` / `20` | Số journals được sinh giải thích (ít hơn = nhanh hơn) |
| `--qwen_model` | `Qwen/Qwen3.5-2B` | Có thể dùng `Qwen/Qwen3.5-0.6B` để nhanh hơn trên CPU |
| `--ltr_model_path` | `models/student_model.json` | Model LTR cho `final_fit_score`; bỏ qua hoặc trỏ tới file không tồn tại để dùng `Base_Score` thô |

---

## 7. Lưu ý

- **Lần đầu chạy** sẽ download vài GB model weights từ HuggingFace — cần kết nối internet ổn định.
- **CPU**: Qwen 2B chạy khá chậm; dùng `Qwen/Qwen3.5-0.6B` (qua tham số `--qwen_model` / sidebar "Qwen model") để nhanh hơn.
- **GPU**: Cần cài `torch` bản CUDA đúng version (xem mục 1); ứng dụng tự phát hiện và dùng GPU nếu có (`torch.cuda.is_available()`).
- `outputs/cache/` lưu embeddings Aims đã tính sẵn (BioBERT + SPECTER2) để load nhanh hơn ở các lần chạy sau — có thể xóa an toàn nếu cần tính lại.
- Các `[ERROR]`/warning từ `transformers` liên quan đến Qwen là warning nội bộ của thư viện, không ảnh hưởng kết quả.
