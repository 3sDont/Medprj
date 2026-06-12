# 1. How to finetuning simcprs 
```
!python3 /Users/macbook/DSProjects/MedPRS/src/model_for_cl.py \
    --working_path  "/Users/macbook/MedPRS/Model4CL/" \
    --data_path  "/Users/macbook/DSProjects/MedPRS/data/preprocessed_data/train_pairs_test.csv" \
    --model_name  "dmis-lab/biobert-v1.1" \
    --batch_size  8 \
    --max_len  50 \
    --pooler_type  "cls_before_pooler" \
    --lr  5e-5 \
    --num_epoch  2 \
    --device mps
```
# 2. How to train SimCPRS 
```
!python3 /workspace/MedPRS/src/main_simcprs_v1.py \
    --working_path /workspace/MedPRS/ \
    --data_path /workspace/MedPRS/data/preprocessed_data/ \
    --model_name dmis-lab/biobert-v1.1 \
    --checkpoint_path workspace/MedPRS/Model4CL/checkpoint/saved_model/Epoch_05_SupCL_dmis-lab_biobert-v1_1.pth \
    --batch_size 64 \
    --pooler_type cls_before_pooler \
    --lr 5e-5 \
    --num_epoch 7 \
    --features TAK \
    --max_len 512 \
    --use_aim \
    --saved_folder SIMCPRS
```
Use --checkpoint_path if you want to load a fine-tuned model trained with contrastive learning; otherwise, skip this option.
Use --use_aim if you want to train the model using the journal's Aim; otherwise, skip this option.

# 3. How to evaluate with mrr, ndcg@k
```
!python compute_metrics.py --input_csv test_detailed_predictions.csv --ks 1 3 5 10
```

# 4. How to inference 
Run full
```
python inference.py `
  --checkpoint_path "đường_dẫn_tới_file.pth" `
  --model_name "roberta-base" `
  --data_path "data/preprocessed_data/" `
  --aims_csv "01_aims.csv" `
  --input_csv "data/preprocessed_data/01_test.csv" `
  --features TAK `
  --use_aim `
  --max_len 64 `
  --topk 10 `
  --output_csv "pred_all.csv"

```


```
# Account 1
python inference.py `
  --checkpoint_path "file.pth" --model_name "roberta-base" `
  --data_path "data/preprocessed_data/" --aims_csv "01_aims.csv" `
  --input_csv "data/preprocessed_data/01_test.csv" `
  --start_idx 0 --end_idx 100000 `
  --features TAK --use_aim --topk 10 `
  --output_csv "pred_0_100k.csv"

# Account 2
python inference.py `
  ... --start_idx 100000 --end_idx 200000 --output_csv "pred_100k_200k.csv"

# Account 3
python inference.py `
  ... --start_idx 200000 --output_csv "pred_200k_end.csv"
```


