# How to run model for CL
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
