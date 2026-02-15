#!/bin/bash
## Global and Local loss with stain encodings pretraining ###
python bin/inference.py \
    --data_root_dir # ptfile paths 
    --results_dir results\
    --batch_size 1\
    --dataset HCC \
    --csv_fpath HCC_256_v2_800.csv \
    --cohort 256 \
    --n_subsamples -1 \
    --num_workers 1 \
    --global_loss "info-nce" \
    --local_loss "got" \
    --intra_modality_loss "info-nce"\
    --patch_embedding_dim 1536 \
    --wsi_encoder abmil \
    --n_heads 4 \
    --wsi_encoder_hidden_dim 512 \
    --local_loss_weight 1.0 \
    --temperature 0.001 \
    --lr 0.0001 \
    --max_epochs 120 \
    --num_gpus 5 \
    --opt adamW \
    --activation softmax \
    --warmup_epochs 5 \
    --warmup \
    --symmetric_cl \
    --add_stain_encoding \
    --precision bfloat16\