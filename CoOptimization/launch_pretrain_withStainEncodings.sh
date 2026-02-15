#!/bin/bash
## Global and Local loss with stain encodings pretraining 4 deivced and batch__zise=40 and n_subsample=2048 ###
python bin/pretrain.py \
    --data_root_dir # pt_file paths 
    --results_dir results \
    --batch_size 40\
    --dataset HCC \
    --csv_fpath HCC_256_v2_800.csv \
    --cohort 256 \
    --n_subsamples 2048 \
    --num_workers 4 \
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
    --precision bfloat16