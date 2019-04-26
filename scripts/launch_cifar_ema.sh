#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--shuffle --batch_size 500 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--use_fid_loss True --fid_weight 0.01 \
--test_every 500 --save_every 200 --num_best_copies 5 --num_save_copies 2 --seed 0
