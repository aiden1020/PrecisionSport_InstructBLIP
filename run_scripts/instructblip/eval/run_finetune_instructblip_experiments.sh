#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=output/torch
export HUGGINGFACE_HUB_CACHE=output/huggingface

# The directory to copy from
dir=./output/results/eval
# The directory to copy to
dest=./input/results/eval
export CUDA_VISIBLE_DEVICES=0
mkdir -p $dir
mkdir -p $dest
touch $dir/eval.log
nohup python3 -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=$((RANDOM%50000+10000)) \
    evaluate.py --cfg-path lavis/projects/instructblip/eval/finetune_instructblip_badminton_caption_3.yaml \
    2>&1 | tee $dir/eval.log

rsync -av --no-o --no-g --chmod=777 --exclude='*.pth' $dir/ $dest/
