#!/bin/bash

for experiment_id in {1..6}
do
    ./run_scripts/instructblip/train/run_finetune_instructblip_experiments.sh badminton_caption $experiment_id
done
