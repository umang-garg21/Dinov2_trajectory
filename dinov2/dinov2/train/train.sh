#!/bin/bash
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=6,7
# Define variables with proper names and paths
$config_file=vit16_short_traj.yaml
$output_dir=/data/home/umang/Dinov2_trajectory/dinov2_outputs

# Run the Python script with the provided arguments
python train.py 
