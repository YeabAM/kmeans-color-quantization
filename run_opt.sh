#!/bin/bash
#SBATCH --account=project_####
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --job-name=kmeans_opt
#SBATCH --output=opt_results.txt

module load gcc/11.3.0
module load cuda/11.7.0

# Resolution scaling (K=8)
./kmeans_optimized data/images/nature_512x512.png   8 data/results/opt_512.png
./kmeans_optimized data/images/nature_1024x1024.png 8 data/results/opt_1024.png
./kmeans_optimized data/images/nature_1920x1080.png 8 data/results/opt_1080.png
./kmeans_optimized data/images/nature_3840x2160.png 8 data/results/opt_4k.png

# K scaling (1920×1080)
./kmeans_optimized data/images/nature_1920x1080.png 16 data/results/opt_k16.png
./kmeans_optimized data/images/nature_1920x1080.png 32 data/results/opt_k32.png
./kmeans_optimized data/images/nature_1920x1080.png 64 data/results/opt_k64.png

# Block-size experiment (1920×1080, K=8)
./kmeans_optimized data/images/nature_1920x1080.png 8 data/results/opt_b128.png 128
./kmeans_optimized data/images/nature_1920x1080.png 8 data/results/opt_b512.png 512
