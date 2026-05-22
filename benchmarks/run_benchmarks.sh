#!/bin/bash
#SBATCH --job-name=kmeans_bench
#SBATCH --account=project_2019091
#SBATCH --partition=test
#SBATCH --time=01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=benchmarks/slurm_%j.out

# Usage: sbatch benchmarks/run_benchmark.slurm <binary> <label>
# Example: sbatch benchmarks/run_benchmark.slurm ./kmeans_cpu cpu_sequential

BINARY=${1:-./kmeans_cpu}
LABEL=${2:-cpu_sequential}

module load python-data/3.12-25.09

echo "=== Benchmark: $LABEL ==="
echo "Binary: $BINARY"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

python3 benchmarks/run_benchmarks.py "$BINARY" "$LABEL"