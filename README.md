# GPU-Accelerated K-Means Image Color Quantization

A parallel computing project that reduces the number of colors in an image using the K-means clustering algorithm. We implement and benchmark multiple versions — from a sequential CPU baseline to optimized GPU kernels — to explore how parallelism accelerates this computationally intensive task.

## Team Members
- Fawaz Abdulsalam (2501240)
- Nasif Sarwar (2502396)
- Ossama Essfadi (2501238)
- Srishti Karanth (2502524)
- Yeabkalu Merkebe (2501239) — CPU sequential baseline, image I/O, K-means++ init

## Project Structure

```
├── src/
│   ├── common/
│   │   ├── image_io.h / .c        # Image loading, saving, palette and comparison utilities
│   │   ├── kmeans_init.h / .c     # K-means++ initialization
│   │   ├── stb_image.h            # Single-header image decoding library
│   │   └── stb_image_write.h      # Single-header image encoding library
│   └── cpu/
│       ├── kmeans_cpu.h / .c      # CPU sequential K-means baseline
│
├── benchmarks/
│   ├── run_benchmarks.py          # Benchmark runner (works with any implementation)
│   └── run_benchmark.slurm        # SLURM wrapper for running benchmarks on CSC Mahti
│
├── data/
│   ├── images/                    # Test images at various resolutions
│   └── results/                   # Output images, palettes, and benchmark CSVs
│
└── Makefile
```

## Build

```bash
make kmeans_cpu
```

## Usage

### Running K-means (CPU sequential)

```bash
./kmeans_cpu <input_image> <K> <output_image>
```

Example:
```bash
./kmeans_cpu data/images/nature_512x512.png 8 data/results/nature_8colors.png
```

This will:
- Run K-means with K colors on the input image
- Print per-iteration timing breakdown (assignment vs update)
- Save the quantized image, a color palette, and a side-by-side comparison to `data/results/`
- Print a summary with total timing and file size comparison

### Benchmarking

The benchmark runner loops through a set of test images and K values, runs the given binary on each combination, and saves all timings to a CSV in `data/results/`.

It works with any implementation as long as the binary follows the same interface: `./binary <input> <K> <output>`.

**Run directly:**
```bash
python3 benchmarks/run_benchmarks.py ./kmeans_cpu cpu_sequential
```

**Run on CSC Mahti via SLURM:**

First update `benchmarks/run_benchmark.slurm` with your CSC project account, then:
```bash
sbatch benchmarks/run_benchmark.slurm ./kmeans_cpu cpu_sequential
```

Output logs go to `benchmarks/slurm_<job_id>.out`. Timing results go to `data/results/<label>_timings.csv`.