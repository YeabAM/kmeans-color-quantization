"""
Benchmark runner for all K-means implementations.

Usage:
    python3 benchmarks/run_benchmarks.py <binary> <label>

Examples:
    python3 benchmarks/run_benchmarks.py ./kmeans_cpu cpu_sequential
    python3 benchmarks/run_benchmarks.py ./kmeans_ispc ispc
    python3 benchmarks/run_benchmarks.py ./kmeans_naive naive_cuda
    python3 benchmarks/run_benchmarks.py ./kmeans_optimized optimized_cuda
"""

import subprocess
import csv
import sys
import re
import os

IMAGES = [
    "data/images/nature_512x512.png",
    "data/images/colorful_1024x1024.png",
    "data/images/portrait_1920x1080.png",
    "data/images/urbn_3840x2160.png",
]

K_VALUES = [4, 8, 16, 32, 64, 128, 256]


def parse_output(output):
    """Extract metrics from the program's summary output."""
    def find(pattern):
        match = re.search(pattern, output)
        return match.group(1) if match else ""

    return {
        "init_time":      find(r"Init time:\s+([\d.]+)s"),
        "kmeans_time":    find(r"K-means time:\s+([\d.]+)s"),
        "total_time":     find(r"Total time:\s+([\d.]+)s"),
        "input_kb":       find(r"Input file:\s+([\d.]+) KB"),
        "output_kb":      find(r"Output file:\s+([\d.]+) KB"),
        "size_reduction": find(r"Size reduction:\s+([\d.-]+)%"),
    }


def run_benchmark(binary, label):
    csv_path = f"data/results/{label}_timings.csv"
    fields = ["label", "image", "resolution", "k",
              "init_time", "kmeans_time", "total_time",
              "input_kb", "output_kb", "size_reduction"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for img in IMAGES:
            # Extract resolution from filename (e.g. "512x512")
            res = re.search(r"(\d+x\d+)", os.path.basename(img))
            res = res.group(1) if res else "unknown"

            for k in K_VALUES:
                basename = os.path.splitext(os.path.basename(img))[0]
                output = f"data/results/{label}_{basename}_k{k}.png"

                print(f"Running: {label} | {basename} | K={k}")

                result = subprocess.run(
                    [binary, img, str(k), output],
                    capture_output=True, text=True
                )

                if result.returncode != 0:
                    print(f"  ERROR: {result.stderr.strip()}")
                    continue

                metrics = parse_output(result.stdout)

                row = {
                    "label": label,
                    "image": img,
                    "resolution": res,
                    "k": k,
                    **metrics,
                }
                writer.writerow(row)

                print(f"  -> total={metrics['total_time']}s")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 benchmarks/run_benchmarks.py <binary> <label>")
        sys.exit(1)

    run_benchmark(sys.argv[1], sys.argv[2])