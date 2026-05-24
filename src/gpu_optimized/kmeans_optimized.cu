
// Optimizing the naive implementation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C"
{
#include "../common/image_io.h"
#include "../common/kmeans_init.h"
}

// CHeck Cuda error
#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t _e = (call);                                 \
        if (_e != cudaSuccess)                                   \
        {                                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Each block loads all K centroids into the shared memory
// Each thread then computes the nearest centroid
__global__ void assign_shared_kernel(
    const float *__restrict__ pixels,
    const float *__restrict__ centroids,
    int *labels,
    int N, int K)
{
    extern __shared__ float s_cent[];

    for (int i = (int)threadIdx.x; i < K * 3; i += (int)blockDim.x)
        s_cent[i] = centroids[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float r = pixels[idx * 3];
    float g = pixels[idx * 3 + 1];
    float b = pixels[idx * 3 + 2];

    float best_dist = FLT_MAX;
    int best_k = 0;

    for (int k = 0; k < K; k++)
    {
        float dr = r - s_cent[k * 3];
        float dg = g - s_cent[k * 3 + 1];
        float db = b - s_cent[k * 3 + 2];
        float d = dr * dr + dg * dg + db * db;
        if (d < best_dist)
        {
            best_dist = d;
            best_k = k;
        }
    }
    labels[idx] = best_k;
}

__global__ void update_shared_kernel(
    const float *__restrict__ pixels,
    const int *__restrict__ labels,
    float *g_sums,
    int *g_counts,
    int N, int K)
{
    // Shared memory layout
    extern __shared__ char smem[];
    float *s_sums = (float *)smem;
    int *s_counts = (int *)(s_sums + K * 3);

    for (int i = (int)threadIdx.x; i < K * 3; i += (int)blockDim.x)
        s_sums[i] = 0.0f;
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x)
        s_counts[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int k = labels[idx];
        atomicAdd(&s_sums[k * 3], pixels[idx * 3]);
        atomicAdd(&s_sums[k * 3 + 1], pixels[idx * 3 + 1]);
        atomicAdd(&s_sums[k * 3 + 2], pixels[idx * 3 + 2]);
        atomicAdd(&s_counts[k], 1);
    }
    __syncthreads();

    for (int i = (int)threadIdx.x; i < K * 3; i += (int)blockDim.x)
        atomicAdd(&g_sums[i], s_sums[i]);
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x)
        atomicAdd(&g_counts[i], s_counts[i]);
}

__global__ void divide_kernel(
    float *centroids,
    const float *sums,
    const int *counts,
    int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K)
        return;
    if (counts[k] > 0)
    {
        float inv = 1.0f / (float)counts[k];
        centroids[k * 3] = sums[k * 3] * inv;
        centroids[k * 3 + 1] = sums[k * 3 + 1] * inv;
        centroids[k * 3 + 2] = sums[k * 3 + 2] * inv;
    }
}

__global__ void recolor_kernel(
    float *pixels,
    const int *labels,
    const float *centroids,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    int k = labels[idx];
    pixels[idx * 3] = centroids[k * 3];
    pixels[idx * 3 + 1] = centroids[k * 3 + 1];
    pixels[idx * 3 + 2] = centroids[k * 3 + 2];
}

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 5)
    {
        fprintf(stderr,
                "Usage: %s <input_image> <K> <output_image> [block_size]\n"
                "  block_size  optional, default 256. Must be power-of-2 in [32, 1024].\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *in_path = argv[1];
    int K = atoi(argv[2]);
    const char *out_path = argv[3];
    int BLOCK = (argc == 5) ? atoi(argv[4]) : 256;

    if (K < 1 || K > 256)
    {
        fprintf(stderr, "K must be in [1, 256]\n");
        return EXIT_FAILURE;
    }
    if (BLOCK < 32 || BLOCK > 1024 || (BLOCK & (BLOCK - 1)) != 0)
    {
        fprintf(stderr, "block_size must be a power-of-2 in [32, 1024]\n");
        return EXIT_FAILURE;
    }

    //  Load image & initialize centroids on the CPU
    Image img = load_image(in_path);
    if (!img.data)
    {
        fprintf(stderr, "Cannot load image: %s\n", in_path);
        return EXIT_FAILURE;
    }

    int N = img.width * img.height;
    printf("Image : %dx%d  (%d pixels)\n", img.width, img.height, N);
    printf("K=%d   block_size=%d\n\n", K, BLOCK);

    float *h_pixels = pixels_to_float(&img);
    float *h_centroids = kmeans_plus_plus(h_pixels, N, K);
    printf("K-means++ init done\n\n");

    size_t smem_assign = (size_t)K * 3 * sizeof(float);
    size_t smem_update = (size_t)K * 3 * sizeof(float) + (size_t)K * sizeof(int);

    {
        int dev;
        cudaGetDevice(&dev);
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("GPU: %s  (sm_%d%d, %.0f KB shared mem/block)\n\n",
               prop.name, prop.major, prop.minor,
               prop.sharedMemPerBlock / 1024.0);

        if (smem_assign > prop.sharedMemPerBlock)
        {
            fprintf(stderr,
                    "Error: assign kernel needs %zu bytes shared mem "
                    "but device has only %zu. Reduce K.\n",
                    smem_assign, prop.sharedMemPerBlock);
            return EXIT_FAILURE;
        }
        if (smem_update > prop.sharedMemPerBlock)
        {
            fprintf(stderr,
                    "Error: update kernel needs %zu bytes shared mem "
                    "but device has only %zu. Reduce K.\n",
                    smem_update, prop.sharedMemPerBlock);
            return EXIT_FAILURE;
        }
    }

    //  Allocate the GPU memory
    float *d_pixels, *d_centroids, *d_sums;
    int *d_labels, *d_counts;

    CUDA_CHECK(cudaMalloc(&d_pixels, (size_t)N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)K * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums, (size_t)K * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)K * sizeof(int)));

    double t_h2d_init_s = now_sec();
    CUDA_CHECK(cudaMemcpy(d_pixels, h_pixels,
                          (size_t)N * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_h2d_init = (now_sec() - t_h2d_init_s) * 1000.0;

    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids,
                          (size_t)K * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    int grid_assign = (N + BLOCK - 1) / BLOCK;
    int grid_update = (N + BLOCK - 1) / BLOCK;
    int grid_divide = (K + BLOCK - 1) / BLOCK;

    const int MAX_ITER = 100;
    const float CONV_TOL = 1.0f;

    double t_assign_total = 0.0;
    double t_update_total = 0.0;
    double t_d2h_total = 0.0;

    int converged = 0;
    int n_iters = 0;

    float *h_new_cents = (float *)malloc((size_t)K * 3 * sizeof(float));

    printf("%-5s  %12s  %12s  %10s  %10s\n",
           "Iter", "Assign(ms)", "Update(ms)", "D2H(ms)", "Total(ms)");
    printf("--------------------------------------------------------------\n");

    double t_wall_start = now_sec();

    for (int iter = 0; iter < MAX_ITER && !converged; iter++)
    {
        n_iters++;

        double ta0 = now_sec();
        assign_shared_kernel<<<grid_assign, BLOCK, smem_assign>>>(
            d_pixels, d_centroids, d_labels, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        double ta1 = now_sec();

        CUDA_CHECK(cudaMemset(d_sums, 0, (size_t)K * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, (size_t)K * sizeof(int)));

        update_shared_kernel<<<grid_update, BLOCK, smem_update>>>(
            d_pixels, d_labels, d_sums, d_counts, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        divide_kernel<<<grid_divide, BLOCK>>>(d_centroids, d_sums, d_counts, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        double tb1 = now_sec();

        CUDA_CHECK(cudaMemcpy(h_new_cents, d_centroids,
                              (size_t)K * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        double tc1 = now_sec();

        converged = 1;
        for (int k = 0; k < K; k++)
        {
            float dr = h_new_cents[k * 3] - h_centroids[k * 3];
            float dg = h_new_cents[k * 3 + 1] - h_centroids[k * 3 + 1];
            float db = h_new_cents[k * 3 + 2] - h_centroids[k * 3 + 2];
            if (sqrtf(dr * dr + dg * dg + db * db) > CONV_TOL)
            {
                converged = 0;
                break;
            }
        }
        memcpy(h_centroids, h_new_cents, (size_t)K * 3 * sizeof(float));

        double ms_assign = (ta1 - ta0) * 1000.0;
        double ms_update = (tb1 - ta1) * 1000.0;
        double ms_d2h = (tc1 - tb1) * 1000.0;
        double ms_tot = ms_assign + ms_update + ms_d2h;

        t_assign_total += ms_assign;
        t_update_total += ms_update;
        t_d2h_total += ms_d2h;

        printf("%-5d  %12.3f  %12.3f  %10.3f  %10.3f\n",
               iter + 1, ms_assign, ms_update, ms_d2h, ms_tot);
    }

    double t_wall = (now_sec() - t_wall_start) * 1000.0;

    printf("\n=== Timing Summary (%d iterations) ===\n", n_iters);
    printf("  Block size: %d\n", BLOCK);
    printf("  %-40s %10.3f ms  (%5.1f%%)\n",
           "Assign kernel (shared mem):", t_assign_total,
           100.0 * t_assign_total / t_wall);
    printf("  %-40s %10.3f ms  (%5.1f%%)\n",
           "Update kernel (parallel reduction):", t_update_total,
           100.0 * t_update_total / t_wall);
    printf("  %-40s %10.3f ms  (%5.1f%%)\n",
           "Device->Host (centroids only, K×3 floats):", t_d2h_total,
           100.0 * t_d2h_total / t_wall);
    printf("  %-40s %10.3f ms  (one-time, outside loop)\n",
           "Initial H2D pixel upload:", t_h2d_init);
    printf("  %-40s %10.3f ms\n", "Total wall time (loop only):", t_wall);
    printf("  Converged: %s\n\n",
           converged ? "yes" : "no (reached max iterations)");

    recolor_kernel<<<grid_assign, BLOCK>>>(d_pixels, d_labels, d_centroids, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_out_f = (float *)malloc((size_t)N * 3 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out_f, d_pixels,
                          (size_t)N * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    Image out_img = float_to_image(h_out_f, img.width, img.height);
    Image palette = make_palette(h_centroids, K, 64);
    Image comparison = make_comparison(&img, &out_img);

    if (!save_image(out_path, &out_img))
        fprintf(stderr, "Warning: could not save %s\n", out_path);
    else
        printf("Saved quantized image: %s\n", out_path);

    char pal_path[512], cmp_path[512];
    snprintf(pal_path, sizeof(pal_path), "%s_palette.png", out_path);
    snprintf(cmp_path, sizeof(cmp_path), "%s_compare.png", out_path);
    if (!save_image(pal_path, &palette))
        fprintf(stderr, "Warning: could not save %s\n", pal_path);
    else
        printf("Saved palette:         %s\n", pal_path);
    if (!save_image(cmp_path, &comparison))
        fprintf(stderr, "Warning: could not save %s\n", cmp_path);
    else
        printf("Saved comparison:      %s\n", cmp_path);

    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);

    free(h_pixels);
    free(h_centroids);
    free(h_new_cents);
    free(h_out_f);

    free_image(&img);
    free_image(&out_img);
    free_image(&palette);
    free_image(&comparison);

    return EXIT_SUCCESS;
}
