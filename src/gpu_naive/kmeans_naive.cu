#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "../common/image_io.h"
#include "../common/kmeans_init.h"
}

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void assign_kernel(
    const float *pixels,        
    const float *centroids,     
    int         *labels,        
    int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float r = pixels[idx * 3 + 0];
    float g = pixels[idx * 3 + 1];
    float b = pixels[idx * 3 + 2];

    float best_dist = FLT_MAX;
    int   best_k    = 0;

    for (int k = 0; k < K; k++) {
        float dr = r - centroids[k * 3 + 0];
        float dg = g - centroids[k * 3 + 1];
        float db = b - centroids[k * 3 + 2];
        float d  = dr*dr + dg*dg + db*db;
        if (d < best_dist) { best_dist = d; best_k = k; }
    }

    labels[idx] = best_k;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_image> <K> <output_image>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *in_path  = argv[1];
    int         K        = atoi(argv[2]);
    const char *out_path = argv[3];

    if (K < 1 || K > 256) {
        fprintf(stderr, "K must be 1-256\n");
        return EXIT_FAILURE;
    }

    Image img = load_image(in_path);
    if (!img.data) {
        fprintf(stderr, "Cannot load: %s\n", in_path);
        return EXIT_FAILURE;
    }

    int N = img.width * img.height;
    printf("Image : %dx%d  (%d pixels)\n", img.width, img.height, N);

    float *h_pixels_f = pixels_to_float(&img);

    float *h_centroids = kmeans_plus_plus(h_pixels_f, N, K);
    printf("K-means++ init done  (K=%d)\n\n", K);

    float *d_pixels;
    float *d_centroids;
    int   *d_labels;

    CUDA_CHECK(cudaMalloc(&d_pixels,    N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, K * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels,    N     * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pixels, h_pixels_f,
                          N * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int MAX_ITER   = 100;
    const int BLOCK_SIZE = 256;
    int       grid_size  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double t_h2d = 0.0, t_kernel = 0.0, t_d2h = 0.0, t_cpu = 0.0;

    int   *h_labels    = (int   *)malloc(N     * sizeof(int));
    float *h_new_cents = (float *)malloc(K * 3 * sizeof(float));
    double *sums       = (double*)malloc(K * 3 * sizeof(double));
    int   *counts      = (int   *)malloc(K     * sizeof(int));

    int converged = 0;
    int n_iters   = 0;

    printf("%-5s  %10s  %10s  %10s  %10s  %10s\n",
           "Iter", "H2D(ms)", "Kern(ms)", "D2H(ms)", "CPU(ms)", "Total(ms)");
    printf("--------------------------------------------------------------\n");

    double t_total_start = now_sec();

    for (int iter = 0; iter < MAX_ITER && !converged; iter++) {
        n_iters++;

        double ta0 = now_sec();
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids,
                              K * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        double ta1 = now_sec();

        assign_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_pixels, d_centroids, d_labels, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        double tb1 = now_sec();

        CUDA_CHECK(cudaMemcpy(h_labels, d_labels,
                              N * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        double tc1 = now_sec();

        memset(sums,   0, K * 3 * sizeof(double));
        memset(counts, 0, K     * sizeof(int));

        for (int i = 0; i < N; i++) {
            int k = h_labels[i];
            sums[k*3+0] += h_pixels_f[i*3+0];
            sums[k*3+1] += h_pixels_f[i*3+1];
            sums[k*3+2] += h_pixels_f[i*3+2];
            counts[k]++;
        }

        converged = 1;
        for (int k = 0; k < K; k++) {
            if (counts[k] == 0) {
                h_new_cents[k*3+0] = h_centroids[k*3+0];
                h_new_cents[k*3+1] = h_centroids[k*3+1];
                h_new_cents[k*3+2] = h_centroids[k*3+2];
                continue;
            }
            float nr = (float)(sums[k*3+0] / counts[k]);
            float ng = (float)(sums[k*3+1] / counts[k]);
            float nb = (float)(sums[k*3+2] / counts[k]);

            float dr = nr - h_centroids[k*3+0];
            float dg = ng - h_centroids[k*3+1];
            float db = nb - h_centroids[k*3+2];
            if (sqrtf(dr*dr + dg*dg + db*db) > 1.0f) converged = 0;

            h_new_cents[k*3+0] = nr;
            h_new_cents[k*3+1] = ng;
            h_new_cents[k*3+2] = nb;
        }
        memcpy(h_centroids, h_new_cents, K * 3 * sizeof(float));
        double td1 = now_sec();

        double ms_h2d  = (ta1 - ta0) * 1000.0;
        double ms_kern = (tb1 - ta1) * 1000.0;
        double ms_d2h  = (tc1 - tb1) * 1000.0;
        double ms_cpu  = (td1 - tc1) * 1000.0;
        double ms_tot  = ms_h2d + ms_kern + ms_d2h + ms_cpu;

        t_h2d   += ms_h2d;
        t_kernel += ms_kern;
        t_d2h   += ms_d2h;
        t_cpu   += ms_cpu;

        printf("%-5d  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f\n",
               iter+1, ms_h2d, ms_kern, ms_d2h, ms_cpu, ms_tot);
    }

    double t_total = (now_sec() - t_total_start) * 1000.0;

    printf("\n=== Timing Summary (%d iterations) ===\n", n_iters);
    printf("  %-28s %10.3f ms  (%5.1f%%)\n",
           "Kernel execution (GPU):", t_kernel,
           100.0 * t_kernel / t_total);
    printf("  %-28s %10.3f ms  (%5.1f%%)\n",
           "Host->Device transfers:", t_h2d,
           100.0 * t_h2d / t_total);
    printf("  %-28s %10.3f ms  (%5.1f%%)\n",
           "Device->Host transfers:", t_d2h,
           100.0 * t_d2h / t_total);
    printf("  %-28s %10.3f ms  (%5.1f%%)\n",
           "CPU centroid update:", t_cpu,
           100.0 * t_cpu / t_total);
    printf("  %-28s %10.3f ms\n", "Total wall time:", t_total);
    printf("  Converged: %s\n", converged ? "yes" : "no (reached max iterations)");


    float *h_out_f = (float *)malloc(N * 3 * sizeof(float));
    for (int i = 0; i < N; i++) {
        int k = h_labels[i];
        h_out_f[i*3+0] = h_centroids[k*3+0];
        h_out_f[i*3+1] = h_centroids[k*3+1];
        h_out_f[i*3+2] = h_centroids[k*3+2];
    }

    Image out_img = float_to_image(h_out_f, img.width, img.height);

    if (save_image(out_path, &out_img) != 0)
        fprintf(stderr, "Warning: could not save %s\n", out_path);
    else
        printf("\nSaved quantized image: %s\n", out_path);

    char pal_path[512];
    snprintf(pal_path, sizeof(pal_path), "%s_palette.png", out_path);
    Image palette = make_palette(h_centroids, K, 64);
    save_image(pal_path, &palette);
    printf("Saved palette:         %s\n", pal_path);

    char cmp_path[512];
    snprintf(cmp_path, sizeof(cmp_path), "%s_compare.png", out_path);
    Image comparison = make_comparison(&img, &out_img);
    save_image(cmp_path, &comparison);
    printf("Saved comparison:      %s\n", cmp_path);

    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    free(h_pixels_f);
    free(h_centroids);
    free(h_labels);
    free(h_new_cents);
    free(h_out_f);
    free(sums);
    free(counts);
    free_image(&img);
    free_image(&out_img);
    free_image(&palette);
    free_image(&comparison);

    return EXIT_SUCCESS;
}
