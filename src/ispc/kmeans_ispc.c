#include "../common/image_io.h"
#include "../common/kmeans_init.h"
#include "kmeans_ispc.h"
#include "kmeans_ispc_ispc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

static float squared_dist(const float *a, const float *b) {
    float dr = a[0] - b[0];
    float dg = a[1] - b[1];
    float db = a[2] - b[2];
    return dr * dr + dg * dg + db * db;
}

static double time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

static long file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return st.st_size;
}

int *kmeans_ispc(const float *pixels, int n, float *centers, int k,
                 int max_iter, float tol) {

    int *assignments = (int *)malloc(n * sizeof(int));
    float *new_centers = (float *)malloc(k * 3 * sizeof(float));
    int *counts = (int *)malloc(k * sizeof(int));
    if (!assignments || !new_centers || !counts) {
        free(assignments);
        free(new_centers);
        free(counts);
        return NULL;
    }

    double total_assign = 0.0, total_update = 0.0, total_convergence = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
        struct timespec t0, t1, t2, t3;

        // --- Assignment step (ISPC) ---
        clock_gettime(CLOCK_MONOTONIC, &t0);
        assign_pixels((float *)pixels, centers, assignments, n, k);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        // --- Update step (CPU, same as baseline) ---
        memset(new_centers, 0, k * 3 * sizeof(float));
        memset(counts, 0, k * sizeof(int));

        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            new_centers[c * 3]     += pixels[i * 3];
            new_centers[c * 3 + 1] += pixels[i * 3 + 1];
            new_centers[c * 3 + 2] += pixels[i * 3 + 2];
            counts[c]++;
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                new_centers[c * 3]     /= counts[c];
                new_centers[c * 3 + 1] /= counts[c];
                new_centers[c * 3 + 2] /= counts[c];
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t2);

        // --- Convergence check ---
        float max_shift = 0.0f;
        for (int c = 0; c < k; c++) {
            float shift = squared_dist(&centers[c * 3], &new_centers[c * 3]);
            if (shift > max_shift) max_shift = shift;
        }

        memcpy(centers, new_centers, k * 3 * sizeof(float));
        clock_gettime(CLOCK_MONOTONIC, &t3);

        double assign_t = time_diff(t0, t1);
        double update_t = time_diff(t1, t2);
        double conv_t   = time_diff(t2, t3);

        total_assign += assign_t;
        total_update += update_t;
        total_convergence += conv_t;

        printf("  Iter %3d: assign=%.4fs  update=%.4fs  shift=%.2f\n",
               iter + 1, assign_t, update_t, max_shift);

        if (max_shift < tol * tol) {
            printf("  Converged at iteration %d\n", iter + 1);
            break;
        }
    }

    printf("\n--- Timing Breakdown ---\n");
    printf("  Assignment total:   %.4fs\n", total_assign);
    printf("  Update total:       %.4fs\n", total_update);
    printf("  Convergence total:  %.4fs\n", total_convergence);
    printf("  Combined:           %.4fs\n", total_assign + total_update + total_convergence);

    free(new_centers);
    free(counts);
    return assignments;
}

void recolor_ispc(float *pixels, int n, const float *centers, const int *assignments) {
    for (int i = 0; i < n; i++) {
        int c = assignments[i];
        pixels[i * 3]     = centers[c * 3];
        pixels[i * 3 + 1] = centers[c * 3 + 1];
        pixels[i * 3 + 2] = centers[c * 3 + 2];
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <input_image> <K> <output_image>\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    int k = atoi(argv[2]);
    const char *output_path = argv[3];

    Image img = load_image(input_path);
    if (!img.data) return 1;

    int n = img.width * img.height;
    long input_size = file_size(input_path);
    printf("Image: %dx%d (%d pixels), K=%d\n", img.width, img.height, n, k);
    printf("Input file size: %.2f KB\n", input_size / 1024.0);

    float *pixels = pixels_to_float(&img);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    float *centers = kmeans_plus_plus(pixels, n, k);
    if (!centers) {
        fprintf(stderr, "Error: kmeans++ init failed\n");
        free(pixels);
        free_image(&img);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double init_time = time_diff(t_start, t_end);
    printf("K-means++ init: %.4fs\n\n", init_time);

    printf("Running K-means (ISPC)...\n");
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    int *assignments = kmeans_ispc(pixels, n, centers, k, 100, 1e-4);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double kmeans_time = time_diff(t_start, t_end);

    recolor_ispc(pixels, n, centers, assignments);
    Image result = float_to_image(pixels, img.width, img.height);
    save_image(output_path, &result);

    long output_size = file_size(output_path);

    Image palette = make_palette(centers, k, 64);
    save_image("data/results/palette.png", &palette);

    Image comparison = make_comparison(&img, &result);
    save_image("data/results/comparison.png", &comparison);

    printf("\n=== Summary ===\n");
    printf("  Image:           %dx%d (%d pixels)\n", img.width, img.height, n);
    printf("  K:               %d\n", k);
    printf("  Init time:       %.4fs\n", init_time);
    printf("  K-means time:    %.4fs\n", kmeans_time);
    printf("  Total time:      %.4fs\n", init_time + kmeans_time);
    printf("  Input file:      %.2f KB\n", input_size / 1024.0);
    printf("  Output file:     %.2f KB\n", output_size / 1024.0);
    printf("  Size reduction:  %.1f%%\n",
           100.0 * (1.0 - (double)output_size / input_size));

    free(pixels);
    free(centers);
    free(assignments);
    free_image(&img);
    free_image(&result);
    free_image(&palette);
    free_image(&comparison);

    return 0;
}