#include "kmeans_init.h"
#include <stdlib.h>
#include <float.h>
#include <time.h>

static float squared_dist(const float *a, const float *b) {
    float dr = a[0] - b[0];
    float dg = a[1] - b[1];
    float db = a[2] - b[2];
    return dr * dr + dg * dg + db * db;
}

float *kmeans_plus_plus(const float *pixels, int n, int k) {
    float *centers = (float *)malloc(k * 3 * sizeof(float));
    float *min_dists = (float *)malloc(n * sizeof(float));
    if (!centers || !min_dists) {
        free(centers);
        free(min_dists);
        return NULL;
    }

    srand(time(NULL));

    // Pick first center randomly
    int idx = rand() % n;
    centers[0] = pixels[idx * 3];
    centers[1] = pixels[idx * 3 + 1];
    centers[2] = pixels[idx * 3 + 2];

    // Initialize min distances to first center
    for (int i = 0; i < n; i++) {
        min_dists[i] = squared_dist(&pixels[i * 3], &centers[0]);
    }

    // Pick remaining centers
    for (int c = 1; c < k; c++) {
        // Build cumulative distribution
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            total += min_dists[i];
        }

        // Sample next center
        double r = ((double)rand() / RAND_MAX) * total;
        double cumulative = 0.0;
        int chosen = n - 1;
        for (int i = 0; i < n; i++) {
            cumulative += min_dists[i];
            if (cumulative >= r) {
                chosen = i;
                break;
            }
        }

        centers[c * 3]     = pixels[chosen * 3];
        centers[c * 3 + 1] = pixels[chosen * 3 + 1];
        centers[c * 3 + 2] = pixels[chosen * 3 + 2];

        // Update min distances with new center
        for (int i = 0; i < n; i++) {
            float d = squared_dist(&pixels[i * 3], &centers[c * 3]);
            if (d < min_dists[i]) {
                min_dists[i] = d;
            }
        }
    }

    free(min_dists);
    return centers;
}