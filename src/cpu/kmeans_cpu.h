#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

// Run sequential K-means clustering
// pixels: (N x 3) float array
// n: number of pixels
// centers: (K x 3) float array (modified in place)
// k: number of clusters
// max_iter: maximum iterations
// tol: convergence threshold
// returns: assignments array of length n (caller must free)
int *kmeans_cpu(const float *pixels, int n, float *centers, int k,
                int max_iter, float tol);

// Replace each pixel with its assigned center color
void recolor(float *pixels, int n, const float *centers, const int *assignments);

#endif