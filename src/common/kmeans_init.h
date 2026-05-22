#ifndef KMEANS_INIT_H
#define KMEANS_INIT_H

// K-means++ initialization
// pixels: (N x 3) float array
// n: number of pixels
// k: number of clusters
// returns: (K x 3) float array of initial centers (caller must free)
float *kmeans_plus_plus(const float *pixels, int n, int k);

#endif