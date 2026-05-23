#ifndef KMEANS_ISPC_H
#define KMEANS_ISPC_H

int *kmeans_ispc(const float *pixels, int n, float *centers, int k,
                 int max_iter, float tol);

void recolor_ispc(float *pixels, int n, const float *centers, const int *assignments);

#endif