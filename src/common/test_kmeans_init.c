#include "image_io.h"
#include "kmeans_init.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_path> <K>\n", argv[0]);
        return 1;
    }

    int k = atoi(argv[2]);

    Image img = load_image(argv[1]);
    if (!img.data) return 1;

    int n = img.width * img.height;
    float *pixels = pixels_to_float(&img);

    float *centers = kmeans_plus_plus(pixels, n, k);
    if (!centers) {
        fprintf(stderr, "Error: kmeans++ failed\n");
        free(pixels);
        free_image(&img);
        return 1;
    }

    printf("K-means++ chose %d centers:\n", k);
    for (int i = 0; i < k; i++) {
        printf("  Center %d: RGB(%.0f, %.0f, %.0f)\n",
               i, centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]);
    }

    // Save palette visualization
    Image palette = make_palette(centers, k, 64);
    save_image("data/results/init_palette.png", &palette);
    printf("Saved palette to data/results/init_palette.png\n");

    free(centers);
    free(pixels);
    free_image(&img);
    free_image(&palette);
    return 0;
}