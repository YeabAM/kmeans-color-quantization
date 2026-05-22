#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    // Load
    Image img = load_image(argv[1]);
    if (!img.data) return 1;
    printf("Loaded: %dx%d\n", img.width, img.height);

    // Flatten to float
    float *pixels = pixels_to_float(&img);
    if (!pixels) {
        fprintf(stderr, "Error: failed to convert to float\n");
        free_image(&img);
        return 1;
    }

    // Unflatten back to image
    Image rebuilt = float_to_image(pixels, img.width, img.height);

    // Compare pixel by pixel
    int n = img.width * img.height * 3;
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (img.data[i] != rebuilt.data[i]) mismatches++;
    }
    printf("Roundtrip mismatches: %d / %d\n", mismatches, n);

    // Save rebuilt image
    save_image("data/results/roundtrip_test.png", &rebuilt);
    printf("Saved roundtrip_test.png\n");

    free(pixels);
    free_image(&img);
    free_image(&rebuilt);
    return 0;
}