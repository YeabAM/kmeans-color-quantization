#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Image load_image(const char *path) {
    Image img = {0};
    int channels;
    // Force 3 channels (RGB)
    img.data = stbi_load(path, &img.width, &img.height, &channels, 3);
    if (!img.data) {
        fprintf(stderr, "Error: could not load image '%s'\n", path);
    }
    return img;
}

int save_image(const char *path, const Image *img) {
    if (!img->data) return 0;
    int ok = stbi_write_png(path, img->width, img->height, 3, img->data, img->width * 3);
    if (!ok) {
        fprintf(stderr, "Error: could not save image '%s'\n", path);
    }
    return ok;
}

void free_image(Image *img) {
    if (img->data) {
        free(img->data);
        img->data = NULL;
    }
}

float *pixels_to_float(const Image *img) {
    int n = img->width * img->height;
    float *out = (float *)malloc(n * 3 * sizeof(float));
    if (!out) return NULL;
    for (int i = 0; i < n * 3; i++) {
        out[i] = (float)img->data[i];
    }
    return out;
}

Image float_to_image(const float *pixels, int width, int height) {
    Image img = {0};
    int n = width * height;
    img.width = width;
    img.height = height;
    img.data = (unsigned char *)malloc(n * 3);
    if (!img.data) return img;
    for (int i = 0; i < n * 3; i++) {
        float v = pixels[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        img.data[i] = (unsigned char)(v + 0.5f);
    }
    return img;
}

Image make_palette(const float *centers, int k, int swatch_size) {
    Image img = {0};
    img.width = k * swatch_size;
    img.height = swatch_size;
    img.data = (unsigned char *)malloc(img.width * img.height * 3);
    if (!img.data) return img;

    for (int c = 0; c < k; c++) {
        unsigned char r = (unsigned char)(centers[c * 3] + 0.5f);
        unsigned char g = (unsigned char)(centers[c * 3 + 1] + 0.5f);
        unsigned char b = (unsigned char)(centers[c * 3 + 2] + 0.5f);

        for (int y = 0; y < swatch_size; y++) {
            for (int x = 0; x < swatch_size; x++) {
                int px = (y * img.width + c * swatch_size + x) * 3;
                img.data[px]     = r;
                img.data[px + 1] = g;
                img.data[px + 2] = b;
            }
        }
    }
    return img;
}

Image make_comparison(const Image *original, const Image *quantized) {
    Image img = {0};
    if (original->height != quantized->height) return img;

    img.width = original->width + quantized->width;
    img.height = original->height;
    img.data = (unsigned char *)malloc(img.width * img.height * 3);
    if (!img.data) return img;

    for (int y = 0; y < img.height; y++) {
        // Copy original row
        memcpy(img.data + (y * img.width) * 3,
               original->data + (y * original->width) * 3,
               original->width * 3);
        // Copy quantized row
        memcpy(img.data + (y * img.width + original->width) * 3,
               quantized->data + (y * quantized->width) * 3,
               quantized->width * 3);
    }
    return img;
}