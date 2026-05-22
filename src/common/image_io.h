#ifndef IMAGE_IO_H
#define IMAGE_IO_H

typedef struct {
    unsigned char *data;  // RGB pixels, row-major (H * W * 3)
    int width;
    int height;
} Image;

// Load image from file (returns NULL data on failure)
Image load_image(const char *path);

// Save RGB pixel data to PNG
int save_image(const char *path, const Image *img);

// Free image data
void free_image(Image *img);

// Convert uint8 RGB to float array (N x 3), where N = width * height
float *pixels_to_float(const Image *img);

// Convert float array (N x 3) back to uint8 RGB image
Image float_to_image(const float *pixels, int width, int height);

// Generate a palette strip image (swatch_size x K*swatch_size)
Image make_palette(const float *centers, int k, int swatch_size);

// Generate side-by-side comparison image
Image make_comparison(const Image *original, const Image *quantized);

#endif