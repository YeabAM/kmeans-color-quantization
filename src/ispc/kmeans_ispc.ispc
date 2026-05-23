export void assign_pixels(
    uniform float pixels[],
    uniform float centers[],
    uniform int assignments[],
    uniform int n,
    uniform int k)
{
    foreach (i = 0 ... n) {
        float pr = pixels[i * 3];
        float pg = pixels[i * 3 + 1];
        float pb = pixels[i * 3 + 2];

        float best_dist = 1e30f;
        int best_k = 0;

        for (uniform int c = 0; c < k; c++) {
            float dr = pr - centers[c * 3];
            float dg = pg - centers[c * 3 + 1];
            float db = pb - centers[c * 3 + 2];
            float dist = dr * dr + dg * dg + db * db;
            if (dist < best_dist) {
                best_dist = dist;
                best_k = c;
            }
        }

        assignments[i] = best_k;
    }
}