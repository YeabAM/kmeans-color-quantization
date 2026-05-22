CC = gcc
CFLAGS = -Wall -O2 -D_POSIX_C_SOURCE=199309L
LDFLAGS = -lm

COMMON = src/common/image_io.c

# Test image I/O roundtrip
test_image_io: src/common/test_image_io.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


# Test K-means++ initialization
test_kmeans_init: src/common/test_kmeans_init.c src/common/kmeans_init.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# CPU sequential baseline (uncomment when ready)
kmeans_cpu: src/cpu/kmeans_cpu.c src/common/kmeans_init.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f test_image_io kmeans_cpu