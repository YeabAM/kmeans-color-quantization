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

# ISPC version
kmeans_ispc.o: src/ispc/kmeans_ispc.ispc
	ispc -O2 --target=avx2-i32x8 -o src/ispc/kmeans_ispc.o -h src/ispc/kmeans_ispc_ispc.h src/ispc/kmeans_ispc.ispc

kmeans_ispc: src/ispc/kmeans_ispc.c src/common/kmeans_init.c $(COMMON) kmeans_ispc.o
	$(CC) $(CFLAGS) -I src/ispc -o $@ src/ispc/kmeans_ispc.c src/common/kmeans_init.c $(COMMON) src/ispc/kmeans_ispc.o $(LDFLAGS)

clean:
	rm -f test_image_io kmeans_cpu kmeans_ispc src/ispc/kmeans_ispc.o src/ispc/kmeans_ispc_ispc.h