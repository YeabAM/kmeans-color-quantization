CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm

# CPU sequential baseline
kmeans_cpu: src/cpu/kmeans_cpu.c src/common/image_io.c src/common/kmeans_init.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f kmeans_cpu