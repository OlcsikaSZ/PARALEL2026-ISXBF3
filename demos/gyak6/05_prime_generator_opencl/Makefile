CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS=-lOpenCL
TARGET=prime_generator_opencl

all: $(TARGET)

$(TARGET): main.c src/kernel_loader.c include/kernel_loader.h kernels/prime_batch.cl
	$(CC) $(CFLAGS) main.c src/kernel_loader.c -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.exe
