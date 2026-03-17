CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS=-lOpenCL

TARGET=counting_sort
SRC=main.c src/kernel_loader.c

all:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)