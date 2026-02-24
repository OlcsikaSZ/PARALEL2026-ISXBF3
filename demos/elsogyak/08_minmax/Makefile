CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude
LDFLAGS=-lOpenCL -lm

SRCS=main.c src/kernel_loader.c src/minmax_opencl.c src/minmax_utils.c
TARGET=main.exe

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
