CC = gcc
CFLAGS = -O2 -Wall -Wextra -Iinclude -Ivendor/include -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS = -Lvendor/lib -lOpenCL
TARGET = main.exe

all: $(TARGET)

$(TARGET): main.c src/kernel_loader.c
	$(CC) $(CFLAGS) main.c src/kernel_loader.c -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f results/*.txt results/*.csv