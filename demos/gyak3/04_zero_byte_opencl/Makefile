CC = gcc
CFLAGS = -O2 -Wall -Wextra -Iinclude -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS = -lOpenCL

TARGET = zero_count
SRC = main.c src/kernel_loader.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	del /Q $(TARGET).exe 2>nul || exit 0

run: $(TARGET)
	.\$(TARGET).exe data\test_256mb.bin 256