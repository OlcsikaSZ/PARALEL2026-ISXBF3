CC = gcc
CFLAGS = -O2 -Wall -Wextra -Iinclude -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS = -lOpenCL

TARGET = quicksort_ranges.exe
SRC = main.c src/kernel_loader.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

bench: $(TARGET)
	bench.bat

clean:
	del /Q $(TARGET) results.csv 2>nul || true