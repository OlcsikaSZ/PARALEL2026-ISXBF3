CC = gcc
CFLAGS = -O2 -Wall -Wextra -Iinclude -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS = -lOpenCL

TARGET = shell_sort.exe
SRC = main.c src/kernel_loader.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

bench: $(TARGET)
	bench.bat

plot:
	python plot.py

rebuild: clean all

clean:
	del /Q $(TARGET) results.csv plot_cpu_vs_gpu.png plot_gap_sequences.png plot_input_types.png plot_local_sizes.png 2>nul || true