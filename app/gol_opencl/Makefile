CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude
LDFLAGS=-lOpenCL

SRC=main.c src/kernel_loader.c

all: gol_opencl

gol_opencl: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o gol_opencl $(LDFLAGS)

clean:
	rm -f gol_opencl
