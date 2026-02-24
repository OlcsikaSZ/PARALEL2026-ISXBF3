CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude
LDFLAGS=-lOpenCL -lm

SRCS=main.c src/kernel_loader.c src/fill_missing_opencl.c src/fill_missing_utils.c

main.exe: $(SRCS) include/fill_missing.h include/kernel_loader.h
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f main.exe
