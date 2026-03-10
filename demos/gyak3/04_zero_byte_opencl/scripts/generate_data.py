import os
import sys
import random

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_data.py <output_file> <size_bytes> <zero_ratio>")
        sys.exit(1)

    output_file = sys.argv[1]
    size_bytes = int(sys.argv[2])
    zero_ratio = float(sys.argv[3])

    if not (0.0 <= zero_ratio <= 1.0):
        print("zero_ratio must be between 0.0 and 1.0")
        sys.exit(1)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    chunk_size = 1024 * 1024
    written = 0

    with open(output_file, "wb") as f:
        while written < size_bytes:
            current_chunk = min(chunk_size, size_bytes - written)
            data = bytearray(current_chunk)

            for i in range(current_chunk):
                if random.random() < zero_ratio:
                    data[i] = 0
                else:
                    data[i] = random.randint(1, 255)

            f.write(data)
            written += current_chunk

    print("Generated %s (%d bytes, zero ratio ~ %.3f)" % (output_file, size_bytes, zero_ratio))

if __name__ == "__main__":
    main()