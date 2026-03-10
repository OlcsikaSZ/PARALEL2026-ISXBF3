import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt

INPUT_CSV = "results/measurements.csv"
OUTPUT_PNG = "results/plot_total_time.png"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Missing file: {INPUT_CSV}")
        return

    grouped = defaultdict(list)

    with open(INPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_size = int(row["file_size"])
            local_size = int(row["local_size"])
            total_ms = float(row["total_ms"])
            grouped[local_size].append((file_size, total_ms))

    plt.figure(figsize=(10, 6))

    for local_size, values in sorted(grouped.items()):
        values.sort(key=lambda x: x[0])
        x = [v[0] / (1024 * 1024) for v in values]
        y = [v[1] for v in values]
        plt.plot(x, y, marker="o", label=f"local_size={local_size}")

    plt.xlabel("File size (MB)")
    plt.ylabel("Total time (ms)")
    plt.title("OpenCL zero-byte counting total runtime")
    plt.grid(True)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()