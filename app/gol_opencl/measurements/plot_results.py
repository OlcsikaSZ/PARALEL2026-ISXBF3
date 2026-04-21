import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path

# Resolve input and output locations relative to this script.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CSV_FILE = BASE_DIR / "results.csv"
FIGURES_DIR = PROJECT_DIR / "documentation" / "figures"

# Create the figure output folder if it does not exist yet.
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load the benchmark CSV produced by the measurement batch script.
df = pd.read_csv(CSV_FILE)

if "mode" not in df.columns:
    df.insert(0, "mode", df["tiled"].apply(lambda v: "gpu_tiled" if int(v) == 1 else "gpu_naive"))
if "wall_total_ms" not in df.columns:
    df["wall_total_ms"] = df["total_ms"]

# Build a readable series label from the kernel mode and local size.
def label_row(row):
    mode = str(row["mode"])
    if mode == "cpu_seq":
        return "cpu seq"
    if mode == "gpu_naive":
        return f"naive {int(row['lx'])}x{int(row['ly'])}"
    return f"tiled {int(row['lx'])}x{int(row['ly'])}"

# Add helper columns used by the plots.
df["label"] = df.apply(label_row, axis=1)
df["cells"] = df["rows"] * df["cols"]
df["size_label"] = df.apply(lambda r: f"{int(r['rows'])}x{int(r['cols'])}", axis=1)
df["kernel_per_iter_ms"] = df["kernel_ms"] / df["iters"]
df["wall_per_iter_ms"] = df["wall_total_ms"] / df["iters"]

# Define a stable x-axis order based on grid size.
size_order_df = (
    df[["rows", "cols", "cells", "size_label"]]
    .drop_duplicates()
    .sort_values(["cells", "rows", "cols"])
    .reset_index(drop=True)
)
size_order = size_order_df["size_label"].tolist()
size_to_x = {label: i for i, label in enumerate(size_order)}

# Keep the legend order consistent across all figures.
preferred_order = [
    "cpu seq",
    "naive 16x16",
    "tiled 4x4",
    "tiled 8x8",
    "tiled 16x16",
]


def sort_labels(labels):
    labels = list(labels)
    return sorted(labels, key=lambda x: (preferred_order.index(x) if x in preferred_order else 999, x))


gpu_df = df[df["mode"] != "cpu_seq"].copy()
if not gpu_df.empty:
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12})

    for label in sort_labels(gpu_df["label"].unique()):
        grp = gpu_df[gpu_df["label"] == label].copy()
        grp["x"] = grp["size_label"].map(size_to_x)
        grp = grp.sort_values("x")
        plt.plot(grp["x"], grp["kernel_per_iter_ms"], marker="o", label=label)

    plt.xticks(range(len(size_order)), size_order)
    plt.xlabel("Rácsméret")
    plt.ylabel("Kernel idő / iteráció (ms)")
    plt.title("OpenCL Game of Life - GPU kernel idő / iteráció")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kernel_per_iter.png", dpi=300)
    plt.close()

# Plot tiled-kernel speedup versus the naive baseline.
plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 12})
for label in sort_labels(df["label"].unique()):
    grp = df[df["label"] == label].copy()
    grp["x"] = grp["size_label"].map(size_to_x)
    grp = grp.sort_values("x")
    plt.plot(grp["x"], grp["wall_total_ms"], marker="o", label=label)

plt.xticks(range(len(size_order)), size_order)
plt.xlabel("Rácsméret")
plt.ylabel("Teljes falióra-idő (ms)")
plt.title("OpenCL Game of Life - teljes falióra-idő")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "total_time.png", dpi=300)
plt.close()

cpu_df = df[df["mode"] == "cpu_seq"][["cells", "size_label", "wall_total_ms"]].copy()
gpu_speed_df = df[df["mode"] != "cpu_seq"].copy()

if not cpu_df.empty and not gpu_speed_df.empty:
    cpu_df = cpu_df.rename(columns={"wall_total_ms": "cpu_wall_total_ms"})
    merged = gpu_speed_df.merge(cpu_df, on=["cells", "size_label"], how="inner")
    merged["speedup"] = merged["cpu_wall_total_ms"] / merged["wall_total_ms"]

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12})
    for label in sort_labels(merged["label"].unique()):
        grp = merged[merged["label"] == label].copy()
        grp["x"] = grp["size_label"].map(size_to_x)
        grp = grp.sort_values("x")
        plt.plot(grp["x"], grp["speedup"], marker="o", label=label)
        
        # Annotate each point with its numeric speedup.
        for _, row in grp.iterrows():
            plt.text(row["x"], row["speedup"] + 0.04, f"{row['speedup']:.2f}x", ha="center", va="bottom", fontsize=9)

    plt.xticks(range(len(size_order)), size_order)
    plt.xlabel("Rácsméret")
    plt.ylabel("Gyorsulás a CPU szekvenciális módhoz képest")
    plt.title("OpenCL Game of Life - gyorsulás a CPU baseline-hoz képest")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "speedup.png", dpi=300)
    plt.close()
else:
    naive = gpu_speed_df[gpu_speed_df["mode"] == "gpu_naive"][["cells", "size_label", "kernel_ms"]].copy()
    naive = naive.rename(columns={"kernel_ms": "naive_kernel_ms"})
    merged = gpu_speed_df[gpu_speed_df["mode"] == "gpu_tiled"].merge(naive, on=["cells", "size_label"], how="inner")
    merged["speedup"] = merged["naive_kernel_ms"] / merged["kernel_ms"]
    
    # Plot the total profiled runtime including transfers and kernel execution.
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12})
    for label in sort_labels(merged["label"].unique()):
        grp = merged[merged["label"] == label].copy()
        grp["x"] = grp["size_label"].map(size_to_x)
        grp = grp.sort_values("x")
        plt.plot(grp["x"], grp["speedup"], marker="o", label=label)

    plt.xticks(range(len(size_order)), size_order)
    plt.xlabel("Rácsméret")
    plt.ylabel("Gyorsulás a naive GPU verzióhoz képest")
    plt.title("OpenCL Game of Life - gyorsulás (fallback, CPU baseline nélkül)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "speedup.png", dpi=300)
    plt.close()

# Print the generated figure paths for quick confirmation.
print(f"Kész: {FIGURES_DIR / 'kernel_per_iter.png'}")
print(f"Kész: {FIGURES_DIR / 'speedup.png'}")
print(f"Kész: {FIGURES_DIR / 'total_time.png'}")
