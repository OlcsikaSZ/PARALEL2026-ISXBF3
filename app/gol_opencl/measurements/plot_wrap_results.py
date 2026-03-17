import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "results_wrap.csv"

df = pd.read_csv(CSV_FILE)

def label_row(row):
    base = "naive" if int(row["tiled"]) == 0 else f"tiled {int(row['lx'])}x{int(row['ly'])}"
    return f"{base} wrap={int(row['wrap'])}"

df["label"] = df.apply(label_row, axis=1)
df["cells"] = df["rows"] * df["cols"]
df["size_label"] = df.apply(lambda r: f"{int(r['rows'])}x{int(r['cols'])}", axis=1)
df["kernel_per_iter_ms"] = df["kernel_ms"] / df["iters"]

size_order_df = (
    df[["rows", "cols", "cells", "size_label"]]
    .drop_duplicates()
    .sort_values(["cells", "rows", "cols"])
    .reset_index(drop=True)
)

size_order = size_order_df["size_label"].tolist()
size_to_x = {label: i for i, label in enumerate(size_order)}

preferred_order = [
    "naive wrap=0",
    "naive wrap=1",
    "tiled 16x16 wrap=0",
    "tiled 16x16 wrap=1",
]

def sort_labels(labels):
    labels = list(labels)
    return sorted(
        labels,
        key=lambda x: (preferred_order.index(x) if x in preferred_order else 999, x)
    )

# ---------- Wrap összehasonlítás: kernel idő / iteráció ----------
plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 12})

for label in sort_labels(df["label"].unique()):
    grp = df[df["label"] == label].copy()
    grp["x"] = grp["size_label"].map(size_to_x)
    grp = grp.sort_values("x")

    plt.plot(
        grp["x"],
        grp["kernel_per_iter_ms"],
        marker="o",
        label=label
    )

    for _, row in grp.iterrows():
        plt.text(
            row["x"],
            row["kernel_per_iter_ms"] + 0.01,
            f"{row['kernel_per_iter_ms']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(range(len(size_order)), size_order)
plt.xlabel("Rácsméret")
plt.ylabel("Kernel idő / iteráció (ms)")
plt.title("OpenCL Game of Life - wrap összehasonlítás")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(BASE_DIR / "wrap_kernel_compare.png", dpi=300)
plt.close()

print("Kész: wrap_kernel_compare.png")