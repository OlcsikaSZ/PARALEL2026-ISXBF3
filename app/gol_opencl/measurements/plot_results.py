import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CSV_FILE = BASE_DIR / "results.csv"
FIGURES_DIR = PROJECT_DIR / "documentation" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

def label_row(row):
    if int(row["tiled"]) == 0:
        return f"naive {int(row['lx'])}x{int(row['ly'])}"
    return f"tiled {int(row['lx'])}x{int(row['ly'])}"

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
    "naive 16x16",
    "tiled 8x8",
    "tiled 16x16",
]

def sort_labels(labels):
    labels = list(labels)
    return sorted(
        labels,
        key=lambda x: (preferred_order.index(x) if x in preferred_order else 999, x)
    )

# ---------- Grafikon 1: kernel idő / iteráció ----------
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

plt.xticks(range(len(size_order)), size_order)
plt.xlabel("Rácsméret")
plt.ylabel("Kernel idő / iteráció (ms)")
plt.title("OpenCL Game of Life - kernel idő / iteráció")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "kernel_per_iter.png", dpi=300)
plt.close()

# ---------- Grafikon 2: gyorsulás a naive verzióhoz képest ----------
naive = df[df["tiled"] == 0][["cells", "size_label", "kernel_ms"]].copy()
naive = naive.rename(columns={"kernel_ms": "naive_kernel_ms"})

tiled = df[df["tiled"] == 1].copy()

merged = tiled.merge(
    naive[["cells", "naive_kernel_ms"]],
    on="cells",
    how="inner"
)

merged["speedup"] = merged["naive_kernel_ms"] / merged["kernel_ms"]
merged["label"] = merged.apply(lambda r: f"tiled {int(r['lx'])}x{int(r['ly'])}", axis=1)

plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 12})

for label in sort_labels(merged["label"].unique()):
    grp = merged[merged["label"] == label].copy()
    grp["x"] = grp["size_label"].map(size_to_x)
    grp = grp.sort_values("x")

    plt.plot(
        grp["x"],
        grp["speedup"],
        marker="o",
        label=label
    )

    for _, row in grp.iterrows():
        plt.text(
            row["x"],
            row["speedup"] + 0.04,
            f"{row['speedup']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(range(len(size_order)), size_order)
plt.xlabel("Rácsméret")
plt.ylabel("Gyorsulás a naive verzióhoz képest")
plt.title("OpenCL Game of Life - gyorsulás")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "speedup.png", dpi=300)
plt.close()

# ---------- Grafikon 3: teljes futási idő ----------
plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 12})

for label in sort_labels(df["label"].unique()):
    grp = df[df["label"] == label].copy()
    grp["x"] = grp["size_label"].map(size_to_x)
    grp = grp.sort_values("x")

    plt.plot(
        grp["x"],
        grp["total_ms"],
        marker="o",
        label=label
    )

plt.xticks(range(len(size_order)), size_order)
plt.xlabel("Rácsméret")
plt.ylabel("Összegzett profilozott végrehajtási idő (ms)")
plt.title("OpenCL Game of Life - összegzett profilozott végrehajtási idő")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "total_time.png", dpi=300)
plt.close()

print(f"Kész: {FIGURES_DIR / 'kernel_per_iter.png'}")
print(f"Kész: {FIGURES_DIR / 'speedup.png'}")
print(f"Kész: {FIGURES_DIR / 'total_time.png'}")
