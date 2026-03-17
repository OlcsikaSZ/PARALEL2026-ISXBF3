import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = "results.csv"

if not os.path.exists(CSV_FILE):
    print(f"Nincs ilyen fájl: {CSV_FILE}")
    raise SystemExit(1)

df = pd.read_csv(CSV_FILE)

# Biztonság kedvéért rendezzük
df = df.sort_values(by=["n", "input_type", "gap_seq", "local_size"])

# Csak helyes futásokat vegyünk figyelembe
df_ok = df[df["correct"] == 1].copy()

if df_ok.empty:
    print("Nincs helyes futás a CSV-ben.")
    raise SystemExit(1)

# 1. CPU vs teljes GPU idő random input, gap=ciura, local=64
sub = df_ok[
    (df_ok["input_type"] == "random") &
    (df_ok["gap_seq"] == "ciura") &
    (df_ok["local_size"] == 64)
].copy()

if not sub.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(sub["n"], sub["cpu_ms"], marker="o", label="CPU")
    plt.plot(sub["n"], sub["total_gpu_ms"], marker="s", label="GPU total")
    plt.xlabel("Elemszám")
    plt.ylabel("Idő (ms)")
    plt.title("CPU vs GPU teljes idő (random input, Ciura, local=64)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_cpu_vs_gpu.png", dpi=150)
    plt.close()

# 2. Gap sorozatok összehasonlítása random input, local=64
sub = df_ok[
    (df_ok["input_type"] == "random") &
    (df_ok["local_size"] == 64)
].copy()

if not sub.empty:
    plt.figure(figsize=(10, 6))
    for gap in sorted(sub["gap_seq"].unique()):
        s = sub[sub["gap_seq"] == gap]
        plt.plot(s["n"], s["kernel_ms"], marker="o", label=gap)

    plt.xlabel("Elemszám")
    plt.ylabel("Kernel idő (ms)")
    plt.title("Gap-sorozatok összehasonlítása (random input, local=64)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend(title="Gap sequence")
    plt.tight_layout()
    plt.savefig("plot_gap_sequences.png", dpi=150)
    plt.close()

# 3. Bemenettípusok összehasonlítása Ciura, local=64
sub = df_ok[
    (df_ok["gap_seq"] == "ciura") &
    (df_ok["local_size"] == 64)
].copy()

if not sub.empty:
    plt.figure(figsize=(10, 6))
    for inp in sorted(sub["input_type"].unique()):
        s = sub[sub["input_type"] == inp]
        plt.plot(s["n"], s["kernel_ms"], marker="o", label=inp)

    plt.xlabel("Elemszám")
    plt.ylabel("Kernel idő (ms)")
    plt.title("Bemenettípusok hatása (Ciura, local=64)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend(title="Input type")
    plt.tight_layout()
    plt.savefig("plot_input_types.png", dpi=150)
    plt.close()

# 4. Local work size összehasonlítás random input, Ciura
sub = df_ok[
    (df_ok["input_type"] == "random") &
    (df_ok["gap_seq"] == "ciura")
].copy()

if not sub.empty:
    plt.figure(figsize=(10, 6))
    for ls in sorted(sub["local_size"].unique()):
        s = sub[sub["local_size"] == ls]
        plt.plot(s["n"], s["kernel_ms"], marker="o", label=f"local={ls}")

    plt.xlabel("Elemszám")
    plt.ylabel("Kernel idő (ms)")
    plt.title("Local work size hatása (random input, Ciura)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_local_sizes.png", dpi=150)
    plt.close()

print("Grafikonok elmentve:")
print(" - plot_cpu_vs_gpu.png")
print(" - plot_gap_sequences.png")
print(" - plot_input_types.png")
print(" - plot_local_sizes.png")