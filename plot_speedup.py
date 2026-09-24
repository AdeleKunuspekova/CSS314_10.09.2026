"""
plot_speedup.py -- generates speedup_plot.png (Phase 6 deliverable)

Expects a results.csv with columns:
    k,T_k,S_emp

Fill this in with YOUR real measured T_k values (average of Run2/Run3)
for k = 1,2,4,8,16. This script then:
  1. computes S_emp(k) = T_1 / T_k  (if not already supplied)
  2. derives p from S_emp(2)
  3. computes S_theo(k) via Amdahl's law
  4. plots S_emp, S_theo, and the linear ideal (S=k) vs k

Usage: python3 plot_speedup.py results.csv
"""
import sys
import csv
import matplotlib.pyplot as plt

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    ks, Tks = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ks.append(int(row["k"]))
            Tks.append(float(row["T_k"]))

    T1 = Tks[ks.index(1)]
    S_emp = [T1 / t for t in Tks]

    # Derive p from empirical speedup at k=2
    if 2 not in ks:
        raise ValueError("Need a k=2 data point to derive p")
    S2 = S_emp[ks.index(2)]
    p = 2 * (1 - (1 / S2))
    print(f"Derived parallel fraction p = {p:.4f}")

    S_theo = [1 / ((1 - p) + (p / k)) for k in ks]
    ideal = ks

    plt.figure(figsize=(7, 5))
    plt.plot(ks, S_emp, "o-", label="S_emp(k) (measured)")
    plt.plot(ks, S_theo, "s--", label=f"S_theo(k) (Amdahl, p={p:.3f})")
    plt.plot(ks, ideal, ":", label="Linear ideal (S=k)")
    plt.xlabel("Threads (k)")
    plt.ylabel("Speedup S(k)")
    plt.title("Amdahl Reality Gap: Empirical vs Theoretical Speedup")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("speedup_plot.png", dpi=150)
    print("Saved speedup_plot.png")

if __name__ == "__main__":
    main()
