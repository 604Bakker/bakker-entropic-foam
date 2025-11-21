import re
import numpy as np
import matplotlib.pyplot as plt
import os

def find_log():
    for root, _, files in os.walk("."):
        for f in files:
            if f.startswith("gold_sweeps") and f.endswith(".log"):
                return os.path.join(root, f)
    raise FileNotFoundError("No gold_sweep*.log found")

logfile = find_log()
print(f"Found log: {logfile}")

data = {'N': [], 'r': [], 'eps': [], 'idx': [], 'A': []}

# Ultra-lenient pattern — eats any amount of spaces, handles negative A/idx
pattern = r"N\s*=\s*(\d+)\s+r\s*=\s*([\d.]+)\s+ε\s*=\s*([\d.]+)\s+\|\s+A\s*=\s*([-\d]+)\s+idx\s*=\s*([-\d]+)"

with open(logfile, 'r', encoding='utf-8') as f:
    for line in f:
        m = re.search(pattern, line)
        if m:
            N = int(m.group(1))
            r = float(m.group(2))
            eps = float(m.group(3))
            A = int(m.group(4))
            idx = int(m.group(5))
            data['N'].append(N)
            data['r'].append(r)
            data['eps'].append(eps)
            data['idx'].append(idx)
            data['A'].append(A)

if len(data['N']) == 0:
    raise ValueError("Still no lines parsed — double-check the log is the right one and has lines like 'N=100 r=0.0 ε=0.000'")

# the rest is unchanged from the last version
N_arr = np.array(data['N'])
r_arr = np.array(data['r'])
eps_arr = np.array(data['eps'])
idx_arr = np.array(data['idx'])

Ns = sorted(set(N_arr))
rs = sorted(set(r_arr))
epss = sorted(set(eps_arr))

os.makedirs('figures', exist_ok=True)

fig, axs = plt.subplots(1, len(Ns), figsize=(5*len(Ns), 5), sharey=True)
if len(Ns) == 1:
    axs = [axs]

for ax_idx, N in enumerate(Ns):
    ax = axs[ax_idx] if len(Ns) > 1 else axs[0]
    mask = N_arr == N
    grid = np.full((len(rs), len(epss)), np.nan)
    for i, r in enumerate(rs):
        for j, eps in enumerate(epss):
            sub = mask & np.isclose(r_arr, r) & np.isclose(eps_arr, eps, atol=1e-4)
            if np.any(sub):
                grid[i,j] = np.mean(idx_arr[sub])
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=80)
    ax.set_xticks(np.arange(len(epss))[::2])
    ax.set_xticklabels([f"{e:.2f}" for e in epss[::2]], rotation=90)
    ax.set_yticks(np.arange(len(rs)))
    ax.set_yticklabels([f"{r:.1f}" for r in rs])
    ax.set_title(f"N = {N}")
    ax.set_xlabel("ε_parity")
    if ax_idx == 0:
        ax.set_ylabel("r_wilson")

cbar = plt.colorbar(im, ax=axs.ravel().tolist() if len(Ns)>1 else axs, label="Mean Chiral Index")
cbar.set_ticks(np.linspace(0, 80, 9))

plt.tight_layout()
plt.savefig("figures/index_phase_diagram.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("index_phase_diagram.pdf created — drop it in Overleaf and recompile!")