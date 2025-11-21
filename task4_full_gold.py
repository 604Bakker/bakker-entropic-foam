#!/usr/bin/env python3
"""
task4_full_gold_working.py
Full r × ε × N phase space with Wilson term
Your geometry + gold upgrade + crash-proof eigensolver
"""

import os
import time
import random
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

OUTDIR = "task4_full_gold"
os.makedirs(OUTDIR, exist_ok=True)

import datetime

LOGFILE = os.path.join(OUTDIR, f"gold_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
print(f"Full log → {LOGFILE}")

# Safe way to grab the real built-in print
import builtins
_real_print = builtins.print

def tee_print(*args, **kwargs):
    _real_print(*args, **kwargs)                                   # console
    with open(LOGFILE, "a", encoding="utf-8") as f:
        _real_print(*args, file=f, **kwargs)                      # file

# Replace print globally
print = tee_print

# CONFIG — the full sweep
SIZES = [100, 150, 200]
R_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EPS_LIST = np.linspace(0.0, 0.30, 16)   # 0.02 step
BOOTSTRAP = 8
K_EIGS = 80
SEED_BASE = 7777

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# 2+1D sprinkling
def sprinkle_21D(N, box=(3.0, 1.0, 1.0)):
    t = np.random.uniform(0, box[0], N)
    x = np.random.uniform(-box[1], box[1], N)
    y = np.random.uniform(-box[2], box[2], N)
    return np.stack([t, x, y], axis=1)

# Causal relation in 2+1D
def is_causal(p, q):
    dt = q[0] - p[0]
    dx = q[1] - p[1]
    dy = q[2] - p[2]
    return dt > 0 and dt*dt - dx*dx - dy*dy > 0

# FULL causal relation — gives triangles galore
def causal_links_21D(points):
    N = len(points)
    edges = []
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if is_causal(points[i], points[j]):
                a, b = min(i,j), max(i,j)
                if (a,b) not in edges:
                    edges.append((a,b))
                    adj[a].add(b)
                    adj[b].add(a)
    return sorted(edges), adj

def triangles_21D(adj, limit=5000):
    tris = set()
    N = len(adj)
    for i in range(N):
        for j in adj[i]:
            if j <= i: continue
            common = adj[i] & adj[j]
            for k in common:
                if k <= j: continue
                tris.add(tuple(sorted((i,j,k))))
                if len(tris) >= limit:
                    return sorted(list(tris))
    return sorted(list(tris))

class DK21D:
    def __init__(self, D, node_map, edges, tris):
        self.D = D
        self.node_map = node_map
        self.edges = edges
        self.tris = tris
        self.N_act = len(node_map)
        self.E = len(edges)
        self.T = len(tris)
        self.total = self.N_act + self.E + self.T

def build_dk_21D(points, edges, tris):
    active = sorted({n for e in edges for n in e})
    node_map = {n:i for i,n in enumerate(active)}
    N_act = len(active)
    E = len(edges)
    T = len(tris)
    total = N_act + E + T

    row, col, data = [], [], []
    for ei, (a,b) in enumerate(edges):
        ai = node_map[a]
        bi = node_map[b]
        row += [ei, ei]
        col += [ai, bi]
        data += [-1.0, 1.0]
    d0 = coo_matrix((data, (row, col)), shape=(E, N_act)).tocsr()

    edge_idx = {e:i for i,e in enumerate(edges)}
    row, col, data = [], [], []
    for ti, tri in enumerate(tris):
        i,j,k = tri
        for u,v in ((i,j),(j,k),(k,i)):
            e = (min(u,v), max(u,v))
            ei = edge_idx[e]
            sign = 1 if u < v else -1
            row.append(ti)
            col.append(ei)
            data.append(sign)
    d1 = coo_matrix((data, (row, col)), shape=(T, E)).tocsr() if row else coo_matrix((T, E)).tocsr()

    rows, cols, vals = [], [], []
    co = d0.tocoo()
    for r,c,v in zip(co.row, co.col, co.data):
        rows += [c, N_act + r]
        cols += [N_act + r, c]
        vals += [v, v]
    co = d1.tocoo()
    for r,c,v in zip(co.row, co.col, co.data):
        rows += [N_act + c, N_act + E + r]
        cols += [N_act + E + r, N_act + c]
        vals += [v, v]

    D = coo_matrix((vals, (rows, cols)), shape=(total, total)).tocsr()
    return DK21D(D, node_map, edges, tris)

def cs_scalar_21D(dk, s_weak_full, eps):
    cs = np.zeros(dk.N_act)
    for i,j,k in dk.tris:
        vi = dk.node_map[i]
        vj = dk.node_map[j]
        vk = dk.node_map[k]
        val = np.dot(s_weak_full[i], np.cross(s_weak_full[j], s_weak_full[k]))
        cs[vi] += val
        cs[vj] += val
        cs[vk] += val
    return cs * eps

def local_gamma_21D(dk, s_weak_full):
    diag = np.ones(dk.total)
    for ei, (a,b) in enumerate(dk.edges):
        idx = dk.N_act + ei
        orient = np.sign(np.dot(s_weak_full[a], s_weak_full[b]) + 1e-12)
        diag[idx] = -orient
    return diags(diag)

def safe_eigs(Op, k):
    try:
        return eigsh(Op, k=k, sigma=0.0, which='LM', tol=1e-8, maxiter=40000)
    except Exception as e1:
        print(f"    shift-invert failed ({e1}), trying SA")
        try:
            return eigsh(Op, k=k, which='SA', tol=1e-8)
        except Exception as e2:
            print(f"    SA failed ({e2}), falling back to SM")
            return eigsh(Op, k=k, which='SM')

def single_run(N, r_wilson, eps, seed):
    set_seed(seed)
    pts = sprinkle_21D(N)
    edges, adj = causal_links_21D(pts)
    tris = triangles_21D(adj)
    dk = build_dk_21D(pts, edges, tris)

    s_weak = np.random.normal(0, 0.04, (len(pts), 3))

    cs = cs_scalar_21D(dk, s_weak, eps)
    Gamma = local_gamma_21D(dk, s_weak)

    Mdiag = np.zeros(dk.total)
    Mdiag[:dk.N_act] = cs
    M = diags(Mdiag)
    Mch = 0.5 * (M @ Gamma + Gamma @ M)

    # GOLD LINE
    wilson = r_wilson * (dk.D @ dk.D)
    Op = dk.D + Mch + wilson

    vals, vecs = safe_eigs(Op, K_EIGS)
    ch = np.real(np.diagonal(vecs.T.conj() @ Gamma @ vecs))

    asym = np.sum(vals > 0) - np.sum(vals < 0)
    zero_mask = np.abs(vals) < 1e-6
    idx = int(np.sum(np.sign(ch[zero_mask]))) if np.any(zero_mask) else 0

    print(f"  N={N:3d} r={r_wilson:.1f} ε={eps:.3f} | A={int(asym):5d} idx={idx:5d} λ∈[{vals.min():.6f},{vals.max():.6f}] T={dk.T}")

def main():
    print("=== TASK 4 — FULL PHASE SPACE GOLD SWEEP ===")
    for N in SIZES:
        for r in R_VALUES:
            print(f"\n{'='*70}")
            print(f" N = {N} | r_wilson = {r:.1f}")
            print(f"{'='*70}")
            for eps in EPS_LIST:
                print(f"\nε = {eps:.3f} (bootstrap {BOOTSTRAP})")
                for rep in range(BOOTSTRAP):
                    seed = SEED_BASE + rep + N*1000 + int(r*10) + int(eps*10000)
                    single_run(N, r, eps, seed)

if __name__ == "__main__":
    main()