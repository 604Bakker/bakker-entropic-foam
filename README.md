# bakker-entropic-foam

**Emergent Topological Chiral Index in Causal-Set Dirac--Kähler Fermions**  
Greg Bakker & Grok (xAI) – November 2025

Random 2+1D causal sets + Dirac--Kähler fermions + Chern--Simons parity bias + Wilson doubler removal → spontaneous saturation of the chiral index at the regulator bound (80 zero modes with uniform chirality) when the Wilson strength r ≳ 0.2.

No gauge group by hand. No Higgs by hand. Just geometry, topology, and a doubler-killer.

### Results
- Sharp transition at r_c ≈ 0.2
- Index jumps to +80 exact zeros and stays saturated
- Full (r × ε × N) phase-space logs included

- Data speaks. Clone it. Break it. Extend it.
Paper (preprint): bakker_grok_2025.pdf
Code + logs in this repo.
Greg Bakker & Grok — November 2025

## Reproduce the phase diagram
python task4_full_gold.py   # runs the sweep (hours on laptop)
python create_pdf.py                # generates figures/index_phase_diagram.pdf from the log

### Reproduce
```bash
python task4_full_gold.py


