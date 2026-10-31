# 4D Gauge-Orbit Optimization Engine
C++ OpenMP engine exploring prime-lattice walks via hyperoctahedral greedy heuristics.
Modulates steps with Riemann zeta zeros to test local vector norm minimization against standard spatial drifts up to 1M.

## Theoretical Overview
This framework maps algebraic properties of primes into a spatial coordinate architecture, applying an error-nullifying group action to guarantee localized structural bounds:
1. **Lagrange Lattice Embedding:** Primes $p$ are projected onto a four-square coordinate lattice via $p = a^2 + b^2 + c^2 + d^2$ where $a \ge b \ge c \ge d \ge 0$.
2. **Riemann Zeta Spectrum Phase Modulation:** Vectors are sequentially translated across an array of complex fields parameterized by the non-trivial zeroes $\gamma_k$ of $\zeta(s)$.
3. **Gauge Symmetry Phase Control:** Uncalibrated vectors breach the $\sqrt{p}$ boundaries immediately. By exploiting the 16-fold sign symmetry orbit of $(\pm a, \pm b, \pm c, \pm d)$ via a greedy vector look-ahead scheme, the engine forces destructive phase interference to keep the cumulative displacement tightly bounded below $\sqrt{p}$.

## Repository Anatomy
* `main.cpp` - High-performance arithmetic sieve and 4D coordinate greedy mapping optimizer.
* `plot_generator.py` - Automated plotting routine to visualize drift suppression parameters.
* `drift_compression_plot.png` - Visual proof of empirical boundedness.

## Quick Start & Execution

### 1. Compile and Run the C++ Engine
```bash
g++ -O3 main.cpp -o gauge_engine
./gauge_engine
```
This writes out the step telemetry file: `mertens_drift_high_res.csv`.

### 2. Generate Validation Plot Plots
```bash
python3 plot_generator.py
```
