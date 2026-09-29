# SoftSeparationARO

Code for the computational experiments accompanying the paper "Soft Separation for Adaptive Robust Optimization". 

## Citation

If you use this repository, please cite the accompanying preprint:

> Qingyuan Xu and Ruiwei Jiang, "Soft Separation for Adaptive Robust Optimization," 2026.
> Available at [Optimization-Online](https://optimization-online.org/2026/09/soft-separation-for-adaptive-robust-optimization/).

## Repository structure
Manuscript section | Notebook | Purpose |
|---|---|---|
| Section 5.1.1 | [Convergence analysis](sec_5_1_1/Section_5_1_1_Convergence_Analysis.ipynb) | Convergence figures: scenario-set size, adaptive versus uniform proposal, and sensitivity to the temperature $w$. |
| Section 5.1.2 | [Continuous runtime experiments](sec_5_1_2/Section_5_1_2_Runtime_Experiments.ipynb) | Runtime comparisons for continuous first-stage decisions. |
| Section 5.2 | [Mixed-integer experiments](sec_5_2/Section_5_2_Mixed_Integer_Experiments.ipynb) | Runtime comparisons for mixed-integer first-stage decisions. |



## Requirements

The code was developed with Python 3.9 and Gurobi 12. A working Gurobi license
is required. The main Python dependencies are:

- `gurobipy`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `jupyterlab` or `notebook`
- `ipython`

A minimal environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install gurobipy numpy pandas scipy matplotlib jupyterlab ipython
```

Verify the Gurobi installation before running the experiments:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

