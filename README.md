# PINN Stability Benchmark

Companion code for **"Physics-Informed Neural Networks for Power System Stability Analysis: A Systematic Review Across Stability Classes, Benchmarks, and Open Challenges"** (IEEE Access, submitted).

It compares three ways of computing power system transient-stability trajectories:

1. **Numerical solver** (SciPy Radau IIA, the reference / ground truth),
2. **MLP**: a purely data-driven neural network trained only on labelled trajectory points,
3. **PINN**: the same network trained on labelled points **plus** the swing-equation residual.

Test systems: single-machine infinite-bus (SMIB) and the IEEE/WSCC 3-machine 9-bus system (classical model, three-phase fault at bus 7 cleared by tripping line 5-7).

## Install

```bash
pip install -r requirements.txt
```

(Google Colab already has everything installed.)

## Run

```bash
python case_study.py --mode test     # 2-5 min, checks that everything works
python case_study.py --mode quick    # 15-60 min, one seed, first real results
python case_study.py --mode full     # results reported in the paper (5 seeds); GPU recommended
```

Options: `--out FOLDER` (output folder, default `results`), `--skip9bus`, `--restart` (ignore finished runs; by default the script resumes where it stopped).

## Experiments

| ID | Question | Output |
|----|----------|--------|
| E1 | Accuracy (relative L2 error, max error) | `summary_table11.csv` |
| E2 | Data efficiency (10 to 1000 labelled points) | `summary_E2_data.csv`, `fig_error_vs_data.png` |
| E3 | Computational cost and break-even number of queries | `summary_E3_cost.csv` |
| E4 | Error growth with time horizon (2 to 20 s) | `summary_E4_horizon.csv`, `fig_error_vs_horizon.png` |
| E5 | Unstable cases, stability classification, CCT (9-bus) | `summary_E5_9bus.csv`, `fig_9bus_trajectories.png` |
| E6 | Scalability SMIB -> 9-bus | `summary_table11.csv` |

All individual runs are in `results.csv`; software/hardware and settings in `run_info.txt`.

## Citation

If you use this code, please cite the paper above (and the Zenodo DOI of this repository).

## License

MIT
