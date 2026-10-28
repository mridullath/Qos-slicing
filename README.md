QoS-aware network slicing simulation (5G/6G)

Code that produced every number in the paper
"Predictive, Cross-Layer QoS Orchestration for Adaptive Network Slicing in 5G/6G Networks".

It is a slice-level Python simulator (NumPy), **not ns-3**: one cell, fluid work units, three slice classes
(URLLC, eMBB, mMTC), a log-AR(1) channel with fades, and deadline-based queue discard.
Results show how policies rank under this model; they are not ns-3 or field measurements.

 What is compared

| Name | Method |
| --- | --- |
| B1 | Static hard slicing (fixed shares) |
| B2 | Threshold-based reactive soft slicing |
| B3 | Backlog-weighted sharing |
| B4 | Tabular Q-learning (27 states, 6 share presets, 60 training episodes) |
| Proposed | Cross-layer telemetry + ridge-AR forecaster + hybrid isolation (reserve for critical slices) |

Ablation variants switch off cross-layer telemetry, forecasting, hybrid isolation, or all of them.
The forecaster is a ridge autoregressive model (window 8), a lightweight stand-in for an LSTM.

 Traffic sets
D1 periodic, D2 bursty, D3 self-similar (Pareto ON/OFF), D4 **real 5G traces** (eMBB demand and channel quality from the
public Irish operator dataset by Raca et al., ACM MMSys 2020), D5 flash crowd. URLLC and mMTC traffic is synthetic in all sets.

Repository layout

src/sim.py       simulator, traffic/channel generators, baselines, proposed controller, forecaster, Q-learning
src/tune.py      hyperparameter tuning on tuning seeds 3000-3002
src/exp.py       final experiments:  python src/exp.py main | cond | ablate | sweep | scale
src/analyze.py   tables and statistics (Wilcoxon + Holm, Friedman, Cliff's delta, bootstrap CIs)
results/         raw per-run outputs (res_*.csv), analysis_output.txt, tuning_output.txt
get_data.sh      downloads the real 5G traces (not stored in this repo)
run_all.sh       runs everything and writes results/analysis_output.txt

Setup and run
bash
pip install -r requirements.txt      
bash get_data.sh                    
data/ (needs git and unzip)
bash run_all.sh                      

Or run parts individually, e.g. `python src/exp.py main` then `python src/analyze.py`.
To use a dataset stored elsewhere: `export QOS_DATA=/path/to/5Gdataset/data`.

Mapping to the paper
| Paper section | Script | Output |
| --- | --- | --- |
| 3.2 main comparison (5 sets x 30 seeds) | `exp.py main` | `results/res_main.csv` |
| 3.3 across datasets | `exp.py main` | `results/res_main.csv` |
| 3.3 degraded channel and noisy telemetry | `exp.py cond` | `results/res_cond.csv` |
| 3.3 load sweep 0.4-1.0 | `exp.py sweep` | `results/res_sweep.csv` |
| 3.3 slice count 3-50 and decision time | `exp.py scale` | `results/res_scale.csv` |
| 3.4 ablation | `exp.py ablate` | `results/res_ablate.csv` |
| 3.4 statistical tests | `analyze.py` | `results/analysis_output.txt` |
| 2.1 hyperparameter choice | `tune.py` | `results/tuning_output.txt` |

Reproducibility
- All traffic and channel realisations are generated from integer seeds and every method sees the same realisation for a seed (paired comparison).
- Seeds: forecaster training 1000-1007, Q-learning training 2000-2059, tuning 3000-3002, testing 0-29 (main, cond, ablate), 0-14 (sweep), 0-9 (scale).
- The results in `results/` were regenerated from this exact code and match the paper's tables.
  The only column that varies between machines is `dec_us` (wall-clock decision time per epoch).
- Tested with Python 3.12.3, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2.

Selection of settings (honest notes)
- Proposed controller: floor 0.5, margins z = 0, chosen on the tuning seeds by lowest priority-weighted violation subject to URLLC violations under 1%.
  B2 (step 0.2, trigger 0.9) was tuned on the same seeds. B1 and B3 are fixed rules; B4 was trained for 60 episodes.
- An early version of the proposed allocator let a backlogged non-critical slice outrank URLLC and starve it; this was fixed
  (reserves are granted before ordering by slack). All reported results use the fixed code.
- Known limits: simplified single-cell model, synthetic URLLC/mMTC traffic, linear forecaster, lightweight RL baseline.

## Data citation
D. Raca, J. Leahy, C. J. Sreenan, J. J. Quinlan, "Beyond Throughput, the Next Generation: A 5G Dataset with Channel and Context Metrics",
ACM MMSys 2020.

Data:  https://github.com/uccmisl/5Gdataset  (see that repository for its licence).
