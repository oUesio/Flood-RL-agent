# Flood Warning Deep RL Agent

A deep reinforcement learning agent that learns to issue flood warnings (None, Yellow, Amber, Red) to minimise casualties and economic damage while limiting false alarms. The agent is trained with RecurrentPPO on a probabilistic simulation environment grounded in real UK data.

---

## Overview

Deciding when and at what level to issue a flood warning involves balancing two competing costs: the social and economic harm of missed warnings versus the credibility cost of unnecessary ones. This project frames that decision as a discrete-action RL problem and trains an LSTM-based policy to outperform rule-based baselines.

**Key features:**
- Gymnasium environment simulating flood dynamics over 200-step episodes
- 18-dimensional observation space drawn from real geospatial, demographic, and meteorological data
- Vulnerability scoring that combines physical, socioeconomic, preparedness, recovery, and exposure factors
- Curriculum learning via `severe_prob` to expose the agent to high-impact events early in training
- Hyperparameter optimisation with Optuna (100+ trials)
- Comparison against random and decision-tree threshold baselines

---
## Setup

```bash
pip install -r requirements.txt
```
Python 3.10+ recommended.


---

## Project Structure

```
.
├── data/                 # Data used by the simulation environment for fitting distributions and sampling
├── agent.py              # FloodWarningEnv (Gymnasium environment, reward function)
├── environment.py        # Vulnerability calculation and impact scoring
├── features.py           # Feature sampling from geospatial/raster data
├── household.py          # Household demographic generator (UK census)
├── util.py               # Constants, file paths, helper functions
├── train.ipynb           # RecurrentPPO Optuna hyperparameter search and full training
├── train_vecnorm.ipynb   # RecurrentPPO with Vecnormalize Optuna hyperparameter search and full training
├── evaluation.ipynb      # Agent evaluation and metric plots
├── baselines/
│   ├── random_baseline.py     # Random baseline using historical warning distribution
│   └── threshold_baseline.py  # Threshold-based baseline using decision tree threshold policy
├── scripts/
│   ├── generate_grid.ipynb       # Build the 1km spatial grid over Greater Manchester from geospatial data
│   ├── historic_flood.ipynb      # Process historical flood warnings (2006–2025), mapping precipitation and response data
│   ├── simulate_historic.ipynb   # Generate historical samples using the historical mapped data
│   ├── simulate.ipynb            # Generate samples from the simulation environment
│   ├── impact_threshold.ipynb    # Calibrate impact score thresholds to match historical warning frequencies
│   └── dep_network.ipynb         # Visualise directed graph of feature dependencies and impact score relationships
├── models/               # Saved RecurrentPPO agents from training
└── results/              # Evaluation plots and figures
```

---

## Training

Open [`train.ipynb`](train.ipynb) or [`train_vecnorm.ipynb`](train_vecnorm.ipynb) to run Optuna hyperparameter search followed by final model training. Trained models are saved to their respective directories (`models/lstm/` or `models/lstm_vecnorm/`).

---

## Evaluation

Open [`evaluation.ipynb`](evaluation.ipynb) to evaluate the RL agents alongside the random and threshold baselines. Results are saved to `results/evaluation_results.csv` and plots to `results/`.

---
## Environment

| Property | Value |
|---|---|
| Observation space | 18 continuous features, normalised to [0, 1] |
| Action space | 4 discrete actions (0 = None, 1 = Yellow, 2 = Amber, 3 = Red) |
| Episode length | 200 steps (truncation) |

**Observation features:**
- *Physical*: precipitation, flood depth, elevation, impervious surface fraction, water distance/density
- *Environmental*: season (sin/cos), soil moisture, historical flood flag
- *Socioeconomic*: deprivation index, land use fractions (residential/commercial/industrial/agriculture/transport), population density
- *Temporal*: public holiday flag

**Reward function:**
```
reward = alignment − false_alarm_penalty − missed_penalty − jump_penalty
```
- `alignment = 1.0` when predicted warning level equals ground-truth impact level
- `false_alarm_penalty = 1.5 × action × max(0, action − impact)`
- `missed_penalty = 2.0 × impact × max(0, impact − action)`
- `jump_penalty = 0.5 × max(0, Δaction − 1)` (penalises rapid escalation)
