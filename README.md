# AlphaHVAC

**Intelligent HVAC Energy Optimization via AlphaGo Zero-Style Reinforcement Learning**

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Architecture](#architecture)
- [System Design](#system-design)
- [Dataset](#dataset)
- [Results](#results)
- [How to Add Graphs to This README](#how-to-add-graphs-to-this-readme)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Evaluation Criteria](#evaluation-criteria)
- [Development History](#development-history)
- [Requirements](#requirements)

---

## Overview

AlphaHVAC applies the AlphaGo Zero planning framework to the real-world problem of energy optimization in HVAC (Heating, Ventilation, and Air Conditioning) systems. The agent controls a Variable Air Volume (VAV) damper in 2.5% increments, learning through iterative self-play to minimize energy consumption while maintaining thermal comfort.

The core hypothesis is that HVAC control is structurally a sequential planning problem: the agent must reason ahead, weigh competing objectives, and improve its own policy over time — exactly the setting for which AlphaGo Zero was designed.

---

## Motivation

HVAC systems are responsible for 40–60% of total energy consumption in commercial buildings. Rule-based control systems — fixed schedules, PID controllers, static setpoint logic — are neither adaptive nor optimal. Buildings continuously generate rich sensor data that is largely unused for intelligent control.

AlphaHVAC treats this as a Markov Decision Process and learns a policy that trades off thermal comfort against energy waste, guided by real sensor data from Building 90, Room 102.

---

## Architecture

The system has four components that work together in a self-improving loop.

### 1. Data-Driven Environment

`HVACEnv` wraps the real building sensor dataset. The agent's damper action modifies airflow proportionally against the historical baseline, so the environment reflects real building physics rather than a simplified simulation.

**Reward function:**

```
reward = -|temp_error| - λ * energy - smooth_penalty + action_shaping
```

Where:
- `temp_error = |room_temp - setpoint|` in normalized space
- `energy = adjusted_airflow × thermal_signal`
- `smooth_penalty = 0.1 × |current_damper - prev_damper|`
- `λ = 0.85` (energy penalty weight)
- `action_shaping` provides context-specific bonuses and penalties

**Action shaping logic:**

| Situation | Action | Reward adjustment |
|---|---|---|
| Room comfortable + energy wasteful | DECREASE (0) | +0.50 |
| Room comfortable + energy wasteful | INCREASE (2) | -0.50 |
| Room uncomfortable | INCREASE (2) | +0.30 |
| Room uncomfortable | DECREASE (0) | -0.30 |
| Room comfortable + energy low | HOLD (1) | +0.10 |

The thresholds for "comfortable" and "wasteful" are calibrated at the **median** of the actual data distribution, ensuring roughly equal firing rates across all three shaping conditions.

### 2. Neural Network — AlphaThermalNet

A dual-headed network that learns both policy and value simultaneously.

```
Input: 15-dimensional state vector
         │
   ┌─────▼─────┐
   │  Shared   │  3 × (Linear 128 → LayerNorm → LeakyReLU(0.01) → Dropout(0.2))
   └─────┬─────┘
         │
   ┌─────┴──────────────┐
   │                    │
   ▼                    ▼
Policy Head          Value Head
128 → 64 → 3        128 → 64 → 1
Softmax              Tanh
   │                    │
P(decrease, hold,    V(state)
   increase)         ∈ [-1, +1]
```

**Design choices:**
- `LayerNorm` instead of `BatchNorm` — works correctly at batch size 1, which occurs during every MCTS inference call
- `LeakyReLU(0.01)` — prevents dead neurons from zero-centered normalized inputs
- `Dropout(0.2)` — regularization against overfitting to a small real-world dataset
- Separate sub-networks for each head — allows policy and value to specialize independently

### 3. Monte Carlo Tree Search

AlphaGo Zero-style MCTS with PUCT selection formula:

```
PUCT(s, a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
```

Where:
- `Q(s,a)` — average value of child node (from value head)
- `P(s,a)` — policy prior from neural network
- `N(s)` — parent visit count
- `N(s,a)` — child visit count
- `c_puct` — exploration constant, annealed from 2.0 → 1.0 over training

Dirichlet noise (`α = 0.3, ε = 0.25`) is added to root priors during training to ensure exploration. MCTS is used to select actions during rollout; policy training targets are generated from direct reward comparison (see Training section).

### 4. Iterative Self-Play Training Loop

```
for iteration in 1..10:
    for episode in 1..20:
        start at random position in training data
        for step in 1..200:
            1. MCTS selects action (with Dirichlet noise)
            2. Evaluate all 3 immediate rewards
            3. Softmax(rewards × temperature) → policy target
            4. Step environment with MCTS action
        compute TD targets: r + γ × V(next_state)
    train network on collected (state, policy_target, td_target) tuples
    step learning rate scheduler
```

**Why direct reward comparison for policy targets (not MCTS visit counts):**

With only 3 actions, MCTS visit counts are near-uniform by construction — the PUCT formula forces all 3 branches to be visited roughly equally when `N` is small. Training the policy on `[0.33, 0.33, 0.33]` targets produces a random policy. Instead, the reward difference between actions (up to 1.0 with shaping) is converted to a soft one-hot target via softmax at temperature 5.0, creating a genuine training signal that breaks the random policy trap.

---

## System Design

```
┌─────────────────────────────────────────────────────────┐
│                   SELF-PLAY LOOP                        │
│                                                         │
│  Real Building Data                                     │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────┐    state     ┌──────────────────┐         │
│  │ HVACEnv  │ ──────────►  │  AlphaThermalNet │         │
│  │          │ ◄──────────  │  (Policy + Value)│         │
│  │  reward  │    P(a), V   └──────────────────┘         │
│  └────┬─────┘                      ▲                    │
│       │                            │  guides search     │
│       │ reward,                    │                    │
│       │ next_state    ┌──────────────────┐              │
│       └────────────►  │      MCTS        │              │
│                       │  (PUCT + noise)  │              │
│                       │  action selection│              │
│                       └──────────────────┘              │
│                                                         │
│  Training targets:                                      │
│    policy ← softmax(all 3 rewards × temp=5.0)           │
│    value  ← r + γ × V(next_state)   [TD target]         │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Building 90, Room 102 — real building automation system |
| Period | April 2021 |
| Frequency | 1-minute intervals |
| Total rows | ~5,747 |
| Train rows | 4,022 (70%) |
| Test rows | 1,724 (30%) |
| Split method | Chronological — no data leakage |

**Raw features used:**

| Feature | Description |
|---|---|
| `room_temp` | Indoor air temperature |
| `thermostat_outside_temp` | Outdoor temperature |
| `damper_position` | VAV damper opening (0 = closed, 1 = fully open) |
| `airflow_current` | Measured airflow through the VAV box |
| `supply_discharge_temp` | Temperature of supply air |
| `clg_signal` | Cooling demand signal |
| `htg_signal` | Heating demand signal |
| `htg_valve_position` | Heating valve position |
| `htg_clg_mode` | Binary: 1 = heating, 0 = cooling |

**Engineered features:**

| Feature | Description |
|---|---|
| `setpoint` | Active setpoint based on mode |
| `thermal_signal` | Active demand signal based on mode |
| `hour_of_day` | Hour normalized to [0, 1] |
| `day_of_week` | Day normalized to [0, 1] |
| `room_temp_lag1` | Previous timestep room temperature |
| `damper_lag1` | Previous timestep damper position |

**Normalization:** `RobustScaler` (median + IQR) with values clipped to `[-3, 3]`. RobustScaler is used instead of MinMaxScaler because building sensor data contains outliers from equipment faults and sensor noise.

---

## Results

After training with calibrated thresholds and the fixed policy target mechanism:

> **To add your actual result graphs here, see the [How to Add Graphs](#how-to-add-graphs-to-this-readme) section below.**

### Expected Evaluation Output

When the model is working correctly, Cell 10 of the notebook will print:

```
============================================================
AlphaHVAC — FINAL HONEST EVALUATION
============================================================

[A] ENERGY SAVING: XX.X% — GOOD
    Base: 0.5344   Model: 0.XXXX

[B] COMFORT: +X.X% change — GOOD
    Base: 0.3801   Model: 0.XXXX

[C] DAMPER: mean=0.XXX   zero%: X.X% — GOOD
    Damper moves in sensible range

[D] POLICY: DEC=XX% HOLD=XX% INC=XX% — GOOD
    No single action dominates

MODEL IS GENUINELY GOOD.
```

### Evaluation Criteria

| Metric | GOOD threshold | WEAK threshold | FAIL threshold |
|---|---|---|---|
| Energy saving | ≥ 20% | ≥ 5% | < 0% |
| Comfort change | ≤ +5% | ≤ +20% | > +20% |
| Mean damper | 0.20 – 0.75 | Outside range | > 0.85 or stuck at 0 |
| Action diversity | No action > 75% | One action 60–75% | Near-uniform (≈33% each) |

### Diagnostic: Policy Loss During Training

The policy loss benchmark is `log(3) = 1.0986`, which is the entropy of a completely random policy over 3 actions. Meaningful learning is confirmed when:

```
policy_loss < 1.05   →  LEARNING
policy_loss < 0.90   →  LEARNING WELL
policy_loss ≈ 1.099  →  STILL RANDOM (policy has not learned)
```

---

**Figure 1 — 4-Panel Test Evaluation**

![4-Panel Evaluation](images/AlphaHVAC_Final.png)

This plot shows: (1) energy consumption baseline vs model with green savings area, (2) damper position over time, (3) action scatter plot colored by action type, (4) temperature deviation comparison.

## Project Structure

```
AlphaHVAC/
│
├── AlphaHVAC_FinalCorrected.ipynb   Main notebook — all cells in order
│
├── Dataset/
│   ├── B90_102_exp30m_202104.csv    Raw sensor data (required)
│   ├── Transformed_Optimized.csv    Generated by Cell 2
│   ├── Train_Optimized.csv          Generated by Cell 2
│   └── Test_Optimized.csv           Generated by Cell 2
│
├── alphaHVAC_trained.pth            Saved model weights (generated by Cell 7)
│
├── AlphaHVAC_Final.png              4-panel evaluation plot (generated by Cell 8)
│
├── images/                          Create this folder for README figures
│   └── (place exported plots here)
│
└── README.md                        This file
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AlphaHVAC.git
cd AlphaHVAC

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

---

## Usage

Run the notebook cells in order from top to bottom. Each cell is self-contained and prints its own status.

**Cell-by-cell description:**

| Cell | Purpose | Expected output |
|---|---|---|
| 1 | Diagnosis history | Printed summary of all previous versions |
| 2 | Data preprocessing | `Train: 4022  Test: 1724` + calibrated thresholds |
| 3 | Environment setup | Reward contrast verification table |
| 4 | Neural network | `Network OK: torch.Size([1, 3]) torch.Size([1, 1])` |
| 5 | MCTS | `MCTS ready.` |
| 6 | Pre-training | `Value(best)=+0.XX  Value(worst)=-0.XX  OK` |
| 7 | Main training | Policy loss should drop below 1.099 by iteration 2 |
| 8 | Evaluation + plots | 4-panel figure saved to `AlphaHVAC_Final.png` |
| 9 | Pass/fail report | Graded evaluation across 4 metrics |

**Interpreting the training output:**

```
Epoch  50/150 | policy_loss=0.6759 [LEARNING WELL] | value_loss=0.0428
```

The `[LEARNING WELL]` flag confirms the policy head has broken out of the random policy regime. If you see `[STILL RANDOM]` for more than 3 iterations, increase `NUM_EPISODES` from 20 to 30.

**Loading and running a saved model:**

```python
import torch

model = AlphaThermalNet()
model.load_state_dict(torch.load("alphaHVAC_trained.pth", map_location="cpu"))
model.eval()

# Run inference on a single state vector (15 features, normalized)
state = torch.tensor(your_state_array, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    policy, value = model(state)

action     = torch.argmax(policy).item()   # 0=decrease, 1=hold, 2=increase
confidence = policy[0][action].item()
state_val  = value.item()
```

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `STATE_SIZE` | 15 | Dimension of state vector |
| `ACTION_SIZE` | 3 | Decrease / Hold / Increase damper |
| `damper_step` | 0.025 | Damper change per action (2.5%) |
| `lam` | 0.85 | Energy penalty weight in reward |
| `GAMMA` | 0.99 | Discount factor for TD targets |
| `LR` | 1e-3 | Initial learning rate |
| `BATCH_SIZE` | 256 | Mini-batch size |
| `NUM_ITERATIONS` | 10 | Self-play → train cycles |
| `NUM_EPISODES` | 20 | Short random-start episodes per iteration |
| `STEPS_PER_EP` | 200 | Steps per episode |
| `EPOCHS_PER_ITER` | 150 | Training epochs per iteration |
| `REWARD_TEMP` | 5.0 | Softmax temperature for policy targets |
| `VALUE_W` | 0.5 | Value loss weight relative to policy loss |
| `SIM_SCHEDULE` | [20,20,30,30,50,50,75,75,100,100] | MCTS simulations per iteration |
| `C_PUCT_SCHEDULE` | 2.0 → 1.0 | Exploration constant annealing |
| `dirichlet_alpha` | 0.3 | Dirichlet noise concentration |
| `dirichlet_epsilon` | 0.25 | Noise mixing fraction at root |

---

## Evaluation Criteria

**Four independent tests are run automatically in Cell 9:**

**[A] Energy Saving** — percentage reduction in `airflow × thermal_signal` relative to the historical baseline. This is the primary objective.

**[B] Thermal Comfort** — percentage change in `|room_temp - setpoint|` relative to baseline. A large positive value means the model achieved energy savings by sacrificing comfort, which is considered a failure mode.

**[C] Damper Health** — mean damper position and fraction of timesteps where damper is at zero. A mean above 0.85 indicates the model learned to open the damper always rather than intelligently. A high fraction at zero indicates the damper stuck closed.

**[D] Policy Diversity** — action distribution across decrease/hold/increase. Near-uniform distribution (each ≈ 33%) indicates the policy has not learned. One action above 75% indicates the policy is biased rather than intelligent.

---

## Development History

This model went through four distinct versions before reaching the current implementation.

**Version 1 — Baseline AlphaGo Zero adaptation**
Policy loss remained at `log(3) = 1.0986` throughout training, indicating a completely random policy. The damper drifted to zero over the test episode, producing an orange line at zero. This was not energy saving — it was a stuck damper. Root cause: MCTS visit counts were always `[0.33, 0.33, 0.33]` because the PUCT formula forces equal exploration with only 3 actions and a small simulation budget.

**Version 2 — Reward shaping added**
Same failure. The reward differences from shaping were not reflected in the MCTS search because the value head evaluates the child state, not the immediate reward. Three child states that differ only in damper position by 0.025 receive nearly identical value head outputs, so Q-values remain equal and visits remain uniform.

**Version 3 — Direct reward comparison for policy targets**
Policy loss dropped to 0.65, confirming the policy was learning. However, `comfort_threshold = 0.15` in RobustScaler-normalized space was far too tight for real building data, causing the "uncomfortable" shaping condition to fire on 85% of timesteps. The model learned to increase the damper almost always (84.4% of actions). Mean damper reached 0.96.

**Version 4 (current) — Calibrated thresholds**
Thresholds set to the median of the actual data distribution, giving roughly balanced firing rates across all three reward conditions. Expected action distribution: DECREASE 40–50%, HOLD 25–35%, INCREASE 15–25%.

---

## Requirements

```
Python        >= 3.9
PyTorch       >= 2.0.0
NumPy         >= 1.24.0
pandas        >= 2.0.0
scikit-learn  >= 1.3.0
matplotlib    >= 3.7.0
```

The code does not require a GPU. Training on CPU takes approximately 2–4 hours depending on the simulation schedule. For GPU acceleration, no code changes are needed — PyTorch will automatically use CUDA if available. For TPU acceleration, replace `optimizer.step()` with `xm.optimizer_step(optimizer)` and add `xm.mark_step()` after each batch.

---

## License

MIT License. See `LICENSE` for details.
