# Report — Continuous Control (DDPG, Reacher 20-agent)

## 1. Problem

Control 20 double-jointed arms so each hand stays on its moving target. Continuous state (33) and
continuous action (4, in `[-1, 1]`); reward +0.1 per in-target step. **Solved = average score
≥ 30 over 100 consecutive episodes**, where an episode's score is the mean over the 20 agents.

## 2. Learning Algorithm — DDPG

**Deep Deterministic Policy Gradient** — an off-policy actor–critic method for continuous action
spaces. It combines DPG (a deterministic actor) with DQN ideas (replay buffer + target networks).

- **Actor** `μ(s|θ)` — deterministic policy mapping state → action.
- **Critic** `Q(s,a|φ)` — estimates the action-value; trained with the Bellman/TD target
  `y = r + γ Q'(s', μ'(s'))` using **target networks** `μ'`, `Q'`.
- **Replay buffer** decorrelates samples; **all 20 agents write to one shared buffer**, which makes
  learning much faster and more stable than a single agent.
- **Exploration** via Ornstein–Uhlenbeck (temporally-correlated) noise added to the actor's action.
- **Soft target updates**: `θ' ← τθ + (1-τ)θ'` (τ = 1e-3).

### Network architecture (`model.py`)
| | Layers |
|---|---|
| **Actor** | `Linear(33→256) → ReLU → Linear(256→128) → ReLU → Linear(128→4) → tanh` |
| **Critic** | `Linear(33→256) → ReLU → [concat action] → Linear(256+4→128) → ReLU → Linear(128→1)` |

Hidden layers initialised from `U(-1/√fan_in, 1/√fan_in)`; final layers from `U(-3e-3, 3e-3)`.

### Key stabilisation choices (`ddpg_agent.py`)
With 20 agents feeding one buffer, vanilla DDPG (learn every step) is unstable. Two changes from the
benchmark fixed it:
- **Learn `LEARN_NUM = 10` times every `LEARN_EVERY = 20` timesteps** (decouples experience
  collection from the number of gradient updates).
- **Critic gradient clipping** to `GRAD_CLIP = 1.0` (prevents value-function blow-ups early on).

### Hyperparameters
| Param | Value | Param | Value |
|---|---|---|---|
| Replay buffer | 1e6 | Tau (soft update) | 1e-3 |
| Batch size | 128 | Actor LR | 1e-3 |
| Gamma | 0.99 | Critic LR | 1e-3 |
| Learn every | 20 | Learn passes | 10 |
| Critic grad clip | 1.0 | Weight decay | 0 |

## 3. Results

**Environment solved in 204 episodes** — the 100-episode moving average first reached ≥ 30 at
episode **204** (Avg(100) = 30.06). Per-episode score rose from ~0.7 to a steady ~35–37.

![Learning curve](learning_curve.png)

The dip-free, monotonic moving-average curve indicates a stable training run with no catastrophic
forgetting. Trained weights: `checkpoint_actor.pth`, `checkpoint_critic.pth`.

## 4. Future Work
- **TD3** (twin critics + delayed policy updates + target-policy smoothing) — usually more stable
  and sample-efficient than DDPG.
- **D4PG** (distributional critic + n-step returns + prioritised replay) — strong on this exact task.
- **PPO / A2C** as an on-policy comparison.
- **Prioritised experience replay** and an **OU-noise decay schedule** for finer late-stage control.
- Systematic hyperparameter search (learning rates, batch size, network width).

## 5. Reproducibility note
The Unity Reacher binary is a 2018 Linux x86_64 / Mono build that does **not** run on Apple-Silicon
Macs (QEMU too slow; Rosetta crashes Mono with SIGABRT). It was built and trained via Docker on a
**native Intel x86_64 Linux host** — see `README.md` for the exact commands.
