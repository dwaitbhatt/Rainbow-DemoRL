# Rainbow-DemoRL

Rainbow-DemoRL is a modular framework for leveraging demonstrations with reinforcement learning for robot manipulation. It implements three orthogonal categories of approaches ("strategies") to use demonstrations, and supports arbitrary hybrid combinations of these strategies.

## Installation

```bash
git clone https://github.com/dwaitbhatt/Rainbow-DemoRL.git && cd Rainbow-DemoRL
pip install -e .
```

## Three Strategies for Using Demonstrations
![Overview of the three strategies for using demonstrations with online RL](docs/rainbow_demorl_overview.png)

**Base RL algorithm:** `--algo` [`SAC` / `TD3`] - All strategies to use demos are applied on top of either [SAC](https://arxiv.org/abs/1801.01290) or [TD3](https://arxiv.org/abs/1802.09477) for stochastic and deterministic policies respectively. We recommend using SAC.

| Strategy                    | Flag / keys                                                           | Idea                                                                                                                                                                                                        |
| ------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A: Direct data sampling** | `--use-offline-data-for-rl` (RLPD), `--use-auxiliary-bc-loss`         | Prefill buffer with demos for direct use in online RL update ([RLPD](https://arxiv.org/abs/2302.02948)) and/or add a BC loss term on the actor.                                                             |
| **B: Offline pretraining**  | `--pretrained-offline-policy-type`, `--pretrained-offline-value-type` | Pretrain actors and/or critics with BC / [CQL](https://arxiv.org/abs/2006.04779) / [CalQL](https://arxiv.org/abs/2303.05479) / MCQ on demos, then finetune online.                                     |
| **C: Action mixing**        | `--algo` [`IBRL_`* / `CHEQ_*` / `RESRL_*`]                        | Mix actions from a frozen control prior and the RL policy ([IBRL](https://arxiv.org/abs/2311.02198), [CHEQ](https://arxiv.org/abs/2406.19768), [Residual RL](https://arxiv.org/abs/1812.06298)). |


**[WSRL](https://arxiv.org/abs/2412.07762) (A+B):** `--offline-buffer-type rollout` - fill the offline buffer with rollouts from the pretrained policy instead of raw demos.


## Training Pipeline

The general workflow has up to three stages - generating demonstrations, offline pretraining (optional), and online RL.

```
  Stage 1: Generate demonstrations       Stage 2 (optional):               Stage 3: Online RL
                                         Offline pretraining
 ┌────────────────────────────┐         ┌────────────────────────┐        ┌──────────────────────────┐
 │ Recommended: Train RL      │──.h5──> │ BC / CQL / CalQL / MCQ │─.pt──> │ SAC or TD3               │
 │ expert with SAC / TD3      │         └────────────────────────┘        │ + Strategy A/B/C flags   │
 │ + --save-buffer            │                                           └──────────────────────────┘
 └────────────────────────────┘
 ┌────────────────────────────┐
 │ Alternative: Motion        │
 │ planning demos             │
 └────────────────────────────┘
```

### Stage 1
**Recommended (RL expert demos):** Run pure online RL with `--save-buffer`. The trainer writes trajectories under `demos/<robot>/<env_id>/rl_buffer/<exp_name>/` as ManiSkill-compatible HDF5. This data can be filtered with [`rainbow_demorl/utils/filter_dataset_by_return.py`](rainbow_demorl/utils/filter_dataset_by_return.py) to use top-X% trajectories as demos for further stages.

_Example_:
```bash
python -m rainbow_demorl.train \
    -a SAC \
    -e PickCube-v1 \
    -r xarm6_robotiq \
    --online-learning-timesteps 1000000 \
    --save-buffer \
    --exp-name my_sac_expert_buffer
python rainbow_demorl/utils/filter_dataset_by_return.py -i demos/xarm6_robotiq/PickCube-v1/rl_buffer/my_sac_expert_buffer/trajectory.h5 -p 0.9
```


**Alternative (motion planning):** Run `python -m rainbow_demorl.generate_motionplanning_demos` to produce solver-generated trajectories without training an RL policy first.

_Example_:
```bash
python -m rainbow_demorl.generate_motionplanning_demos \
    -nt 1000 \
    -e PickCube-v1 \
    -r xarm6_robotiq
```


### Stage 2 
Optional stage, only required if using strategies B or C in stage 3. Trains an offline policy and/or value function from the HDF5 demonstrations. Produces a `.pt` checkpoint.

_Examples_:
```bash
# Simple Behavior Cloning
python -m rainbow_demorl.train \
    -a BC_DET \
    -e PickCube-v1 \
    -r xarm6_robotiq \
    --demo-path path/to/trajectory.h5

# CQL (offline RL)
python -m rainbow_demorl.train \
    -a CQL \
    --cql_variant cql-rho \
    -e PickCube-v1 \
    -r xarm6_robotiq \
    --demo-path path/to/trajectory.h5

```


### Stage 3 
Trains the online RL agent, optionally leveraging the pretrained checkpoint (strategy B), demo data (strategy A), and/or a control prior for action mixing (strategy C).

_Examples_:
```bash
# Strategy A: RLPD
python -m rainbow_demorl.train -a SAC -e PickCube-v1 -r xarm6_robotiq \
  --use-offline-data-for-rl --offline-buffer-type demos --demo-path path/to/trajectory.h5

# Strategy A+B: RLPD + pretrained BC
python -m rainbow_demorl.train -a SAC -e PickCube-v1 -r xarm6_robotiq \
  --use-offline-data-for-rl --offline-buffer-type demos --demo-path path/to/trajectory.h5 \
  --pretrained-offline-policy-type BC_GAUSS

# Strategy A+B+C: Auxiliary BC on rollouts + pretrained CalQL + IBRL
python -m rainbow_demorl.train -a IBRL_SAC -e PickCube-v1 -r xarm6_robotiq \
  --use-auxiliary-bc-loss --offline-buffer-type rollout \
  --pretrained-offline-policy-type CALQL --pretrained-offline-value-type CALQL \
  --control-prior-type BC_DET
```

*Note*:  Using pretrained policy/value type as an argument (e.g. `--pretrained-offline-policy-type BC_DET`) looks for model checkpoints at the default paths as set in [`rainbow_demorl/utils/defaults.py`](rainbow_demorl/utils/defaults.py). Alternatively, you can directly pass pretrained model paths (e.g. `--pretrained-offline-policy-path path/to/model.pt`).


## Main CLI flags


| Flag                                                                    | Alias   | Role                                                                            |
| ----------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------- |
| `--algorithm`                                                           | `-a`    | Algorithm name - `SAC`, `TD3`, `BC_DET`, `CALQL`, `IBRL_TD3`, etc               |
| `--env-id`                                                              | `-e`    | ManiSkill env - `PushCube-v1`, `PickCube-v1`, `StackCube-v1`                    |
| `--robot`                                                               | `-r`    | Maniskill robot - `xarm6_robotiq`, `panda`                                      |
| `--online-learning-timesteps`                                           | `-ton`  | Online environment interaction steps                                            |
| `--offline-learning-grad-steps`                                         | `-toff` | Offline training steps                                                          |
| `--demo-path`                                                           |         | HDF5 demonstrations                                                             |
| `--offline-buffer-type`                                                 |         | `none` / `demos` / `rollout`                                                    |
| `--use-offline-data-for-rl`                                             |         | Strategy A: RLPD-style buffer prefill                                           |
| `--use-auxiliary-bc-loss`                                               |         | Strategy A: Extra BC loss on online actor                                       |
| `--pretrained-offline-policy-type` / `--pretrained-offline-policy-path` |         | Strategy B: Pretrained policy/actor                                             |
| `--pretrained-offline-value-type` / `--pretrained-offline-value-path`   |         | Strategy B: Pretrained value/critic                                             |
| `--control-prior-type` / `--control-prior-path`                         |         | Strategy C: Control prior for reference actions                                 |
| `--save-buffer`                                                         |         | Save online replay buffer under `demos/`                                        |


`python -m rainbow_demorl.train --help` lists everything (sim parameters, env parameters, algorithm-specific hyperparams, etc). For manual inspection, all arguments and hyperparam defaults are listed at `rainbow_demorl/utils/common.py`.

## Environments

Any registered [ManiSkill](https://github.com/haosulab/ManiSkill) env (e.g. `PickCube-v1`, `PushCube-v1`, `StackCube-v1`) can be used. We provide examples of custom variants of PickCube in [`rainbow_demorl/envs/maniskill.py`](rainbow_demorl/envs/maniskill.py) as reference for creating your own versions of ManiSkill tasks.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.