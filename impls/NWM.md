# Running the OGBench reference impls on nwm datasets

The six reference agents (GCBC / GCIVL / GCIQL / QRL / CRL / HIQL) trained offline on our own exports
instead of the official OGBench downloads, in two observation modalities:

| mod | observations | encoder | augmentation |
| --- | --- | --- | --- |
| `pix` | `uint8 (N, 64, 64, 3)` | `impala_small` | `--agent.p_aug=0.5` |
| `tok` | `float16 (N, 2048, 14)`, standardized | `token_pooler` | `--agent.p_aug=0.0` |

Online evaluation is off (`--eval_interval=0`); rollouts happen in a separate process
(`nwm_policy_worker.py` restores a checkpoint and serves actions).

## Venv

Self-contained, inside this fork, separate from the parent repo's `.venv` (never run `uv sync` in the parent):

```bash
cd third_party/ogbench
uv venv .venv --python 3.11
uv pip install -p .venv/bin/python -e . \
    'jax[cuda12]>=0.4.26' 'flax>=0.8.4' 'distrax>=0.1.5' ml_collections matplotlib moviepy wandb
```

Installed: jax/jaxlib 0.10.2 (cuda12 plugin), flax 0.12.8, distrax 0.1.9, ogbench 1.2.1 (editable, local).

**CUDA gotcha.** This machine exports `LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64`, whose `libnvJitLink.so.12`
is too old for the CUDA 12.9 `libcusparse` in the wheels — jax then reports
`Unable to load cuSPARSE` and silently falls back to CPU. `.venv/bin/activate` has been extended to prepend
the wheels' `nvidia/*/lib` dirs, so:

```bash
source .venv/bin/activate
python -c "import jax; print(jax.devices())"   # -> [CudaDevice(id=0)]
```

Without activating (e.g. from a launcher), either prepend those dirs yourself or just drop the system path:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python impls/main.py ...
```

Add `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing the 3090 with another job.

## Datasets

Produced in the parent repo (its own venv), one dir per split:

```bash
PYTHONPATH=src STABLEWM_HOME=/mnt/Data/stable_worldmodel python -m nwm.offlinerl.export_ogb \
    --mod tok --env scene --n-eps 1000 --n-val 20 --out /mnt/Data/nwm-orl/impls
PYTHONPATH=src STABLEWM_HOME=/mnt/Data/stable_worldmodel python -m nwm.offlinerl.export_ogb \
    --mod pix --env scene --n-eps 1000 --n-val 20 --out /mnt/Data/nwm-orl/impls
```

Layout of `<out>/<env>_<mod>_<train|val>/`, read by `utils/nwm_data.py`:

- `observations.npy` — `uint8 (N, 64, 64, 3)` (pix) or `float16 (N, 2048, 14)` (tok), memmapped, never copied into RAM
- `actions.npy` — `float32 (N, A)` in `[-1, 1]`
- `terminals.npy` — `uint8 (N,)`, 1 on each episode's last frame

`load_nwm_dataset` builds **compact** datasets (`observations` / `actions` / `terminals` / `valids`, no
`next_observations`) applying the exact transform `ogbench.load_dataset(compact_dataset=True)` uses:
`valids = 1 - terminals`, then the terminal flag is pulled back one step so `observations[t + 1]` is always a
valid `next_observations[t]`. Frame stacking is rejected (`--agent.frame_stack` must stay unset).

## Command matrix

Per-method extra flags are the official `hyperparameters.sh` values for `visual-scene-play-v0` and
`visual-cube-single-play-v0` (identical for both envs), with online-eval flags replaced by our offline ones.
Set once:

```bash
cd third_party/ogbench/impls
source ../.venv/bin/activate
D=/mnt/Data/nwm-orl/impls        # dataset + checkpoint root
ENV=scene                        # or cube_single
SEED=0                           # seeds 0,1,2 per cell
COMMON="--train_steps=500000 --save_interval=100000 --eval_interval=0 --seed=$SEED --agent.batch_size=256"
```

### pix (`impala_small`, `p_aug=0.5`)

```bash
M="--nwm_train_dir=$D/${ENV}_pix_train --nwm_val_dir=$D/${ENV}_pix_val --agent.encoder=impala_small --agent.p_aug=0.5"

python main.py --agent=agents/gcbc.py  $COMMON $M --run_name=gcbc-pix-$ENV-s$SEED  --save_dir=$D/runs/gcbc-pix-$ENV-s$SEED
python main.py --agent=agents/gcivl.py $COMMON $M --agent.alpha=10.0 --run_name=gcivl-pix-$ENV-s$SEED --save_dir=$D/runs/gcivl-pix-$ENV-s$SEED
python main.py --agent=agents/gciql.py $COMMON $M --agent.alpha=1.0  --run_name=gciql-pix-$ENV-s$SEED --save_dir=$D/runs/gciql-pix-$ENV-s$SEED
python main.py --agent=agents/qrl.py   $COMMON $M --agent.alpha=0.3  --run_name=qrl-pix-$ENV-s$SEED   --save_dir=$D/runs/qrl-pix-$ENV-s$SEED
python main.py --agent=agents/crl.py   $COMMON $M --agent.alpha=3.0  --run_name=crl-pix-$ENV-s$SEED   --save_dir=$D/runs/crl-pix-$ENV-s$SEED
python main.py --agent=agents/hiql.py  $COMMON $M --agent.high_alpha=3.0 --agent.low_alpha=3.0 \
    --agent.low_actor_rep_grad=True --agent.subgoal_steps=10 --run_name=hiql-pix-$ENV-s$SEED --save_dir=$D/runs/hiql-pix-$ENV-s$SEED
```

### tok (`token_pooler`, `p_aug=0.0`)

```bash
M="--nwm_train_dir=$D/${ENV}_tok_train --nwm_val_dir=$D/${ENV}_tok_val --agent.encoder=token_pooler --agent.p_aug=0.0"
```

then the same six lines with `pix` swapped for `tok` in `--run_name` / `--save_dir`.

`GCDataset.augment` only crops arrays with `ndim == 4`, so token batches would skip augmentation even at
`p_aug > 0` — `p_aug=0.0` on tok rows keeps the branch from firing at all.

## Cost and GPU budget

Measured on the 3090 (steady-state step time, `peak_bytes_in_use`), `bs=256` extrapolated linearly from `bs=128`:

| run | bs=64 | bs=128 | bs=256 (est.) | 500k steps @ bs=256 |
| --- | --- | --- | --- | --- |
| gcbc-pix | 8.2 ms / 0.38 GiB | 13.5 ms / 0.62 GiB | ~27 ms / ~1.2 GiB | ~4 h |
| gcbc-tok | 16.0 ms / 1.02 GiB | — | ~64 ms / ~4 GiB | ~9 h |
| hiql-pix | 43.3 ms / 1.81 GiB | 43.1 ms / 1.81 GiB | ~45 ms / ~2 GiB | ~6 h |
| hiql-tok | — | 171.8 ms / 6.61 GiB | ~344 ms / ~13 GiB | ~48 h |

Token runs are the expensive ones: the pooler expands each `(B, 2048, 14)` fp16 batch into `(B, 4096, 128)` fp32
activations, and HIQL instantiates the pooler in seven places (value + target value + low actor state encoders,
the shared `goal_rep` head, the high-actor concat encoder). **hiql-tok at `bs=256` needs roughly half the card**
— check `nvidia-smi` before launching; a sibling torch job holding ~20 GiB will OOM it. Drop to `bs=128` (and
say so in the results table) if the 3090 has to be shared.

## Token encoder

`utils/encoders.TokenPooler`, registered as `'token_pooler'`, 594,304 params. It sees whatever the agents hand
their encoders and infers the number of token sets from the last dim:

- `14` — one token set `(B, 2048, 14)`: QRL's quasimetric encoder, CRL's `critic_state`/`critic_goal`/`value_*`,
  and the `state_encoder` slots in HIQL's value / low-actor `GCEncoder`s.
- `28` — `GCEncoder`'s channel-concat of obs+goal `(B, 2048, 14+14)`: every `concat_encoder`, plus HIQL's
  `goal_rep` head, which is fed `concatenate([observations, goals], -1)` directly.

Architecture (mirror of the torch pooler): shared `Dense 14 -> 128` on each set, plus a learned type embedding
(obs=0, goal=1; a single set uses type 0); sets are unioned along the token axis; `LayerNorm` on the keys/values;
8 learned queries through one `MultiHeadDotProductAttention` (4 heads, d=128); queries flattened to
`Dense 512 + GELU`. Both input widths yield the same parameter tree, so a checkpoint is shape-compatible either way.

## Checkpoints

With `--nwm_train_dir` set, `--save_dir` is used **verbatim** (the stock code appends
`<project>/<run_group>/<exp_name>` — that nesting is skipped so paths stay predictable). `--save_interval=100000`
over `--train_steps=500000` gives:

```
$D/runs/gcbc-tok-scene-s0/
    flags.json          # every flag, incl. the full 'agent' config -- needed to rebuild the agent
    train.csv           # training/ + validation/ metrics at each --log_interval
    params_{100000,200000,300000,400000,500000}.pkl
```

Each pkl is `pickle` of `{'agent': flax.serialization.to_state_dict(agent)}` (see `utils/flax_utils.save_agent`).
Restoring needs a freshly constructed agent of the same shape to deserialize into:

```python
import json, ml_collections, numpy as np, jax
from agents import agents
from utils.flax_utils import restore_agent

ckpt = '/mnt/Data/nwm-orl/impls/runs/gcbc-tok-scene-s0'
config = ml_collections.ConfigDict(json.load(open(f'{ckpt}/flags.json'))['agent'])
ex_obs = np.zeros((1, 2048, 14), np.float16)   # (1, 64, 64, 3) uint8 for pix
ex_act = np.zeros((1, 5), np.float32)
agent = agents[config['agent_name']].create(0, ex_obs, ex_act, config)
agent = restore_agent(agent, ckpt, 500000)     # globs ckpt, opens params_500000.pkl

actions = agent.sample_actions(obs, goals, seed=jax.random.PRNGKey(0), temperature=0)
```

`restore_agent` globs `restore_path` (so it must match exactly one dir) and appends `/params_<epoch>.pkl`.
`nwm_policy_worker.py` is the production version of this: it resolves the latest `params_*.pkl` in a run dir,
infers the action dim from the actor's `mean_net` kernel, and serves actions over stdin/stdout.

Token observations must be standardized with the same per-channel stats the export wrote to
`<out>/<env>_tok_stats.npz` before being fed to a restored tok agent.

## Logging

nwm runs go to the existing W&B project `nwm-policy` — they are *not* a separate project; `job_type='orl'` and
tags `['orl', 'ogbench-impls']` separate them from the other work there. Run name comes from `--run_name`
(falls back to the generated `sd000_<timestamp>` name). `WANDB_MODE=disabled` / `offline` is honored.
Non-nwm runs keep the stock `OGBench` project and directory nesting.

## Changes to the fork

- `utils/nwm_data.py` (new) — `load_nwm_dataset(train_dir, val_dir)`.
- `utils/encoders.py` — `TokenPooler` + `'token_pooler'` in `encoder_modules`.
- `main.py` — `--nwm_train_dir` / `--nwm_val_dir` / `--run_name`; env creation and the online-eval block are
  skipped when nwm dirs are given; `--eval_interval=0` is required there.
- `utils/log_utils.py` — `setup_wandb` takes `job_type` / `tags` and no longer shadows `WANDB_MODE`.
