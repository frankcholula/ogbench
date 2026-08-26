"""Persistent JAX policy worker for the nwm eval bridge; runs in this repo's venv, talks via stdin/stdout.

  python nwm_policy_worker.py <ckpt> [action_dim]
<ckpt> is a save_agent output: either the run dir (flags.json + params_*.pkl, latest epoch used) or a
params_<epoch>.pkl inside it. Protocol: prints READY, then one request path per stdin line ->
npz {obs (B,...), goal (B,...)} -> writes <path>.out.npy (B,A) fp32, prints the path back.
The agent is built on the first request (its example batch needs the live observation shape).
"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flax
import jax
import ml_collections
import numpy as np
from agents import agents


def _tuples(x):
    """JSON round-trips tuples as lists; flax module fields must stay hashable."""
    if isinstance(x, dict):
        return {k: _tuples(v) for k, v in x.items()}
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def _resolve(ckpt):
    """-> (params pickle path, flags.json dict)."""
    p = Path(ckpt)
    if p.is_dir():
        cands = sorted(glob.glob(str(p / "params_*.pkl")), key=lambda f: int(Path(f).stem.split("_")[-1]))
        assert cands, f"no params_*.pkl in {p}"
        p = Path(cands[-1])
    return p, json.load(open(p.parent / "flags.json"))


def _action_dim(sd, path=""):
    """Infer the action dim from the actor's mean_net kernel in a saved flax state dict."""
    for k, v in sd.items():
        if not isinstance(v, dict):
            continue
        if k == "mean_net" and "kernel" in v and "high" not in path:
            return v["kernel"].shape[-1]
        got = _action_dim(v, f"{path}/{k}")
        if got:
            return got
    return None


def main():
    pkl, flag_dict = _resolve(sys.argv[1])
    config = ml_collections.ConfigDict(_tuples(flag_dict["agent"]))
    assert not config.get("discrete", False), "discrete actions are not supported by the bridge"
    state = pickle.load(open(pkl, "rb"))["agent"]
    A = int(sys.argv[2]) if len(sys.argv) > 2 else (_action_dim(state) or 5)
    seed = int(flag_dict.get("seed", 0))
    agent, actor_fn = None, None
    print("READY", flush=True)

    for line in sys.stdin:
        req = line.strip()
        if not req:
            continue
        z = np.load(req)
        obs, goal = z["obs"], z["goal"]
        if agent is None:
            agent = agents[config["agent_name"]].create(seed, obs[:1], np.zeros((1, A), np.float32), config)
            agent = flax.serialization.from_state_dict(agent, state)
            print(f"restored {pkl} agent={config['agent_name']} A={A} obs{obs.shape[1:]}", file=sys.stderr, flush=True)
            rng = jax.random.PRNGKey(seed)

            def actor_fn(observations, goals):  # utils.evaluation.supply_rng, deterministic actions
                nonlocal rng
                rng, key = jax.random.split(rng)
                return agent.sample_actions(observations=observations, goals=goals, seed=key, temperature=0)
        act = np.asarray(actor_fn(obs, goal), np.float32)
        np.save(req + ".out.npy", act)
        print(req, flush=True)


if __name__ == "__main__":
    main()
