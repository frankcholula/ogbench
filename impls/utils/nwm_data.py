import os

import numpy as np

from utils.datasets import Dataset

FILES = ('observations.npy', 'actions.npy', 'terminals.npy')


def _load_split(path):
    """Load one memmap export dir as a compact Dataset (observations + terminals + valids, no next_observations)."""
    path = os.path.expanduser(path)
    for f in FILES:
        assert os.path.exists(os.path.join(path, f)), f'missing {f} in {path}'

    # NWM_MEMMAP=0 reads the arrays into RAM: random memmap reads over a network filesystem (cluster) run ~10x
    # slower than the update step; the tok train split is ~11 GB, so budget the job memory accordingly.
    mmap = None if os.environ.get('NWM_MEMMAP', '1') == '0' else 'r'
    observations = np.load(os.path.join(path, 'observations.npy'), mmap_mode=mmap)
    actions = np.load(os.path.join(path, 'actions.npy'), mmap_mode=mmap)
    terminals = np.load(os.path.join(path, 'terminals.npy')).astype(np.float32)

    assert len(observations) == len(actions) == len(terminals), 'length mismatch across arrays'
    assert terminals[-1] > 0, 'the final frame must terminate an episode'
    if actions.dtype != np.float32:
        actions = actions.astype(np.float32)

    # Same transform as ogbench.load_dataset(compact_dataset=True): invalidate each episode's last frame so
    # next_observations[t] == observations[t + 1] is always safe, and pull the terminal flag back one step.
    valids = 1.0 - terminals
    terminals = np.minimum(terminals + np.concatenate([terminals[1:], [1.0]]), 1.0)

    return Dataset.create(observations=observations, actions=actions, terminals=terminals, valids=valids)


def load_nwm_dataset(train_dir, val_dir=None):
    """Load an nwm memmap export pair; observations stay memmapped (the token modality is tens of GB)."""
    train_dataset = _load_split(train_dir)
    val_dataset = _load_split(val_dir) if val_dir is not None else None
    return train_dataset, val_dataset
