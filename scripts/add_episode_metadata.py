#!/usr/bin/env python3
"""Add stable-worldmodel episode metadata to a collected HDF5 dataset."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--episode-key", default="episode_idx")
    args = parser.parse_args()

    with h5py.File(args.path, "r+") as h5:
        if args.episode_key not in h5:
            raise KeyError(f"Missing episode column {args.episode_key!r}")

        episode_idx = h5[args.episode_key][:]
        _, first_indices, lengths = np.unique(
            episode_idx, return_index=True, return_counts=True
        )
        order = np.argsort(first_indices)
        lengths = lengths[order].astype(np.int64)
        offsets = np.zeros(len(lengths), dtype=np.int64)
        if len(lengths) > 1:
            offsets[1:] = np.cumsum(lengths[:-1])

        for key in ("ep_len", "ep_offset"):
            if key in h5:
                del h5[key]
        h5.create_dataset("ep_len", data=lengths)
        h5.create_dataset("ep_offset", data=offsets)
        h5.flush()

        print(f"episodes: {len(lengths):,}")
        print(f"steps: {int(lengths.sum()):,}")
        values, counts = np.unique(lengths, return_counts=True)
        print("episode lengths:", dict(zip(values.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
