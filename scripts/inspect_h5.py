#!/usr/bin/env python3
"""Inspect an HDF5 dataset without loading it all into memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def short(value: Any, limit: int = 160) -> str:
    """Return a compact, robust representation for metadata values."""
    try:
        text = repr(value)
    except Exception as exc:  # Corrupt attributes can fail while decoding.
        return f"<unreadable: {exc}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_attrs(obj: h5py.Group | h5py.Dataset, indent: str) -> None:
    try:
        names = list(obj.attrs.keys())
    except Exception as exc:
        print(f"{indent}attributes: <unreadable: {exc}>")
        return
    for name in names:
        try:
            value = obj.attrs[name]
        except Exception as exc:
            value = f"<unreadable: {exc}>"
        print(f"{indent}@{name} = {short(value)}")


def sample_dataset(dataset: h5py.Dataset, rows: int) -> None:
    if rows <= 0 or dataset.size == 0:
        return
    try:
        if dataset.ndim == 0:
            data = np.asarray(dataset[()])
        else:
            data = np.asarray(dataset[: min(rows, dataset.shape[0])])
        print(f"    sample shape={data.shape}")
        if np.issubdtype(data.dtype, np.number):
            finite = data[np.isfinite(data)]
            if finite.size:
                print(
                    "    sample stats: "
                    f"min={finite.min():.6g} max={finite.max():.6g} "
                    f"mean={finite.mean():.6g} finite={finite.size}/{data.size}"
                )
        else:
            print(f"    sample={short(data)}")
    except Exception as exc:
        print(f"    sample: <unreadable: {exc}>")


def walk(
    group: h5py.Group,
    *,
    depth: int,
    max_items: int,
    attrs: bool,
    sample_rows: int,
    level: int = 0,
) -> bool:
    if level >= depth:
        return True
    try:
        names = list(group.keys())
    except Exception as exc:
        print(f"  {'  ' * level}<group contents unreadable: {exc}>")
        return False

    healthy = True
    for index, name in enumerate(names):
        indent = "  " * (level + 1)
        if index >= max_items:
            print(f"{indent}... {len(names) - max_items} more item(s)")
            break
        try:
            obj = group[name]
            if isinstance(obj, h5py.Group):
                print(f"{indent}[group] {obj.name}")
                if attrs:
                    print_attrs(obj, indent + "  ")
                healthy = walk(
                    obj,
                    depth=depth,
                    max_items=max_items,
                    attrs=attrs,
                    sample_rows=sample_rows,
                    level=level + 1,
                ) and healthy
            else:
                storage = (
                    f", chunks={obj.chunks}, compression={obj.compression}"
                    if obj.ndim > 0
                    else ""
                )
                print(
                    f"{indent}[dataset] {obj.name}: shape={obj.shape}, "
                    f"dtype={obj.dtype}{storage}"
                )
                if attrs:
                    print_attrs(obj, indent + "  ")
                sample_dataset(obj, sample_rows)
        except Exception as exc:
            print(f"{indent}{name}: <unreadable: {exc}>")
            healthy = False
    return healthy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to an .h5/.hdf5 file")
    parser.add_argument("--depth", type=int, default=4, help="Maximum group depth")
    parser.add_argument(
        "--max-items", type=int, default=50, help="Maximum children shown per group"
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Read this many leading rows from every dataset (default: metadata only)",
    )
    parser.add_argument("--attrs", action="store_true", help="Print HDF5 attributes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.path.is_file():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    print(f"file: {args.path.resolve()}")
    print(f"size: {args.path.stat().st_size:,} bytes")
    try:
        with h5py.File(args.path, "r") as handle:
            print(f"HDF5 library: {h5py.version.hdf5_version}")
            if args.attrs:
                print_attrs(handle, "  ")
            healthy = walk(
                handle,
                depth=max(1, args.depth),
                max_items=max(1, args.max_items),
                attrs=args.attrs,
                sample_rows=max(0, args.sample_rows),
            )
            if not healthy:
                print(
                    "error: one or more HDF5 groups are unreadable; do not train on "
                    "this file",
                    file=sys.stderr,
                )
                return 1
    except Exception as exc:
        print(f"error: HDF5 structure is unreadable: {exc}", file=sys.stderr)
        print(
            "The file may be incomplete, still open in a writer process, or corrupt. "
            "Finish/close the writer or copy the dataset again before training.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
