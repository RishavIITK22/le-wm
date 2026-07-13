import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export HDF5 pixels to an MP4 video.")
    parser.add_argument("input", type=Path, help="Path to an HDF5 dataset.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--pixels-key", type=str, default="pixels")
    parser.add_argument("--episode-key", type=str, default="episode_idx")
    parser.add_argument("--step-key", type=str, default="step_idx")
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Force a vertical flip while exporting. Use only for old raw/inverted datasets.",
    )
    return parser.parse_args()


def select_episode_indices(h5, args):
    num_frames = h5[args.pixels_key].shape[0]
    if args.episode_key not in h5:
        return np.arange(num_frames)

    episode_idx = h5[args.episode_key][:]
    indices = np.flatnonzero(episode_idx == args.episode)
    if indices.size == 0:
        available = np.unique(episode_idx)
        raise ValueError(
            f"Episode {args.episode} not found. Available episodes: {available[:20]}"
        )

    if args.step_key in h5:
        steps = h5[args.step_key][indices]
        indices = indices[np.argsort(steps)]
    return indices


def as_uint8_rgb(frame):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3:
        raise ValueError(f"Expected image frame with 3 dims, got shape {frame.shape}")
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame, got shape {frame.shape}")
    return frame


def main():
    args = parse_args()
    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}_ep{args.episode}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as h5:
        if args.pixels_key not in h5:
            keys = ", ".join(sorted(h5.keys()))
            raise KeyError(f"Missing key '{args.pixels_key}'. Available keys: {keys}")

        indices = select_episode_indices(h5, args)
        stored_upright = bool(h5.attrs.get("images_flipped_vertical", False))
        if args.flip_vertical and stored_upright:
            print("warning: dataset says pixels are already upright; applying --flip-vertical anyway")

        first = as_uint8_rgb(h5[args.pixels_key][indices[0]])
        if args.flip_vertical:
            first = np.flip(first, axis=0).copy()
        height, width = first.shape[:2]

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output}")

        try:
            for idx in indices:
                rgb = as_uint8_rgb(h5[args.pixels_key][idx])
                if args.flip_vertical:
                    rgb = np.flip(rgb, axis=0).copy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

    print(f"wrote {output} ({len(indices)} frames)")


if __name__ == "__main__":
    main()
