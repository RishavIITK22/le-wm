#!/usr/bin/env python3
"""Evaluate a trained LeWM checkpoint with MPC in robosuite Stack."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir(), "numba-cache")))

import cv2
import gymnasium as gym
import h5py
import numpy as np
import robosuite as suite
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=1)
    parser.add_argument("--cem-samples", type=int, default=128)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--cem-topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--output", type=Path, default=Path("eval_robosuite_stack"))
    return parser.parse_args()


def image_transform(height: int, width: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize((height, width), antialias=True),
        ]
    )


def upright_pixels(obs: dict, camera: str) -> np.ndarray:
    """Match the upright orientation used by collect_dataset.py."""
    return np.flip(obs[f"{camera}_image"], axis=0).copy().astype(np.uint8)


def load_goal_and_action_stats(path: Path, seed: int):
    """Use the final frame of a demonstrated successful episode as visual goal."""
    with h5py.File(path, "r") as h5:
        required = {"pixels", "action", "episode_idx", "success"}
        missing = required.difference(h5.keys())
        if missing:
            raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")

        episode_idx = h5["episode_idx"][:]
        success = h5["success"][:].astype(bool)
        successful_episodes = np.unique(episode_idx[success])
        if not len(successful_episodes):
            raise ValueError("Dataset has no successful episodes")

        rng = np.random.default_rng(seed)
        goal_episode = int(rng.choice(successful_episodes))
        rows = np.flatnonzero(episode_idx == goal_episode)
        success_rows = rows[success[rows]]
        goal_row = int(success_rows[-1] if len(success_rows) else rows[-1])
        goal = h5["pixels"][goal_row]

        # Match train.py's torch.std default (sample standard deviation).
        actions = h5["action"][:].astype(np.float32)
        action_mean = actions.mean(axis=0)
        action_std = actions.std(axis=0, ddof=1)
        action_std = np.where(action_std < 1e-6, 1.0, action_std)

    return goal, action_mean, action_std, goal_episode, goal_row


def make_env(args: argparse.Namespace):
    controller = suite.load_composite_controller_config(controller="BASIC")
    return suite.make(
        env_name="Stack",
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        reward_shaping=True,
        control_freq=args.control_freq,
        horizon=args.max_steps,
        ignore_done=False,
    )


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    try:
        for rgb in frames:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    if args.horizon < args.history:
        raise ValueError("--horizon must be at least --history")
    if args.cem_topk > args.cem_samples:
        raise ValueError("--cem-topk cannot exceed --cem-samples")
    args.output.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = swm.wm.utils.load_pretrained(str(args.checkpoint.resolve()))
    model = model.to(device).eval().requires_grad_(False)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = model.to(dtype=dtype)

    goal, action_mean, action_std, goal_episode, goal_row = (
        load_goal_and_action_stats(args.dataset, args.seed)
    )
    preprocess = image_transform(args.height, args.width)

    plan_cfg = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        history_len=args.history,
        action_block=args.action_block,
        warm_start=True,
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.cem_samples,
        var_scale=1.0,
        n_steps=args.cem_iters,
        topk=args.cem_topk,
        device=device,
        seed=args.seed,
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1, action_mean.size), dtype=np.float32
    )
    solver.configure(action_space=action_space, n_envs=1, config=plan_cfg)

    def prep_image(image: np.ndarray) -> torch.Tensor:
        return preprocess(image).to(device=device, dtype=dtype)

    goal_tensor = prep_image(goal).unsqueeze(0).unsqueeze(0)
    env = make_env(args)
    results = []

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            first = upright_pixels(obs, args.camera)
            history = deque([first.copy() for _ in range(args.history)], maxlen=args.history)
            frames = [first]
            success = False
            total_reward = 0.0
            steps = 0

            while steps < args.max_steps and not success:
                pixels = torch.stack([prep_image(frame) for frame in history])
                info = {
                    "pixels": pixels.unsqueeze(0),
                    "goal": goal_tensor,
                }
                solved = solver.solve(info)
                normalized = solved["actions"][0, : args.receding_horizon]
                normalized = normalized.reshape(-1, action_mean.size).numpy()
                actions = normalized * action_std + action_mean
                actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

                for action in actions:
                    if steps >= args.max_steps:
                        break
                    obs, reward, done, _ = env.step(action)
                    total_reward += float(reward)
                    steps += 1
                    frame = upright_pixels(obs, args.camera)
                    frames.append(frame)
                    success = bool(env._check_success())
                    if success or done:
                        break

                history.append(frames[-1])

            video = args.output / f"episode_{episode:03d}.mp4"
            write_video(video, frames, args.control_freq)
            record = {
                "episode": episode,
                "success": success,
                "steps": steps,
                "reward": total_reward,
                "video": str(video),
            }
            results.append(record)
            print(json.dumps(record), flush=True)
    finally:
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "goal_episode": goal_episode,
        "goal_row": goal_row,
        "episodes": args.episodes,
        "successes": sum(r["success"] for r in results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
