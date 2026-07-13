import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np
import robosuite as suite


PROPRIO_KEY = "robot0_proprio-state"
STATE_KEY = "object-state"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Panda Block Stacking trajectories from robosuite."
    )
    parser.add_argument("--output", type=Path, default=Path("robosuite_stack_panda.h5"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["random", "scripted"], default="scripted")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.03,
        help="Uniform action noise added to scripted actions.",
    )
    parser.add_argument(
        "--no-flip-images",
        action="store_true",
        help="Store raw robosuite camera frames without vertical flipping.",
    )
    return parser.parse_args()


def make_env(args):
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
    )


def ensure_array(value, dtype=np.float32):
    return np.asarray(value, dtype=dtype)


def make_action_towards(obs, target_pos, gripper, gain=12.0):
    eef_pos = ensure_array(obs["robot0_eef_pos"])
    delta = gain * (ensure_array(target_pos) - eef_pos)
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(delta, -1.0, 1.0)
    action[6] = gripper
    return action


def scripted_stack_action(obs, step):
    cube_a = ensure_array(obs["cubeA_pos"])
    cube_b = ensure_array(obs["cubeB_pos"])

    # Panda with robosuite 1.5's BASIC composite controller uses positive
    # gripper action to close and negative action to open.
    open_gripper = -1.0
    close_gripper = 1.0

    if step < 35:
        target = cube_a + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        return make_action_towards(obs, target, open_gripper)
    if step < 65:
        target = cube_a + np.array([0.0, 0.0, 0.015], dtype=np.float32)
        return make_action_towards(obs, target, open_gripper)
    if step < 90:
        target = cube_a + np.array([0.0, 0.0, 0.015], dtype=np.float32)
        return make_action_towards(obs, target, close_gripper)
    if step < 125:
        target = cube_a + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        return make_action_towards(obs, target, close_gripper)
    if step < 165:
        target = cube_b + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        return make_action_towards(obs, target, close_gripper)
    if step < 190:
        target = cube_b + np.array([0.0, 0.0, 0.055], dtype=np.float32)
        return make_action_towards(obs, target, close_gripper)

    target = cube_b + np.array([0.0, 0.0, 0.08], dtype=np.float32)
    return make_action_towards(obs, target, open_gripper)


def policy_action(env, obs, step, args, rng):
    if args.policy == "random":
        low, high = env.action_spec
        return rng.uniform(low, high).astype(np.float32)

    action = scripted_stack_action(obs, step)
    if args.noise > 0:
        action += rng.uniform(-args.noise, args.noise, size=action.shape).astype(np.float32)
    return np.clip(action, -1.0, 1.0)


class H5Writer:
    def __init__(self, path):
        self.path = path
        self.file = None
        self.datasets = {}
        self.size = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.file is not None:
            self.file.attrs["num_steps"] = self.size
            self.file.close()

    def _require_dataset(self, key, value):
        if key in self.datasets:
            return self.datasets[key]

        arr = np.asarray(value)
        maxshape = (None,) + arr.shape
        chunks = (1,) + arr.shape if arr.ndim > 0 else (1024,)
        dataset = self.file.create_dataset(
            key,
            shape=(0,) + arr.shape,
            maxshape=maxshape,
            chunks=chunks,
            dtype=arr.dtype,
            compression="gzip" if arr.ndim >= 3 else None,
            compression_opts=4 if arr.ndim >= 3 else None,
        )
        self.datasets[key] = dataset
        return dataset

    def append(self, row):
        for key, value in row.items():
            dataset = self._require_dataset(key, value)
            dataset.resize(self.size + 1, axis=0)
            dataset[self.size] = value
        self.size += 1


def get_pixels(obs, camera, flip_images=True):
    pixels = obs[f"{camera}_image"].astype(np.uint8, copy=False)
    if flip_images:
        # MuJoCo offscreen camera frames arrive vertically inverted. Store the
        # corrected, contiguous image in HDF5 so training reads upright pixels.
        return np.flip(pixels, axis=0).copy()
    return pixels.copy()


def build_row(obs, action, reward, done, success, episode_idx, step_idx, camera, flip_images=True):
    return {
        "pixels": get_pixels(obs, camera, flip_images),
        "action": ensure_array(action),
        "proprio": ensure_array(obs[PROPRIO_KEY]),
        "state": ensure_array(obs[STATE_KEY]),
        "reward": np.asarray(reward, dtype=np.float32),
        "done": np.asarray(done, dtype=np.bool_),
        "success": np.asarray(success, dtype=np.bool_),
        "episode_idx": np.asarray(episode_idx, dtype=np.int64),
        "step_idx": np.asarray(step_idx, dtype=np.int64),
        "cubeA_pos": ensure_array(obs["cubeA_pos"]),
        "cubeA_quat": ensure_array(obs["cubeA_quat"]),
        "cubeB_pos": ensure_array(obs["cubeB_pos"]),
        "cubeB_quat": ensure_array(obs["cubeB_quat"]),
        "eef_pos": ensure_array(obs["robot0_eef_pos"]),
        "eef_quat": ensure_array(obs["robot0_eef_quat"]),
        "gripper_qpos": ensure_array(obs["robot0_gripper_qpos"]),
        "gripper_qvel": ensure_array(obs["robot0_gripper_qvel"]),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = make_env(args)

    with H5Writer(args.output) as writer:
        writer.file.attrs["env_name"] = "Stack"
        writer.file.attrs["robot"] = "Panda"
        writer.file.attrs["camera"] = args.camera
        writer.file.attrs["policy"] = args.policy
        writer.file.attrs["horizon"] = args.horizon
        writer.file.attrs["control_freq"] = args.control_freq
        writer.file.attrs["images_flipped_vertical"] = not args.no_flip_images
        writer.file.attrs["image_orientation"] = (
            "upright_vertical_flip_applied"
            if not args.no_flip_images
            else "raw_robosuite_camera"
        )

        successes = 0
        for episode_idx in range(args.episodes):
            obs = env.reset()
            episode_success = False

            for step_idx in range(args.horizon):
                action = policy_action(env, obs, step_idx, args, rng)
                next_obs, reward, done, _ = env.step(action)
                success = bool(env._check_success())
                episode_success = episode_success or success

                row = build_row(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    success=success,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    camera=args.camera,
                    flip_images=not args.no_flip_images,
                )
                writer.append(row)

                obs = next_obs
                if done:
                    break

            successes += int(episode_success)
            print(
                f"episode {episode_idx + 1}/{args.episodes}: "
                f"success={episode_success} total_rows={writer.size}",
                flush=True,
            )

    env.close()
    print(f"wrote {args.output} with {successes}/{args.episodes} successful episodes")


if __name__ == "__main__":
    main()
