"""Train PPO for 3D joint-angle palm-down reaching."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)

# Allow running this folder as a standalone subproject from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from joint_angle_3d_palm_rl.env import JointAnglePalmDownReach3DEnv  # noqa: E402


class SpeedPrintCallback(BaseCallback):
    """Minimal throughput/progress logger."""

    def __init__(self, every_steps: int = 2_000, verbose: int = 1):
        super().__init__(verbose)
        self.every_steps = int(max(1, every_steps))
        self._last_t = 0.0
        self._last_n = 0

    def _on_training_start(self) -> None:
        self._last_t = time.time()
        self._last_n = 0

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if n - self._last_n < self.every_steps:
            return True
        now = time.time()
        dt = max(now - self._last_t, 1e-9)
        sps = (n - self._last_n) / dt
        self._last_t = now
        self._last_n = n
        if self.verbose:
            ep_rew = float(self.logger.name_to_value.get("rollout/ep_rew_mean", float("nan")))
            ep_len = float(self.logger.name_to_value.get("rollout/ep_len_mean", float("nan")))
            if (not np.isfinite(ep_rew)) or (not np.isfinite(ep_len)):
                buf = list(getattr(self.model, "ep_info_buffer", []) or [])
                if buf:
                    rews = [float(d.get("r", np.nan)) for d in buf if "r" in d]
                    lens = [float(d.get("l", np.nan)) for d in buf if "l" in d]
                    if rews:
                        ep_rew = float(np.nanmean(np.asarray(rews, dtype=np.float64)))
                    if lens:
                        ep_len = float(np.nanmean(np.asarray(lens, dtype=np.float64)))
            print(f"[train] steps={n:,} sps={sps:,.1f} ep_rew={ep_rew:.3f} ep_len={ep_len:.1f}")
        return True


class BestModelByDistanceCallback(BaseCallback):
    """Evaluate periodically and save best model by success/final-distance score."""

    def __init__(
        self,
        eval_env,
        run_dir: Path,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 5,
        tracking_eval_horizon: int = 400,
        tracking_eval_cmd_every: int = 60,
        tracking_eval_project_to_reachable: bool = True,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.run_dir = Path(run_dir)
        self.eval_freq = int(max(1, eval_freq))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.tracking_eval_horizon = int(max(1, tracking_eval_horizon))
        self.tracking_eval_cmd_every = int(max(1, tracking_eval_cmd_every))
        self.tracking_eval_project_to_reachable = bool(tracking_eval_project_to_reachable)
        self.deterministic = bool(deterministic)
        self.last_eval_timestep = 0
        self.best_score = -float("inf")
        self.best_model_path = self.run_dir / "best_model.zip"
        self.best_vec_path = self.run_dir / "best_vecnormalize.pkl"

    def _evaluate_once(self):
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except Exception:
                pass
            self.eval_env.training = False
            self.eval_env.norm_reward = False

        obs = self.eval_env.reset()
        ep_rewards = np.zeros(self.eval_env.num_envs, dtype=np.float64)
        final_dists = []
        successes = []
        out_of_controls = []
        final_line_devs = []
        final_directness = []
        episodes = 0

        while episodes < self.n_eval_episodes:
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, rewards, dones, infos = self.eval_env.step(actions)
            ep_rewards += np.asarray(rewards, dtype=np.float64).reshape(-1)
            for i, done in enumerate(dones):
                if not bool(done):
                    continue
                info = infos[i]
                final_dists.append(float(info.get("distance_error", np.nan)))
                successes.append(float(bool(info.get("success", False))))
                out_of_controls.append(float(bool(info.get("out_of_control", False))))
                final_line_devs.append(float(info.get("line_deviation", np.nan)))
                final_directness.append(float(info.get("directness_error", np.nan)))
                episodes += 1
                ep_rewards[i] = 0.0
                if episodes >= self.n_eval_episodes:
                    break

        mean_final_dist = float(np.nanmean(final_dists)) if final_dists else float("inf")
        success_rate = float(np.mean(successes)) if successes else 0.0
        out_of_control_rate = float(np.mean(out_of_controls)) if out_of_controls else 0.0
        mean_line_dev = float(np.nanmean(final_line_devs)) if final_line_devs else 0.0
        mean_directness = float(np.nanmean(final_directness)) if final_directness else 0.0
        return mean_final_dist, success_rate, out_of_control_rate, mean_line_dev, mean_directness

    def _unwrap_single_eval_env(self):
        env = self.eval_env
        while hasattr(env, "venv"):
            env = env.venv
        if not hasattr(env, "envs") or len(env.envs) == 0:
            return None
        raw = env.envs[0]
        # Unwrap gym wrappers but stop at our custom env (it also has `.env` internally).
        while hasattr(raw, "env") and (not hasattr(raw, "set_target")):
            raw = raw.env
        return raw

    def _evaluate_tracking_once(self):
        raw_env = self._unwrap_single_eval_env()
        if raw_env is None or not hasattr(raw_env, "set_target"):
            return float("nan"), float("nan")

        obs = self.eval_env.reset()
        pts = np.asarray(getattr(raw_env, "_reachable_points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            try:
                spec = raw_env._resolve_joint_spec()
                pts = raw_env._sample_reachable_points(spec, 128)
                raw_env._reachable_points = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
            except Exception:
                pts = np.zeros((0, 3), dtype=np.float64)
        if pts.shape[0] == 0:
            return float("nan"), float("nan")

        rng = np.random.default_rng(12345 + int(self.model.num_timesteps))
        dists = []
        for t in range(self.tracking_eval_horizon):
            if (t % self.tracking_eval_cmd_every) == 0:
                cmd = np.asarray(pts[int(rng.integers(0, pts.shape[0]))], dtype=np.float64).reshape(3)
                try:
                    raw_env.set_target(
                        cmd,
                        project_to_reachable=bool(self.tracking_eval_project_to_reachable),
                        line_from_tip=True,
                    )
                except Exception:
                    pass

            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, rewards, dones, infos = self.eval_env.step(actions)
            _ = rewards
            info = infos[0] if infos else {}
            d = float(info.get("distance_error", np.nan))
            if np.isfinite(d):
                dists.append(d)
            if bool(dones[0]):
                obs = self.eval_env.reset()

        if not dists:
            return float("nan"), float("nan")
        arr = np.asarray(dists, dtype=np.float64)
        return float(np.nanmean(arr)), float(np.nanpercentile(arr, 90.0))

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if (n - self.last_eval_timestep) < self.eval_freq:
            return True
        self.last_eval_timestep = n

        mean_final_dist, success_rate, out_of_control_rate, mean_line_dev, mean_directness = self._evaluate_once()
        tracking_mean_dist, tracking_p90_dist = self._evaluate_tracking_once()
        tracking_mean_term = float(tracking_mean_dist) if np.isfinite(tracking_mean_dist) else 0.0
        tracking_p90_term = float(tracking_p90_dist) if np.isfinite(tracking_p90_dist) else 0.0
        score = (
            (10.0 * success_rate)
            - (1.0 * mean_final_dist)
            - (4.0 * out_of_control_rate)
            - (0.8 * mean_line_dev)
            - (0.5 * mean_directness)
            - (4.0 * tracking_mean_term)
            - (2.0 * tracking_p90_term)
        )
        self.logger.record("eval/mean_final_distance", float(mean_final_dist))
        self.logger.record("eval/success_rate", float(success_rate))
        self.logger.record("eval/out_of_control_rate", float(out_of_control_rate))
        self.logger.record("eval/mean_line_deviation", float(mean_line_dev))
        self.logger.record("eval/mean_directness_error", float(mean_directness))
        self.logger.record("eval/tracking_mean_distance", float(tracking_mean_dist))
        self.logger.record("eval/tracking_p90_distance", float(tracking_p90_dist))
        self.logger.record("eval/score", float(score))

        if score > self.best_score + 1e-12:
            self.best_score = score
            try:
                self.model.save(str(self.best_model_path))
            except Exception:
                pass
            if isinstance(self.training_env, VecNormalize):
                try:
                    self.training_env.save(str(self.best_vec_path))
                except Exception:
                    pass
            if self.verbose:
                print(
                    f"[best] step={n:,} score={score:.4f} "
                    f"success_rate={success_rate:.3f} mean_final_dist={mean_final_dist:.4f} "
                    f"ooc={out_of_control_rate:.3f} line_dev={mean_line_dev:.4f} direct={mean_directness:.4f} "
                    f"track_mean={tracking_mean_dist:.4f} track_p90={tracking_p90_dist:.4f}"
                )
        elif self.verbose:
            print(
                f"[eval] step={n:,} score={score:.4f} "
                f"success_rate={success_rate:.3f} mean_final_dist={mean_final_dist:.4f} "
                f"ooc={out_of_control_rate:.3f} line_dev={mean_line_dev:.4f} direct={mean_directness:.4f} "
                f"track_mean={tracking_mean_dist:.4f} track_p90={tracking_p90_dist:.4f} "
                f"(best={self.best_score:.4f})"
            )
        return True


def linear_schedule(initial: float):
    init = float(initial)

    def _schedule(progress_remaining: float) -> float:
        return float(progress_remaining * init)

    return _schedule


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401

        return True
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PPO on 3D joint-angle palm-down reaching")

    # Runtime
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--tb-log-dir", type=str, default=None, help="TensorBoard root log dir (default: <run_dir>/tensorboard)")
    p.add_argument("--tb-run-name", type=str, default="", help="TensorBoard run name (default: run-dir folder name)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--vec-env", type=str, default="auto", choices=("auto", "dummy", "subproc"))
    p.add_argument("--no-vecnorm", action="store_true")
    p.add_argument("--log-every-steps", type=int, default=2_000)

    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-anneal", dest="lr_anneal", action="store_true", default=True)
    p.add_argument("--no-lr-anneal", dest="lr_anneal", action="store_false")
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.00)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--eval-freq", type=int, default=10_000, help="evaluation frequency in env steps")
    p.add_argument("--eval-episodes", type=int, default=5, help="number of episodes per evaluation")
    p.add_argument("--tracking-eval-horizon", type=int, default=400, help="steps for drag-like tracking evaluation")
    p.add_argument("--tracking-eval-cmd-every", type=int, default=60, help="target-update period in tracking eval")
    p.add_argument(
        "--tracking-eval-project-targets-to-reachable",
        dest="tracking_eval_project_targets_to_reachable",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-tracking-eval-project-targets-to-reachable",
        dest="tracking_eval_project_targets_to_reachable",
        action="store_false",
    )

    # Environment
    p.add_argument("--env-id", type=str, default="myoArmReachRandom-v0")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument(
        "--joint-profile",
        type=str,
        default="opensim_arm_wrist",
        choices=("opensim_arm", "opensim_arm_wrist", "opensim_full_shoulder_wrist"),
    )
    p.add_argument("--joint-names", type=str, default="", help="comma-separated joint names overriding --joint-profile")
    p.add_argument("--action-max-delta-deg", type=float, default=5.0)
    p.add_argument("--enable-root-yaw", dest="enable_root_yaw", action="store_true", default=True)
    p.add_argument("--disable-root-yaw", dest="enable_root_yaw", action="store_false")
    p.add_argument("--root-body-name", type=str, default="full_body")
    p.add_argument("--root-yaw-limit-deg", type=float, default=75.0)
    p.add_argument("--root-yaw-step-deg", type=float, default=3.0)
    p.add_argument("--root-yaw-action-mode", type=str, default="absolute", choices=("delta", "absolute"))
    p.add_argument("--root-yaw-smooth-alpha", type=float, default=0.25)
    p.add_argument("--target-rollout-samples", type=int, default=96)
    p.add_argument("--target-jitter-std", type=float, default=0.012)
    p.add_argument("--target-min-start-dist", type=float, default=0.10)
    p.add_argument("--target-max-start-dist", type=float, default=0.55)
    p.add_argument("--project-targets-to-reachable", dest="project_targets_to_reachable", action="store_true", default=True)
    p.add_argument("--no-project-targets-to-reachable", dest="project_targets_to_reachable", action="store_false")
    p.add_argument("--retarget-interval-steps", type=int, default=0)
    p.add_argument("--success-radius", type=float, default=0.030)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--distance-weight", type=float, default=2.2)
    p.add_argument("--progress-weight", type=float, default=1.0)
    p.add_argument("--line-deviation-weight", type=float, default=0.25)
    p.add_argument("--yaw-line-weight", type=float, default=0.10)
    p.add_argument("--palm-weight", type=float, default=0.22)
    p.add_argument("--directness-weight", type=float, default=0.18)
    p.add_argument("--stability-weight", type=float, default=0.04)
    p.add_argument("--away-penalty-weight", type=float, default=0.10)
    p.add_argument("--palm-near-target-scale", type=float, default=0.18)
    p.add_argument("--action-cost-weight", type=float, default=0.010)
    p.add_argument("--smoothness-weight", type=float, default=0.020)
    p.add_argument("--joint-limit-weight", type=float, default=0.060)
    p.add_argument("--time-penalty", type=float, default=0.003)
    p.add_argument("--success-bonus", type=float, default=12.0)
    p.add_argument("--reward-clip", type=float, default=0.0, help="0 disables reward clipping")
    p.add_argument("--terminate-on-success", dest="terminate_on_success", action="store_true", default=True)
    p.add_argument("--no-terminate-on-success", dest="terminate_on_success", action="store_false")
    p.add_argument("--palm-lock-alpha", type=float, default=0.0, help="hard blend to palm preset; keep 0 for soft-only")

    return p


def _parse_joint_names(csv_text: str):
    vals = [x.strip() for x in str(csv_text).split(",") if x.strip()]
    return tuple(vals) if vals else None


def make_env(rank: int, args) -> Callable[[], JointAnglePalmDownReach3DEnv]:
    def _init():
        seed = int(args.seed) + int(rank)
        env_make_kwargs = {}
        if args.model_path:
            env_make_kwargs["model_path"] = str(args.model_path)
            env_make_kwargs["obsd_model_path"] = str(args.model_path)
        env = JointAnglePalmDownReach3DEnv(
            env_id=args.env_id,
            env_make_kwargs=env_make_kwargs,
            joint_profile=args.joint_profile,
            joint_names=_parse_joint_names(args.joint_names),
            action_max_delta_deg=float(args.action_max_delta_deg),
            enable_root_yaw=bool(args.enable_root_yaw),
            root_body_name=str(args.root_body_name),
            root_yaw_limit_deg=float(args.root_yaw_limit_deg),
            root_yaw_step_deg=float(args.root_yaw_step_deg),
            root_yaw_action_mode=str(args.root_yaw_action_mode),
            root_yaw_smooth_alpha=float(args.root_yaw_smooth_alpha),
            target_rollout_samples=int(args.target_rollout_samples),
            target_jitter_std=float(args.target_jitter_std),
            target_min_start_dist=float(args.target_min_start_dist),
            target_max_start_dist=float(args.target_max_start_dist),
            project_targets_to_reachable=bool(args.project_targets_to_reachable),
            retarget_interval_steps=int(args.retarget_interval_steps),
            success_radius=float(args.success_radius),
            max_steps=int(args.max_steps),
            distance_weight=float(args.distance_weight),
            progress_weight=float(args.progress_weight),
            line_deviation_weight=float(args.line_deviation_weight),
            yaw_line_weight=float(args.yaw_line_weight),
            palm_weight=float(args.palm_weight),
            directness_weight=float(args.directness_weight),
            stability_weight=float(args.stability_weight),
            away_penalty_weight=float(args.away_penalty_weight),
            palm_near_target_scale=float(args.palm_near_target_scale),
            action_cost_weight=float(args.action_cost_weight),
            smoothness_weight=float(args.smoothness_weight),
            joint_limit_weight=float(args.joint_limit_weight),
            time_penalty=float(args.time_penalty),
            success_bonus=float(args.success_bonus),
            reward_clip=float(args.reward_clip),
            terminate_on_success=bool(args.terminate_on_success),
            palm_lock_alpha=float(args.palm_lock_alpha),
            seed=seed,
        )
        return env

    return _init


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(int(args.seed))

    run_dir = Path(args.run_dir).expanduser() if args.run_dir else (
        _THIS_DIR / "runs" / f"ppo_joint3d_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_dir] {run_dir}")
    tb_log_dir = Path(args.tb_log_dir).expanduser() if args.tb_log_dir else (run_dir / "tensorboard")
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_run_name = str(args.tb_run_name).strip() or run_dir.name
    tb_enabled = _tensorboard_available()
    if tb_enabled:
        print(f"[tensorboard] logdir={tb_log_dir} run={tb_run_name}")
    else:
        print("[warn] tensorboard package is not installed in current env; TensorBoard logging disabled.")

    n_envs = int(max(1, args.n_envs))
    if args.vec_env == "dummy" or (args.vec_env == "auto" and n_envs == 1):
        vec = DummyVecEnv([make_env(i, args) for i in range(n_envs)])
    else:
        vec = SubprocVecEnv([make_env(i, args) for i in range(n_envs)])
    vec = VecMonitor(vec)
    if not args.no_vecnorm:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_vec = DummyVecEnv([make_env(10_000, args)])
    eval_vec = VecMonitor(eval_vec)
    if isinstance(vec, VecNormalize):
        eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    n_steps = int(max(16, args.n_steps))
    batch_size = int(max(16, args.batch_size))
    rollout_size = n_steps * n_envs
    if batch_size > rollout_size:
        batch_size = rollout_size
        print(f"[adjust] batch_size clipped to rollout_size={rollout_size}")

    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=linear_schedule(float(args.lr)) if bool(args.lr_anneal) else float(args.lr),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(args.n_epochs),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        verbose=1,
        seed=int(args.seed),
        device=args.device,
        tensorboard_log=(str(tb_log_dir) if tb_enabled else None),
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    eval_freq = int(max(1, int(args.eval_freq) // max(1, n_envs)))
    callbacks = [
        SpeedPrintCallback(every_steps=int(args.log_every_steps), verbose=1),
        BestModelByDistanceCallback(
            eval_env=eval_vec,
            run_dir=run_dir,
            eval_freq=eval_freq,
            n_eval_episodes=int(max(1, args.eval_episodes)),
            tracking_eval_horizon=int(max(1, args.tracking_eval_horizon)),
            tracking_eval_cmd_every=int(max(1, args.tracking_eval_cmd_every)),
            tracking_eval_project_to_reachable=bool(args.tracking_eval_project_targets_to_reachable),
            deterministic=True,
            verbose=1,
        ),
    ]
    callback = CallbackList(callbacks)
    learn_kwargs = dict(
        total_timesteps=int(args.total_steps),
        callback=callback,
        progress_bar=False,
    )
    if tb_enabled:
        learn_kwargs["tb_log_name"] = tb_run_name
    model.learn(**learn_kwargs)

    model_path = run_dir / "model_final.zip"
    model.save(str(model_path))
    print(f"[saved] model={model_path}")
    best_model_path = run_dir / "best_model.zip"
    if best_model_path.exists():
        print(f"[saved] best_model={best_model_path}")
    else:
        print("[saved] best_model not produced yet (no evaluation run completed)")

    if isinstance(vec, VecNormalize):
        vec_path = run_dir / "vecnormalize.pkl"
        vec.save(str(vec_path))
        print(f"[saved] vecnormalize={vec_path}")
        best_vec_path = run_dir / "best_vecnormalize.pkl"
        if best_vec_path.exists():
            print(f"[saved] best_vecnormalize={best_vec_path}")

    vec.close()
    eval_vec.close()


if __name__ == "__main__":
    main()
