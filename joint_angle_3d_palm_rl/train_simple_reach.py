"""Train PPO on a simple goal-conditioned 3D reaching task."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from joint_angle_3d_palm_rl.simple_reach_env import SimpleJointReach3DEnv


class SpeedPrintCallback(BaseCallback):
    def __init__(self, every_steps: int = 2_000, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.every_steps = int(max(1, every_steps))
        self._last_t = 0.0
        self._last_n = 0

    def _on_training_start(self) -> None:
        self._last_t = time.time()
        self._last_n = int(getattr(self.model, "num_timesteps", 0))

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if n - self._last_n < self.every_steps:
            return True
        now = time.time()
        dt = max(now - self._last_t, 1e-9)
        sps = (n - self._last_n) / dt
        self._last_t = now
        self._last_n = n
        ep_rew_val = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        ep_len_val = self.logger.name_to_value.get("rollout/ep_len_mean", None)
        if ep_rew_val is None or ep_len_val is None:
            ep_rew_text = "n/a"
            ep_len_text = "n/a"
        else:
            ep_rew_text = f"{float(ep_rew_val):.3f}"
            ep_len_text = f"{float(ep_len_val):.1f}"
        print(f"[train] steps={n:,} sps={sps:,.1f} ep_rew={ep_rew_text} ep_len={ep_len_text}")
        return True


class BestModelByDistanceCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        run_dir: Path,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.run_dir = Path(run_dir)
        self.eval_freq = int(max(1, eval_freq))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.deterministic = bool(deterministic)
        self.last_eval_timestep = 0
        self.best_score = -float("inf")
        self.best_model_path = self.run_dir / "best_model.zip"

    def _on_training_start(self) -> None:
        self.last_eval_timestep = int(getattr(self.model, "num_timesteps", 0))

    def _evaluate_once(self):
        obs = self.eval_env.reset()
        finals = []
        succ = []
        episodes = 0
        while episodes < self.n_eval_episodes:
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, rewards, dones, infos = self.eval_env.step(actions)
            _ = rewards
            for i, done in enumerate(dones):
                if not bool(done):
                    continue
                info = infos[i]
                finals.append(float(info.get("distance_error", np.nan)))
                succ.append(float(bool(info.get("success", False))))
                episodes += 1
                if episodes >= self.n_eval_episodes:
                    break
        mean_final_dist = float(np.nanmean(finals)) if finals else float("inf")
        success_rate = float(np.mean(succ)) if succ else 0.0
        return mean_final_dist, success_rate

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if (n - self.last_eval_timestep) < self.eval_freq:
            return True
        self.last_eval_timestep = n
        mean_final_dist, success_rate = self._evaluate_once()
        score = (5.0 * success_rate) - mean_final_dist
        self.logger.record("eval/mean_final_distance", float(mean_final_dist))
        self.logger.record("eval/success_rate", float(success_rate))
        self.logger.record("eval/score", float(score))
        if score > self.best_score + 1e-12:
            self.best_score = score
            try:
                self.model.save(str(self.best_model_path))
            except Exception:
                pass
            print(
                f"[best] step={n:,} score={score:.4f} "
                f"success_rate={success_rate:.3f} mean_final_dist={mean_final_dist:.4f}"
            )
        else:
            print(
                f"[eval] step={n:,} score={score:.4f} "
                f"success_rate={success_rate:.3f} mean_final_dist={mean_final_dist:.4f} "
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
    p = argparse.ArgumentParser(description="Train PPO on simple goal-conditioned 3D reach")

    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--resume-model", type=str, default=None, help="Optional PPO .zip checkpoint to resume from")
    p.add_argument(
        "--resume-reset-timesteps",
        action="store_true",
        default=False,
        help="When resuming, reset global timesteps/TensorBoard curve instead of continuing",
    )
    p.add_argument("--tb-log-dir", type=str, default=None, help="TensorBoard root log dir (default: <run_dir>/tensorboard)")
    p.add_argument("--tb-run-name", type=str, default="", help="TensorBoard run name (default: run-dir folder name)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--vec-env", type=str, default="auto", choices=("auto", "dummy", "subproc"))
    p.add_argument("--log-every-steps", type=int, default=2_000)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-anneal", dest="lr_anneal", action="store_true", default=True)
    p.add_argument("--no-lr-anneal", dest="lr_anneal", action="store_false")
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.00)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=10)

    p.add_argument("--env-id", type=str, default="myoArmReachRandom-v0")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--joint-profile", type=str, default="opensim_arm_wrist", choices=("opensim_arm", "opensim_arm_wrist"))
    p.add_argument("--joint-names", type=str, default="")
    p.add_argument("--action-mode", type=str, default="delta", choices=("delta", "absolute"))
    p.add_argument("--action-max-delta-deg", type=float, default=5.0)
    p.add_argument("--target-rollout-samples", type=int, default=128)
    p.add_argument("--target-min-start-dist", type=float, default=0.08)
    p.add_argument("--target-max-start-dist", type=float, default=0.55)
    p.add_argument("--randomize-start-pose", dest="randomize_start_pose", action="store_true", default=True)
    p.add_argument("--fixed-start-pose", dest="randomize_start_pose", action="store_false")
    p.add_argument("--start-pose-trials", type=int, default=64)
    p.add_argument("--project-external-targets", dest="project_external_targets", action="store_true", default=True)
    p.add_argument("--no-project-external-targets", dest="project_external_targets", action="store_false")
    p.add_argument("--success-radius", type=float, default=0.03)
    p.add_argument("--success-steps-required", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--reward-mode", type=str, default="distance", choices=("distance", "progress"))
    p.add_argument("--action-penalty", type=float, default=1e-3)
    p.add_argument("--palm-weight", type=float, default=0.20)
    p.add_argument("--palm-near-target-scale", type=float, default=0.12)
    p.add_argument("--success-bonus", type=float, default=2.0)
    p.add_argument("--terminate-on-success", dest="terminate_on_success", action="store_true", default=True)
    p.add_argument("--no-terminate-on-success", dest="terminate_on_success", action="store_false")
    return p


def _parse_joint_names(text: str):
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return tuple(vals) if vals else None


def make_env(rank: int, args) -> Callable[[], SimpleJointReach3DEnv]:
    def _init():
        seed = int(args.seed) + int(rank)
        env_kwargs = {}
        if args.model_path:
            env_kwargs["model_path"] = str(args.model_path)
            env_kwargs["obsd_model_path"] = str(args.model_path)
        env = SimpleJointReach3DEnv(
            env_id=args.env_id,
            env_make_kwargs=env_kwargs,
            joint_profile=args.joint_profile,
            joint_names=_parse_joint_names(args.joint_names),
            action_mode=args.action_mode,
            action_max_delta_deg=float(args.action_max_delta_deg),
            target_rollout_samples=int(args.target_rollout_samples),
            target_min_start_dist=float(args.target_min_start_dist),
            target_max_start_dist=float(args.target_max_start_dist),
            randomize_start_pose=bool(args.randomize_start_pose),
            start_pose_trials=int(args.start_pose_trials),
            project_external_targets=bool(args.project_external_targets),
            success_radius=float(args.success_radius),
            success_steps_required=int(args.success_steps_required),
            max_steps=int(args.max_steps),
            reward_mode=args.reward_mode,
            action_penalty=float(args.action_penalty),
            palm_weight=float(args.palm_weight),
            palm_near_target_scale=float(args.palm_near_target_scale),
            success_bonus=float(args.success_bonus),
            terminate_on_success=bool(args.terminate_on_success),
            seed=seed,
        )
        return env

    return _init


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(int(args.seed))

    this_dir = Path(__file__).resolve().parent
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else (
        this_dir / "runs" / f"ppo_simple_reach_{time.strftime('%Y%m%d_%H%M%S')}"
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

    eval_vec = DummyVecEnv([make_env(10_000, args)])
    eval_vec = VecMonitor(eval_vec)

    n_steps = int(max(64, args.n_steps))
    batch_size = int(max(64, args.batch_size))
    rollout_size = n_steps * n_envs
    if batch_size > rollout_size:
        batch_size = rollout_size
        print(f"[adjust] batch_size clipped to rollout_size={rollout_size}")

    lr_value = linear_schedule(float(args.lr)) if bool(args.lr_anneal) else float(args.lr)
    if args.resume_model:
        resume_path = Path(args.resume_model).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume model not found: {resume_path}")
        print(f"[resume] model={resume_path}")
        model = PPO.load(
            str(resume_path),
            env=vec,
            device=args.device,
            custom_objects={"learning_rate": lr_value},
            tensorboard_log=(str(tb_log_dir) if tb_enabled else None),
        )
        print(f"[resume] loaded_num_timesteps={int(getattr(model, 'num_timesteps', 0)):,}")
    else:
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=lr_value,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(args.n_epochs),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            seed=int(args.seed),
            device=args.device,
            verbose=1,
            tensorboard_log=(str(tb_log_dir) if tb_enabled else None),
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    eval_freq = int(max(1, int(args.eval_freq) // max(1, n_envs)))
    callback = CallbackList(
        [
            SpeedPrintCallback(every_steps=int(args.log_every_steps), verbose=1),
            BestModelByDistanceCallback(
                eval_env=eval_vec,
                run_dir=run_dir,
                eval_freq=eval_freq,
                n_eval_episodes=int(args.eval_episodes),
                deterministic=True,
                verbose=1,
            ),
        ]
    )
    reset_num_timesteps = True
    total_timesteps = int(args.total_steps)
    if args.resume_model and not bool(args.resume_reset_timesteps):
        reset_num_timesteps = False
        loaded_steps = int(getattr(model, "num_timesteps", 0))
        total_timesteps = max(0, int(args.total_steps) - loaded_steps)
        print(
            f"[resume] continue_timesteps=True loaded={loaded_steps:,} "
            f"target_total={int(args.total_steps):,} remaining={total_timesteps:,}"
        )

    learn_kwargs = dict(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )
    if tb_enabled:
        learn_kwargs["tb_log_name"] = tb_run_name
    if total_timesteps > 0:
        model.learn(**learn_kwargs)
    else:
        print("[train] remaining steps is 0; skip learn()")

    model_path = run_dir / "model_final.zip"
    model.save(str(model_path))
    print(f"[saved] model={model_path}")
    best = run_dir / "best_model.zip"
    if best.exists():
        print(f"[saved] best_model={best}")

    vec.close()
    eval_vec.close()


if __name__ == "__main__":
    main()
