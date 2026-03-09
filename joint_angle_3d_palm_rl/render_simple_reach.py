"""Render / drag-target demo for a trained simple 3D reaching policy."""

from __future__ import annotations

import argparse
import math
import re
import time
from typing import Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for render_simple_reach.py") from exc

from joint_angle_3d_palm_rl.simple_reach_env import SimpleJointReach3DEnv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render trained simple reaching policy")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
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
    p.add_argument("--project-targets-to-reachable", dest="project_targets_to_reachable", action="store_true", default=True)
    p.add_argument("--no-project-targets-to-reachable", dest="project_targets_to_reachable", action="store_false")
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
    p.add_argument("--hide-background", dest="hide_background", action="store_true", default=False)
    p.add_argument("--show-background", dest="hide_background", action="store_false")
    p.add_argument(
        "--hold-target-across-reset",
        dest="hold_target_across_reset",
        action="store_true",
        default=False,
        help="Keep the same manual target after episode reset (non-eval behavior).",
    )
    p.add_argument("--no-hold-target-across-reset", dest="hold_target_across_reset", action="store_false")
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--interactive-render", dest="interactive_render", action="store_true", default=True)
    p.add_argument("--no-interactive-render", dest="interactive_render", action="store_false")
    p.add_argument("--no-embedded-render", action="store_true")
    p.add_argument("--render-width", type=int, default=960)
    p.add_argument("--render-height", type=int, default=720)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Run eval-aligned episodes at start (0 disables).",
    )
    return p


def _parse_joint_names(text: str):
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return tuple(vals) if vals else None


def _build_env(args) -> SimpleJointReach3DEnv:
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
        project_external_targets=bool(args.project_targets_to_reachable),
        success_radius=float(args.success_radius),
        success_steps_required=int(args.success_steps_required),
        max_steps=int(args.max_steps),
        reward_mode=args.reward_mode,
        action_penalty=float(args.action_penalty),
        palm_weight=float(args.palm_weight),
        palm_near_target_scale=float(args.palm_near_target_scale),
        success_bonus=float(args.success_bonus),
        terminate_on_success=bool(args.terminate_on_success),
        hide_background=bool(args.hide_background),
        seed=int(args.seed),
    )
    return env


def _safe_render_frame(env: SimpleJointReach3DEnv, req_w: int, req_h: int) -> Tuple[np.ndarray, int, int]:
    try:
        frame = env.render_rgb_array(width=int(req_w), height=int(req_h))
        return frame, int(req_w), int(req_h)
    except Exception as exc:
        msg = str(exc)
        fbw_m = re.search(r"framebuffer width\s+(\d+)", msg)
        fbh_m = re.search(r"framebuffer height\s+(\d+)", msg)
        fbw = int(fbw_m.group(1)) if fbw_m else 640
        fbh = int(fbh_m.group(1)) if fbh_m else 480
        nw = int(min(max(1, int(req_w)), max(1, fbw)))
        nh = int(min(max(1, int(req_h)), max(1, fbh)))
        frame = env.render_rgb_array(width=nw, height=nh)
        return frame, nw, nh


def _init_embedded_camera_state(env: SimpleJointReach3DEnv) -> dict:
    tip = env.get_tip_position()
    tgt = env.get_target_position()
    look = 0.5 * (np.asarray(tip, dtype=np.float64).reshape(3) + np.asarray(tgt, dtype=np.float64).reshape(3))
    if not np.isfinite(look).all():
        look = np.asarray(tip, dtype=np.float64).reshape(3)
    return {
        "azimuth": float(getattr(env, "render_camera_azimuth", 145.0)),
        "elevation": float(getattr(env, "render_camera_elevation", -22.0)),
        "distance": float(getattr(env, "render_camera_distance", 1.10)),
        "lookat": look.astype(np.float64).copy(),
        "default_azimuth": float(getattr(env, "render_camera_azimuth", 145.0)),
        "default_elevation": float(getattr(env, "render_camera_elevation", -22.0)),
        "default_distance": float(getattr(env, "render_camera_distance", 1.10)),
        "default_lookat": look.astype(np.float64).copy(),
        "drag_mode": None,
        "last_xy": None,
    }


def _apply_embedded_camera(env: SimpleJointReach3DEnv, cam: dict) -> None:
    look = np.asarray(cam.get("lookat", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
    az = float(cam.get("azimuth", 145.0))
    el = float(np.clip(float(cam.get("elevation", -22.0)), -89.0, 89.0))
    dist = float(np.clip(float(cam.get("distance", 1.10)), 0.20, 6.0))

    try:
        env.render_camera_azimuth = az
        env.render_camera_elevation = el
        env.render_camera_distance = dist
        env._camera_lookat = look.copy()  # pylint: disable=protected-access
    except Exception:
        pass

    try:
        uw = env.unwrapped
        sim = uw.sim
        renderer = sim.renderer
        if hasattr(renderer, "set_free_camera_settings"):
            renderer.set_free_camera_settings(
                distance=dist - 2.0,
                azimuth=az,
                elevation=el,
                lookat=look,
                center=False,
            )
        if hasattr(uw, "viewer_setup"):
            uw.viewer_setup(
                distance=dist - 2.0,
                azimuth=az,
                elevation=el,
                lookat=look,
            )
    except Exception:
        pass


def _update_interactive_view(env: SimpleJointReach3DEnv, state: dict) -> None:
    if not bool(state.get("interactive_view_on", True)):
        return
    try:
        env.render()
        state["interactive_view_ok"] = True
    except Exception as exc:
        state["interactive_view_ok"] = False
        state["interactive_view_error"] = str(exc)
        state["interactive_view_on"] = False


def _sync_z_slider_value(slider: Optional[Slider], z_val: float, pad: float = 0.35) -> Optional[float]:
    if slider is None:
        return None
    z = float(z_val)
    lo = float(slider.valmin)
    hi = float(slider.valmax)
    if not (lo <= z <= hi):
        new_lo = float(z - pad)
        new_hi = float(z + pad)
        slider.valmin = new_lo
        slider.valmax = new_hi
        slider.ax.set_ylim(new_lo, new_hi)
    slider.set_val(z)
    return float(z)


def _run_eval_aligned_check(model: PPO, vec, n_episodes: int) -> None:
    n_eps = int(max(0, n_episodes))
    if n_eps <= 0:
        return
    obs = vec.reset()
    finals = []
    succ = []
    episodes = 0
    while episodes < n_eps:
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, dones, infos = vec.step(act)
        _ = rew
        if bool(dones[0]):
            info = infos[0]
            finals.append(float(info.get("distance_error", np.nan)))
            succ.append(float(bool(info.get("success", False))))
            episodes += 1
    mean_final_dist = float(np.nanmean(np.asarray(finals, dtype=np.float64))) if finals else float("nan")
    success_rate = float(np.mean(np.asarray(succ, dtype=np.float64))) if succ else float("nan")
    score = float(5.0 * success_rate - mean_final_dist) if np.isfinite(success_rate) and np.isfinite(mean_final_dist) else float("nan")
    print(
        "[eval-aligned-startup] "
        f"episodes={n_eps} "
        f"success_rate={success_rate:.3f} "
        f"mean_final_dist={mean_final_dist:.4f} "
        f"score={score:.4f}"
    )


def run_demo(args: argparse.Namespace) -> None:
    env = _build_env(args)
    _obs0, _ = env.reset(seed=int(args.seed))
    vec = DummyVecEnv([lambda: env])
    model = PPO.load(str(args.model), env=vec)
    obs = vec.reset()
    _run_eval_aligned_check(model, vec, n_episodes=int(args.eval_episodes))
    obs = vec.reset()

    tip = env.get_tip_position()
    target = env.get_target_position()
    target_xy = target[:2].copy()
    target_z = float(target[2])
    last_set = target.copy()
    manual_target_active = False
    slider_sync_value: Optional[float] = None

    ui_state = {
        "interactive_view_on": bool(args.interactive_render),
        "interactive_view_ok": True,
        "interactive_view_error": "",
        "interactive_view_warned": False,
    }
    render_cam = None
    dragging = {"on": False}

    if not args.headless:
        fig = plt.figure(figsize=(13.0, 7.0))
        ax = fig.add_axes([0.05, 0.16, 0.42, 0.78])
        axz = fig.add_axes([0.95, 0.18, 0.02, 0.72])
        if not args.no_embedded_render:
            ax3d = fig.add_axes([0.52, 0.18, 0.40, 0.72])
            render_cam = _init_embedded_camera_state(env)
            _apply_embedded_camera(env, render_cam)
            frame0, rw, rh = _safe_render_frame(env, int(args.render_width), int(args.render_height))
            rgb_img = ax3d.imshow(frame0)
            ax3d.set_title(f"3D Render ({rw}x{rh}) | L-rotate R-pan M-lift Wheel-zoom R-reset")
            ax3d.axis("off")
        else:
            ax3d = None
            rgb_img = None

        x_lo, x_hi, y_lo, y_hi = env.estimate_target_bounds(n_samples=256, margin=0.1)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Simple Reach: drag target in XY, adjust Z slider, press Q to quit")
        ax.set_xlabel("world X (m)")
        ax.set_ylabel("world Y (m)")
        ax.grid(True, alpha=0.3)
        tip_pt = ax.scatter([tip[0]], [tip[1]], c="tab:red", s=80, label="hand tip")
        tgt_pt = ax.scatter([target_xy[0]], [target_xy[1]], c="tab:green", s=80, label="target")
        ax.legend(loc="upper right")
        status = ax.text(
            0.01,
            0.01,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
        )

        z_slider = Slider(axz, "Z", float(target_z - 0.35), float(target_z + 0.35), valinit=float(target_z), orientation="vertical")

        def _set_drag_target(evt):
            nonlocal target_xy
            if evt.inaxes is not ax or evt.xdata is None or evt.ydata is None:
                return
            target_xy = np.array([float(evt.xdata), float(evt.ydata)], dtype=np.float64)
            nonlocal manual_target_active
            manual_target_active = True

        def _on_press(evt):
            if evt.button != 1 or evt.inaxes is not ax:
                return
            dragging["on"] = True
            _set_drag_target(evt)

        def _on_release(evt):
            _ = evt
            dragging["on"] = False

        def _on_move(evt):
            if dragging["on"]:
                _set_drag_target(evt)

        def _on_key(evt):
            k = str(evt.key).lower()
            if k == "q":
                plt.close(fig)
            elif k == "r" and render_cam is not None:
                render_cam["azimuth"] = float(render_cam["default_azimuth"])
                render_cam["elevation"] = float(render_cam["default_elevation"])
                render_cam["distance"] = float(render_cam["default_distance"])
                render_cam["lookat"] = np.asarray(render_cam["default_lookat"], dtype=np.float64).reshape(3).copy()
                _apply_embedded_camera(env, render_cam)

        def _on_render_press(evt):
            if ax3d is None or evt.inaxes is not ax3d:
                return
            if evt.button == 1:
                render_cam["drag_mode"] = "rotate"
            elif evt.button == 3:
                render_cam["drag_mode"] = "pan"
            elif evt.button == 2:
                render_cam["drag_mode"] = "lift"
            else:
                render_cam["drag_mode"] = None
            if render_cam["drag_mode"] is not None and evt.x is not None and evt.y is not None:
                render_cam["last_xy"] = (float(evt.x), float(evt.y))

        def _on_render_release(evt):
            _ = evt
            if render_cam is not None:
                render_cam["drag_mode"] = None
                render_cam["last_xy"] = None

        def _on_render_move(evt):
            if render_cam is None:
                return
            mode = render_cam.get("drag_mode", None)
            last = render_cam.get("last_xy", None)
            if mode is None or last is None or ax3d is None or evt.inaxes is not ax3d or evt.x is None or evt.y is None:
                return
            x0, y0 = [float(v) for v in last]
            x1, y1 = float(evt.x), float(evt.y)
            dx = x1 - x0
            dy = y1 - y0
            render_cam["last_xy"] = (x1, y1)

            if mode == "rotate":
                render_cam["azimuth"] = float(render_cam["azimuth"]) - 0.28 * dx
                render_cam["elevation"] = float(np.clip(float(render_cam["elevation"]) - 0.22 * dy, -89.0, 89.0))
            else:
                az = math.radians(float(render_cam["azimuth"]))
                fwd = np.array([math.cos(az), math.sin(az)], dtype=np.float64)
                right = np.array([fwd[1], -fwd[0]], dtype=np.float64)
                scale = 0.0022 * float(render_cam["distance"])
                look = np.asarray(render_cam["lookat"], dtype=np.float64).copy()
                if mode == "pan":
                    d2 = scale * ((-dx) * right + (dy) * fwd)
                    look[0] += float(d2[0])
                    look[1] += float(d2[1])
                elif mode == "lift":
                    look[2] += float(-dy * scale)
                render_cam["lookat"] = look
            _apply_embedded_camera(env, render_cam)

        def _on_render_scroll(evt):
            if render_cam is None or ax3d is None or evt.inaxes is not ax3d:
                return
            step_sign = 1.0 if str(evt.button).lower() == "up" else -1.0
            d = float(render_cam["distance"])
            render_cam["distance"] = float(np.clip(d * (0.92 ** step_sign), 0.20, 6.0))
            _apply_embedded_camera(env, render_cam)

        fig.canvas.mpl_connect("button_press_event", _on_press)
        fig.canvas.mpl_connect("button_release_event", _on_release)
        fig.canvas.mpl_connect("motion_notify_event", _on_move)
        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.canvas.mpl_connect("button_press_event", _on_render_press)
        fig.canvas.mpl_connect("button_release_event", _on_render_release)
        fig.canvas.mpl_connect("motion_notify_event", _on_render_move)
        fig.canvas.mpl_connect("scroll_event", _on_render_scroll)
    else:
        fig = None
        ax3d = None
        rgb_img = None
        tip_pt = None
        tgt_pt = None
        status = None
        z_slider = None

    t0 = time.time()
    last_print = t0
    dt = 1.0 / float(max(1e-3, args.control_hz))
    step_idx = 0
    ep_idx = 0
    ep_final_dists = []
    ep_success = []

    try:
        while True:
            loop_t0 = time.time()
            if args.max_seconds > 0 and (loop_t0 - t0) >= float(args.max_seconds):
                break
            if fig is not None and not plt.fignum_exists(fig.number):
                break

            if z_slider is not None:
                z_new = float(z_slider.val)
                if (slider_sync_value is not None) and (abs(z_new - float(slider_sync_value)) <= 1e-6):
                    slider_sync_value = None
                elif abs(z_new - target_z) > 1e-6:
                    manual_target_active = True
                target_z = z_new
            if manual_target_active:
                cmd = np.array([float(target_xy[0]), float(target_xy[1]), float(target_z)], dtype=np.float64)
                if float(np.linalg.norm(cmd - last_set)) > 1e-6:
                    env.set_target(cmd, project_to_reachable=bool(args.project_targets_to_reachable))
                    last_set = env.get_target_position().copy()

            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, reward, dones, infos = vec.step(action)
            _ = reward
            info = infos[0]
            tip = env.get_tip_position()
            target_now = env.get_target_position()
            err = float(np.linalg.norm(target_now - tip))
            success = bool(info.get("success", False))

            _update_interactive_view(env, ui_state)
            if (
                (not bool(ui_state.get("interactive_view_on", True)))
                and (not bool(ui_state.get("interactive_view_warned", False)))
                and str(ui_state.get("interactive_view_error", "")).strip()
            ):
                print(f"[warn] interactive 3D viewer disabled: {ui_state.get('interactive_view_error')}")
                ui_state["interactive_view_warned"] = True

            if fig is not None and rgb_img is not None and (step_idx % 1 == 0):
                try:
                    if render_cam is not None:
                        _apply_embedded_camera(env, render_cam)
                    frame, _, _ = _safe_render_frame(env, int(args.render_width), int(args.render_height))
                    rgb_img.set_data(frame)
                except Exception:
                    rgb_img = None

            if fig is not None:
                tip_pt.set_offsets(np.array([[tip[0], tip[1]]], dtype=np.float64))
                tgt_pt.set_offsets(np.array([[target_now[0], target_now[1]]], dtype=np.float64))
                status.set_text(
                    f"err={err:.4f} m | success={int(success)}\n"
                    f"target=({target_now[0]:+.3f},{target_now[1]:+.3f},{target_now[2]:+.3f}) "
                    f"tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})"
                )
                fig.canvas.draw_idle()
                plt.pause(0.001)
            elif (loop_t0 - last_print) >= 1.0:
                last_print = loop_t0
                print(f"[demo] err={err:.4f} success={int(success)} target={np.round(target_now,3)} tip={np.round(tip,3)}")

            if bool(dones[0]):
                ep_idx += 1
                ep_final_dists.append(float(info.get("distance_error", err)))
                ep_success.append(float(bool(info.get("success", False))))
                if (ep_idx % 5) == 0:
                    print(
                        "[eval-like] "
                        f"episodes={ep_idx} "
                        f"success_rate={float(np.mean(ep_success)):.3f} "
                        f"mean_final_dist={float(np.mean(ep_final_dists)):.4f}"
                    )
                obs = vec.reset()
                if bool(args.hold_target_across_reset):
                    env.set_target(last_set, project_to_reachable=bool(args.project_targets_to_reachable))
                    last_set = env.get_target_position().copy()
                else:
                    last_set = env.get_target_position().copy()
                    target_xy = last_set[:2].copy()
                    target_z = float(last_set[2])
                    slider_sync_value = _sync_z_slider_value(z_slider, target_z, pad=0.35)
                    manual_target_active = False

            step_idx += 1
            sleep_t = dt - (time.time() - loop_t0)
            if sleep_t > 0.0:
                time.sleep(float(sleep_t))
    finally:
        if ep_idx > 0:
            print(
                "[eval-like-summary] "
                f"episodes={ep_idx} "
                f"success_rate={float(np.mean(ep_success)):.3f} "
                f"mean_final_dist={float(np.mean(ep_final_dists)):.4f}"
            )
        vec.close()
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


def main():
    args = build_parser().parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
