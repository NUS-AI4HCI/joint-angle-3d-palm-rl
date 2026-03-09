"""Interactive drag-target demo for a trained 3D joint-angle policy.

Controls:
- Drag target in XY plot with left mouse.
- Adjust target Z using slider.
- Press Q to quit.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for drag_policy_demo.py") from exc

# Allow running this folder as a standalone subproject from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from joint_angle_3d_palm_rl.env import JointAnglePalmDownReach3DEnv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drag-target interactive demo for 3D joint-angle palm-down policy")
    p.add_argument("--model", type=str, required=True, help="Path to PPO model zip")
    p.add_argument("--vecnormalize", type=str, default=None, help="Optional VecNormalize pickle from training")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--env-id", type=str, default="myoArmReachRandom-v0")
    p.add_argument("--model-path", type=str, default=None, help="Optional custom MyoSuite XML")
    p.add_argument(
        "--joint-profile",
        type=str,
        default="opensim_arm_wrist",
        choices=("opensim_arm", "opensim_arm_wrist", "opensim_full_shoulder_wrist"),
    )
    p.add_argument("--joint-names", type=str, default="", help="comma-separated override list")
    p.add_argument("--action-max-delta-deg", type=float, default=5.0)
    p.add_argument("--enable-root-yaw", dest="enable_root_yaw", action="store_true", default=True)
    p.add_argument("--disable-root-yaw", dest="enable_root_yaw", action="store_false")
    p.add_argument("--root-body-name", type=str, default="full_body")
    p.add_argument("--root-yaw-limit-deg", type=float, default=75.0)
    p.add_argument("--root-yaw-step-deg", type=float, default=3.0)
    p.add_argument("--root-yaw-action-mode", type=str, default="absolute", choices=("delta", "absolute"))
    p.add_argument("--root-yaw-smooth-alpha", type=float, default=0.25)
    p.add_argument("--retarget-interval-steps", type=int, default=0, help="set 0 to disable random in-episode retargeting")
    p.add_argument("--project-targets-to-reachable", dest="project_targets_to_reachable", action="store_true", default=False)
    p.add_argument("--no-project-targets-to-reachable", dest="project_targets_to_reachable", action="store_false")
    p.add_argument("--terminate-on-success", dest="terminate_on_success", action="store_true", default=False)
    p.add_argument("--no-terminate-on-success", dest="terminate_on_success", action="store_false")
    p.add_argument("--auto-retarget-on-success", dest="auto_retarget_on_success", action="store_true", default=False)
    p.add_argument("--no-auto-retarget-on-success", dest="auto_retarget_on_success", action="store_false")
    p.add_argument("--success-radius", type=float, default=0.030)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--interactive-render", dest="interactive_render", action="store_true", default=True)
    p.add_argument("--no-interactive-render", dest="interactive_render", action="store_false")
    p.add_argument("--no-render", action="store_true", help="Alias for --no-interactive-render")
    p.add_argument("--no-embedded-render", action="store_true", help="Disable embedded 3D panel in control UI")
    p.add_argument("--render-width", type=int, default=960)
    p.add_argument("--render-height", type=int, default=720)
    p.add_argument("--render-every", type=int, default=1, help="update embedded 3D panel every N control steps")
    p.add_argument("--workspace-samples", type=int, default=256, help="samples for estimating XY workspace")
    p.add_argument("--workspace-margin", type=float, default=0.10, help="margin added to estimated XY workspace")
    p.add_argument("--min-xy-width", type=float, default=1.60, help="minimum width/height shown in XY panel")
    p.add_argument("--x-min", type=float, default=None, help="optional fixed x-axis min")
    p.add_argument("--x-max", type=float, default=None, help="optional fixed x-axis max")
    p.add_argument("--y-min", type=float, default=None, help="optional fixed y-axis min")
    p.add_argument("--y-max", type=float, default=None, help="optional fixed y-axis max")
    p.add_argument("--auto-expand-axes", dest="auto_expand_axes", action="store_true", default=True)
    p.add_argument("--no-auto-expand-axes", dest="auto_expand_axes", action="store_false")
    p.add_argument("--headless", action="store_true", help="Run without matplotlib interaction loop")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Limit run time (0 = unlimited)")
    return p


def _parse_joint_names(csv_text: str):
    vals = [x.strip() for x in str(csv_text).split(",") if x.strip()]
    return tuple(vals) if vals else None


def _build_env(args) -> JointAnglePalmDownReach3DEnv:
    env_kwargs = {}
    if args.model_path:
        env_kwargs["model_path"] = str(args.model_path)
        env_kwargs["obsd_model_path"] = str(args.model_path)

    env = JointAnglePalmDownReach3DEnv(
        env_id=args.env_id,
        env_make_kwargs=env_kwargs,
        joint_profile=args.joint_profile,
        joint_names=_parse_joint_names(args.joint_names),
        action_max_delta_deg=float(args.action_max_delta_deg),
        enable_root_yaw=bool(args.enable_root_yaw),
        root_body_name=str(args.root_body_name),
        root_yaw_limit_deg=float(args.root_yaw_limit_deg),
        root_yaw_step_deg=float(args.root_yaw_step_deg),
        root_yaw_action_mode=str(args.root_yaw_action_mode),
        root_yaw_smooth_alpha=float(args.root_yaw_smooth_alpha),
        project_targets_to_reachable=bool(args.project_targets_to_reachable),
        retarget_interval_steps=int(args.retarget_interval_steps),
        success_radius=float(args.success_radius),
        max_steps=int(args.max_steps),
        terminate_on_success=bool(args.terminate_on_success),
        auto_retarget_on_success=bool(args.auto_retarget_on_success),
        seed=int(args.seed),
    )
    return env


def _load_model(args, vec_env):
    model = PPO.load(str(args.model), env=vec_env)
    return model


def _set_target_from_xy_z(
    env: JointAnglePalmDownReach3DEnv,
    xy: np.ndarray,
    z: float,
    *,
    project_to_reachable: bool,
    last_cmd: Optional[np.ndarray] = None,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    t = env.get_target_position()
    tgt = np.array([float(xy[0]), float(xy[1]), float(z if np.isfinite(z) else t[2])], dtype=np.float64)
    if last_cmd is not None:
        last_cmd = np.asarray(last_cmd, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(tgt - last_cmd)) <= float(max(0.0, tol)):
            return last_cmd.copy(), False
    env.set_target(
        tgt,
        project_to_reachable=bool(project_to_reachable),
        line_from_tip=True,
    )
    return env.get_target_position().copy(), True


def _safe_render_frame(env: JointAnglePalmDownReach3DEnv, req_w: int, req_h: int) -> Tuple[np.ndarray, int, int]:
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


def _init_embedded_camera_state(env: JointAnglePalmDownReach3DEnv) -> dict:
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


def _apply_embedded_camera(env: JointAnglePalmDownReach3DEnv, cam: dict) -> None:
    look = np.asarray(cam.get("lookat", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
    az = float(cam.get("azimuth", 145.0))
    el = float(np.clip(float(cam.get("elevation", -22.0)), -89.0, 89.0))
    dist = float(np.clip(float(cam.get("distance", 1.10)), 0.20, 6.0))

    # Keep env-side camera attributes in sync.
    try:
        env.render_camera_azimuth = az
        env.render_camera_elevation = el
        env.render_camera_distance = dist
        env._camera_lookat = look.copy()  # pylint: disable=protected-access
    except Exception:
        pass

    # Apply camera to offscreen renderer + interactive window (if available).
    try:
        uw = env.unwrapped
        sim = uw.sim
        renderer = sim.renderer
        if hasattr(renderer, "set_free_camera_settings"):
            renderer.set_free_camera_settings(
                distance=dist - 2.0,  # MyoSuite renderer internally adds +2
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


def _update_interactive_view(env: JointAnglePalmDownReach3DEnv, state: dict) -> None:
    if not bool(state.get("interactive_view_on", True)):
        return
    try:
        env.render()
        state["interactive_view_ok"] = True
        state["interactive_view_error"] = ""
    except Exception as exc:
        state["interactive_view_ok"] = False
        state["interactive_view_error"] = str(exc)
        state["interactive_view_on"] = False


def _compute_initial_xy_bounds(env: JointAnglePalmDownReach3DEnv, args) -> Tuple[float, float, float, float]:
    try:
        x_lo, x_hi, y_lo, y_hi = env.estimate_target_bounds(
            n_samples=int(max(16, args.workspace_samples)),
            margin=float(max(0.0, args.workspace_margin)),
        )
    except Exception:
        tip = env.get_tip_position()
        span = float(max(0.6, args.min_xy_width))
        x_lo, x_hi = float(tip[0] - 0.5 * span), float(tip[0] + 0.5 * span)
        y_lo, y_hi = float(tip[1] - 0.5 * span), float(tip[1] + 0.5 * span)

    min_w = float(max(0.4, args.min_xy_width))
    if (x_hi - x_lo) < min_w:
        cx = 0.5 * (x_lo + x_hi)
        x_lo, x_hi = float(cx - 0.5 * min_w), float(cx + 0.5 * min_w)
    if (y_hi - y_lo) < min_w:
        cy = 0.5 * (y_lo + y_hi)
        y_lo, y_hi = float(cy - 0.5 * min_w), float(cy + 0.5 * min_w)

    # Keep a reasonably large positive-x zone visible by default.
    x_hi = max(float(x_hi), 0.85)
    x_lo = min(float(x_lo), -0.85)
    y_hi = max(float(y_hi), 0.85)
    y_lo = min(float(y_lo), -0.85)

    if args.x_min is not None:
        x_lo = float(args.x_min)
    if args.x_max is not None:
        x_hi = float(args.x_max)
    if args.y_min is not None:
        y_lo = float(args.y_min)
    if args.y_max is not None:
        y_hi = float(args.y_max)
    return float(x_lo), float(x_hi), float(y_lo), float(y_hi)


def _maybe_expand_axes(ax, pt_xy: np.ndarray, margin_ratio: float = 0.08) -> None:
    x = float(pt_xy[0])
    y = float(pt_xy[1])
    x0, x1 = [float(v) for v in ax.get_xlim()]
    y0, y1 = [float(v) for v in ax.get_ylim()]
    wx = max(1e-6, x1 - x0)
    wy = max(1e-6, y1 - y0)
    mx = margin_ratio * wx
    my = margin_ratio * wy
    changed = False
    if x < x0 + mx:
        x0 = x - 0.25 * wx
        changed = True
    if x > x1 - mx:
        x1 = x + 0.25 * wx
        changed = True
    if y < y0 + my:
        y0 = y - 0.25 * wy
        changed = True
    if y > y1 - my:
        y1 = y + 0.25 * wy
        changed = True
    if changed:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)


def run_demo(args: argparse.Namespace) -> None:
    if bool(getattr(args, "no_render", False)):
        args.interactive_render = False

    env = _build_env(args)
    base_obs, _ = env.reset(seed=int(args.seed))
    _ = base_obs
    vec = DummyVecEnv([lambda: env])
    if args.vecnormalize:
        vec = VecNormalize.load(str(args.vecnormalize), vec)
        vec.training = False
        vec.norm_reward = False

    model = _load_model(args, vec)
    obs = vec.reset()

    tip = env.get_tip_position()
    target = env.get_target_position()
    target_xy = target[:2].copy()
    target_z = float(target[2])
    dragging = {"on": False}
    ui_state = {
        "interactive_view_on": bool(args.interactive_render),
        "interactive_view_ok": True,
        "interactive_view_error": "",
        "interactive_view_warned": False,
    }
    render_cam = None

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

        ax.set_title("Drag target (XY) + Z slider. Q to quit.")
        ax.set_xlabel("world X (m)")
        ax.set_ylabel("world Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        x_lo, x_hi, y_lo, y_hi = _compute_initial_xy_bounds(env, args)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

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

        zmin = float(target_z - 0.35)
        zmax = float(target_z + 0.35)
        z_slider = Slider(axz, "Z", zmin, zmax, valinit=float(target_z), orientation="vertical")

        def _set_drag_target(evt) -> None:
            nonlocal target_xy
            if evt.inaxes is not ax or evt.xdata is None or evt.ydata is None:
                return
            target_xy = np.array([float(evt.xdata), float(evt.ydata)], dtype=np.float64)

        def _on_press(evt) -> None:
            if evt.button != 1 or evt.inaxes is not ax:
                return
            dragging["on"] = True
            _set_drag_target(evt)

        def _on_release(evt) -> None:
            _ = evt
            dragging["on"] = False

        def _on_move(evt) -> None:
            if not dragging["on"]:
                return
            _set_drag_target(evt)

        def _on_key(evt) -> None:
            k = str(evt.key).lower()
            if k == "q":
                plt.close(fig)
            elif k == "r" and render_cam is not None:
                render_cam["azimuth"] = float(render_cam["default_azimuth"])
                render_cam["elevation"] = float(render_cam["default_elevation"])
                render_cam["distance"] = float(render_cam["default_distance"])
                render_cam["lookat"] = np.asarray(render_cam["default_lookat"], dtype=np.float64).reshape(3).copy()
                _apply_embedded_camera(env, render_cam)

        def _on_render_press(evt) -> None:
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

        def _on_render_release(evt) -> None:
            _ = evt
            if render_cam is not None:
                render_cam["drag_mode"] = None
                render_cam["last_xy"] = None

        def _on_render_move(evt) -> None:
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

        def _on_render_scroll(evt) -> None:
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
        tip_pt = None
        tgt_pt = None
        status = None
        z_slider = None
        ax3d = None
        rgb_img = None

    t0 = time.time()
    dt = 1.0 / float(max(1e-3, args.control_hz))
    last_print = t0
    episodes = 0
    last_target = env.get_target_position().copy()
    last_set_target_cmd = last_target.copy()
    step_idx = 0

    try:
        while True:
            loop_t0 = time.time()
            if args.max_seconds > 0 and (loop_t0 - t0) >= float(args.max_seconds):
                break
            if fig is not None and not plt.fignum_exists(fig.number):
                break

            if z_slider is not None:
                target_z = float(z_slider.val)
            last_set_target_cmd, _ = _set_target_from_xy_z(
                env,
                target_xy,
                target_z,
                project_to_reachable=bool(args.project_targets_to_reachable),
                last_cmd=last_set_target_cmd,
                tol=1e-6,
            )
            last_target = env.get_target_position().copy()

            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, reward, dones, infos = vec.step(action)
            _ = reward
            info = infos[0]

            tip = env.get_tip_position()
            target_now = env.get_target_position()
            err = float(np.linalg.norm(target_now - tip))
            palm_err = float(info.get("palm_down_error", 0.0))
            flu_cost = float(info.get("smoothness_cost", 0.0))
            direct_err = float(info.get("directness_error", 0.0))
            yaw_deg = float(info.get("root_yaw_deg", 0.0))

            _update_interactive_view(env, ui_state)
            if (
                (not bool(ui_state.get("interactive_view_on", True)))
                and (not bool(ui_state.get("interactive_view_warned", False)))
                and str(ui_state.get("interactive_view_error", "")).strip()
            ):
                print(f"[warn] interactive 3D viewer disabled: {ui_state.get('interactive_view_error')}")
                ui_state["interactive_view_warned"] = True

            if fig is not None and rgb_img is not None and (step_idx % int(max(1, args.render_every)) == 0):
                try:
                    if render_cam is not None:
                        _apply_embedded_camera(env, render_cam)
                    frame, _, _ = _safe_render_frame(env, int(args.render_width), int(args.render_height))
                    rgb_img.set_data(frame)
                except Exception as exc:
                    if ax3d is not None:
                        ax3d.set_title(f"3D Model Render (error: {str(exc)[:50]})")
                    rgb_img = None

            if fig is not None:
                if bool(args.auto_expand_axes):
                    _maybe_expand_axes(ax, target_now[:2], margin_ratio=0.08)
                    _maybe_expand_axes(ax, tip[:2], margin_ratio=0.06)
                tip_pt.set_offsets(np.array([[tip[0], tip[1]]], dtype=np.float64))
                tgt_pt.set_offsets(np.array([[target_now[0], target_now[1]]], dtype=np.float64))
                status.set_text(
                    f"err={err:.4f} m | direct={direct_err:.4f} | palm_err={palm_err:.4f} | smooth={flu_cost:.5f}\n"
                    f"target=({target_now[0]:+.3f},{target_now[1]:+.3f},{target_now[2]:+.3f}) "
                    f"tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})\n"
                    f"yaw={yaw_deg:+.1f} deg | "
                    f"viewer={'ON' if ui_state.get('interactive_view_on', False) else 'OFF'} "
                    f"viewer_ok={'Y' if ui_state.get('interactive_view_ok', False) else 'N'}"
                )
                fig.canvas.draw_idle()
                plt.pause(0.001)
            elif (loop_t0 - last_print) >= 1.0:
                last_print = loop_t0
                print(
                    f"[demo] err={err:.4f} direct={direct_err:.4f} palm_err={palm_err:.4f} smooth={flu_cost:.5f} "
                    f"target={np.round(target_now, 3)} tip={np.round(tip, 3)}"
                )

            if bool(dones[0]):
                episodes += 1
                obs = vec.reset()
                env.set_target(
                    last_target,
                    project_to_reachable=bool(args.project_targets_to_reachable),
                    line_from_tip=True,
                )
                last_set_target_cmd = env.get_target_position().copy()
            step_idx += 1

            sleep_t = dt - (time.time() - loop_t0)
            if sleep_t > 0.0:
                time.sleep(float(sleep_t))
    finally:
        vec.close()
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


def main() -> None:
    args = build_parser().parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
