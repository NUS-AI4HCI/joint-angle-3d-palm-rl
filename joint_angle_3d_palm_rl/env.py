"""Joint-angle RL environment for robust 3D palm-down reaching in MyoSuite.

Key properties:
- Continuous control over arm joints + optional bounded root-yaw action.
- Random reachable 3D targets with robustness filters.
- Reward centered on hand-target error, with soft palm-down penalty.
- Line-following terms between previous and current targets.
- Safety guards for unstable/out-of-control states.
"""

from __future__ import annotations

import math
import sys
import warnings
import io
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Gymnasium prints repeated "Overriding environment ... already in registry" warnings
# when MyoSuite re-registers envs across processes. This is expected and safe.
warnings.filterwarnings(
    "ignore",
    message=r".*Overriding environment .* already in registry.*",
    category=UserWarning,
)

try:
    # MyoSuite prints registration banners on import in each subprocess.
    # Silence import-time stdout/stderr to keep training logs readable.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import myosuite  # noqa: F401  # side-effect env registration
except Exception:
    myosuite = None  # type: ignore

try:
    import mujoco
except Exception:
    mujoco = None  # type: ignore


# Allow running this folder as a standalone subproject from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_JOINT_PROFILES: Dict[str, Tuple[str, ...]] = {
    "opensim_arm": (
        "elv_angle",
        "shoulder_elv",
        "shoulder1_r2",
        "shoulder_rot",
        "elbow_flexion",
        "pro_sup",
    ),
    "opensim_arm_wrist": (
        "elv_angle",
        "shoulder_elv",
        "shoulder1_r2",
        "shoulder_rot",
        "elbow_flexion",
        "pro_sup",
        "deviation",
        "flexion",
    ),
    "opensim_full_shoulder_wrist": (
        "sternoclavicular_r2",
        "sternoclavicular_r3",
        "acromioclavicular_r2",
        "acromioclavicular_r3",
        "acromioclavicular_r1",
        "unrothum_r1",
        "unrothum_r3",
        "unrothum_r2",
        "elv_angle",
        "shoulder_elv",
        "shoulder1_r2",
        "shoulder_rot",
        "elbow_flexion",
        "pro_sup",
        "deviation",
        "flexion",
    ),
}

_PALM_PRESET = {
    "pro_sup": -0.8844,
    "deviation": 0.1376,
    "flexion": -0.7506,
    "cmc_abduction": 0.0,
    "cmc_flexion": 0.0,
}


@dataclass
class JointSpec:
    names: Tuple[str, ...]
    q_adrs: np.ndarray
    dof_adrs: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(qq))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return qq / n


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(4)
    b = np.asarray(b, dtype=np.float64).reshape(4)
    w1, x1, y1, z1 = [float(v) for v in a]
    w2, x2, y2, z2 = [float(v) for v in b]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _axis_quat(axis_index: int, angle_rad: float) -> np.ndarray:
    q = np.zeros(4, dtype=np.float64)
    q[0] = float(np.cos(0.5 * float(angle_rad)))
    q[1 + int(axis_index)] = float(np.sin(0.5 * float(angle_rad)))
    return q


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    pp = np.asarray(p, dtype=np.float64).reshape(3)
    aa = np.asarray(a, dtype=np.float64).reshape(3)
    bb = np.asarray(b, dtype=np.float64).reshape(3)
    ab = bb - aa
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(pp - aa))
    t = float(np.clip(np.dot(pp - aa, ab) / denom, 0.0, 1.0))
    proj = aa + t * ab
    return float(np.linalg.norm(pp - proj))


class JointAnglePalmDownReach3DEnv(gym.Env):
    """MyoSuite 3D target reach environment with robust joint-angle control."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        env_id: str = "myoArmReachRandom-v0",
        env_make_kwargs: Optional[Dict[str, object]] = None,
        joint_profile: str = "opensim_arm_wrist",
        joint_names: Optional[Sequence[str]] = None,
        action_max_delta_deg: float = 5.0,
        # Root yaw control
        enable_root_yaw: bool = True,
        root_body_name: str = "full_body",
        root_yaw_limit_deg: float = 70.0,
        root_yaw_step_deg: float = 3.5,
        root_yaw_action_mode: str = "absolute",
        root_yaw_smooth_alpha: float = 0.25,
        # Target setup
        target_rollout_samples: int = 128,
        target_jitter_std: float = 0.010,
        target_min_start_dist: float = 0.12,
        target_max_start_dist: float = 0.60,
        project_targets_to_reachable: bool = True,
        retarget_interval_steps: int = 80,
        success_radius: float = 0.030,
        max_steps: int = 220,
        # Reward
        distance_weight: float = 3.5,
        progress_weight: float = 0.8,
        line_deviation_weight: float = 0.65,
        yaw_line_weight: float = 0.20,
        palm_weight: float = 0.30,
        directness_weight: float = 0.40,
        stability_weight: float = 0.20,
        away_penalty_weight: float = 0.25,
        palm_near_target_scale: float = 0.18,
        action_cost_weight: float = 0.020,
        smoothness_weight: float = 0.050,
        joint_limit_weight: float = 0.060,
        time_penalty: float = 0.003,
        success_bonus: float = 3.0,
        reward_clip: float = 0.0,
        terminate_on_success: bool = False,
        auto_retarget_on_success: bool = True,
        # Soft palm constraint only (no hard lock by default)
        palm_lock_alpha: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if myosuite is None:
            raise RuntimeError("myosuite is required for JointAnglePalmDownReach3DEnv")

        self.env = gym.make(env_id, **dict(env_make_kwargs or {}))
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

        # Control
        self.action_max_delta_rad = float(np.deg2rad(max(1e-3, action_max_delta_deg)))
        self.enable_root_yaw = bool(enable_root_yaw)
        self.root_body_name = str(root_body_name)
        self.root_yaw_limit_rad = float(np.deg2rad(max(1e-3, root_yaw_limit_deg)))
        self.root_yaw_step_rad = float(np.deg2rad(max(1e-4, root_yaw_step_deg)))
        self.root_yaw_action_mode = str(root_yaw_action_mode).strip().lower()
        if self.root_yaw_action_mode not in {"delta", "absolute"}:
            self.root_yaw_action_mode = "absolute"
        self.root_yaw_smooth_alpha = float(np.clip(root_yaw_smooth_alpha, 0.0, 1.0))

        # Target / episode
        self.target_rollout_samples = int(max(8, target_rollout_samples))
        self.target_jitter_std = float(max(0.0, target_jitter_std))
        self.target_min_start_dist = float(max(0.0, target_min_start_dist))
        self.target_max_start_dist = float(max(self.target_min_start_dist + 1e-6, target_max_start_dist))
        self.project_targets_to_reachable = bool(project_targets_to_reachable)
        self.retarget_interval_steps = int(max(0, retarget_interval_steps))
        self.success_radius = float(max(1e-4, success_radius))
        self.max_steps = int(max(1, max_steps))

        # Reward
        self.distance_weight = float(distance_weight)
        self.progress_weight = float(progress_weight)
        self.line_deviation_weight = float(max(0.0, line_deviation_weight))
        self.yaw_line_weight = float(max(0.0, yaw_line_weight))
        self.palm_weight = float(max(0.0, palm_weight))
        self.directness_weight = float(max(0.0, directness_weight))
        self.stability_weight = float(max(0.0, stability_weight))
        self.away_penalty_weight = float(max(0.0, away_penalty_weight))
        self.palm_near_target_scale = float(max(1e-4, palm_near_target_scale))
        self.action_cost_weight = float(max(0.0, action_cost_weight))
        self.smoothness_weight = float(max(0.0, smoothness_weight))
        self.joint_limit_weight = float(max(0.0, joint_limit_weight))
        self.time_penalty = float(max(0.0, time_penalty))
        self.success_bonus = float(success_bonus)
        self.reward_clip = float(max(0.0, reward_clip))
        self.terminate_on_success = bool(terminate_on_success)
        self.auto_retarget_on_success = bool(auto_retarget_on_success)
        self.palm_lock_alpha = float(np.clip(palm_lock_alpha, 0.0, 1.0))

        # Lazy state
        self._joint_spec: Optional[JointSpec] = None
        self._joint_profile = str(joint_profile).strip().lower()
        self._joint_names_override = tuple(str(n) for n in joint_names) if joint_names else None

        self._target_site_id: Optional[int] = None
        self._tip_site_id: Optional[int] = None

        self._palm_ref_normal: Optional[np.ndarray] = None
        self._palm_ref_q: Dict[int, float] = {}

        self._root_body_id: Optional[int] = None
        self._root_base_quat: Optional[np.ndarray] = None
        self._root_yaw_rad: float = 0.0
        self._root_yaw_enabled_runtime: bool = False

        self._reachable_points = np.zeros((0, 3), dtype=np.float64)

        self._target_pos = np.zeros(3, dtype=np.float64)
        self._prev_target_pos = np.zeros(3, dtype=np.float64)
        self._line_start = np.zeros(3, dtype=np.float64)
        self._line_end = np.zeros(3, dtype=np.float64)
        self._line_dir = np.zeros(3, dtype=np.float64)
        self._line_len = 0.0

        self._step_count = 0
        self._prev_action = np.zeros(1, dtype=np.float64)
        self._prev_tip_pos = np.zeros(3, dtype=np.float64)
        self._last_dist = 0.0
        self._success_bonus_given = False
        self._last_target_projected = False
        self.render_camera_distance = 1.10
        self.render_camera_azimuth = 145.0
        self.render_camera_elevation = -22.0
        self._camera_lookat: Optional[np.ndarray] = None
        self._offscreen_camera = None

        # Initialize once to build spaces.
        obs, _info = self.reset(seed=seed)
        act_dim = int(self._joint_spec.q_adrs.shape[0]) if self._joint_spec is not None else 1
        if self._root_yaw_enabled_runtime:
            act_dim += 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float32)
        self._prev_action = np.zeros(act_dim, dtype=np.float64)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    # ------------------------------------------------------------------
    # Base sim accessors
    # ------------------------------------------------------------------

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        uw = self.unwrapped
        if hasattr(uw, "obs_dict"):
            return uw.obs_dict
        return {}

    def _refresh_obs_cache(self) -> None:
        uw = self.unwrapped
        try:
            sim_src = uw.sim_obsd if hasattr(uw, "sim_obsd") else uw.sim
            if hasattr(uw, "get_obs_dict"):
                uw.obs_dict = uw.get_obs_dict(sim_src)
        except Exception:
            try:
                uw.obs_dict = uw.get_obs_dict(uw.sim)
            except Exception:
                pass

    def _tip_pos(self) -> np.ndarray:
        obs = self._obs_dict()
        if "tip_pos" in obs:
            return np.asarray(obs["tip_pos"], dtype=np.float64).reshape(-1)[:3].copy()
        sim = self.unwrapped.sim
        if self._tip_site_id is None:
            try:
                self._tip_site_id = int(sim.model.site_name2id("IFtip"))
            except Exception:
                self._tip_site_id = 0
        try:
            return np.asarray(sim.data.site_xpos[self._tip_site_id], dtype=np.float64).copy()
        except Exception:
            return np.zeros(3, dtype=np.float64)

    def _ensure_camera_lookat(self) -> np.ndarray:
        look = np.asarray(self._camera_lookat, dtype=np.float64).reshape(3) if self._camera_lookat is not None else None
        if look is None or (not np.isfinite(look).all()):
            tip = self._tip_pos()
            look = np.asarray(tip, dtype=np.float64).reshape(3)
            if np.isfinite(self._target_pos).all():
                look = 0.5 * (look + np.asarray(self._target_pos, dtype=np.float64).reshape(3))
            self._camera_lookat = look.copy()
        return np.asarray(self._camera_lookat, dtype=np.float64).reshape(3).copy()

    def _apply_render_camera(self) -> None:
        look = self._ensure_camera_lookat()
        dist = float(np.clip(float(self.render_camera_distance), 0.20, 6.0))
        az = float(self.render_camera_azimuth)
        el = float(np.clip(float(self.render_camera_elevation), -89.0, 89.0))
        try:
            uw = self.unwrapped
            renderer = uw.sim.renderer
            if hasattr(renderer, "set_free_camera_settings"):
                renderer.set_free_camera_settings(
                    distance=dist - 2.0,  # MyoSuite renderer internally adds +2.
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
            if mujoco is not None and hasattr(renderer, "_update_camera_properties"):
                if self._offscreen_camera is None:
                    cam = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    cam.fixedcamid = -1
                    cam.trackbodyid = -1
                    self._offscreen_camera = cam
                renderer._update_camera_properties(self._offscreen_camera)
        except Exception:
            pass

    def _resolve_joint_names(self) -> Tuple[str, ...]:
        if self._joint_names_override is not None and len(self._joint_names_override) > 0:
            return self._joint_names_override
        if self._joint_profile in _JOINT_PROFILES:
            return _JOINT_PROFILES[self._joint_profile]
        return _JOINT_PROFILES["opensim_arm_wrist"]

    def _resolve_joint_spec(self) -> JointSpec:
        if self._joint_spec is not None:
            return self._joint_spec

        sim = self.unwrapped.sim
        m = sim.model
        names = self._resolve_joint_names()

        q_adrs: List[int] = []
        d_adrs: List[int] = []
        q_lo: List[float] = []
        q_hi: List[float] = []
        resolved: List[str] = []

        for jn in names:
            try:
                jid = int(m.joint_name2id(str(jn)))
            except Exception:
                continue
            qadr = int(m.jnt_qposadr[jid])
            dadr = int(m.jnt_dofadr[jid])
            lo = float(m.jnt_range[jid, 0])
            hi = float(m.jnt_range[jid, 1])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            q_adrs.append(qadr)
            d_adrs.append(dadr)
            q_lo.append(lo)
            q_hi.append(hi)
            resolved.append(str(jn))

        if not q_adrs:
            raise RuntimeError("No valid joint controls resolved. Adjust joint profile or joint names.")

        self._joint_spec = JointSpec(
            names=tuple(resolved),
            q_adrs=np.asarray(q_adrs, dtype=np.int32),
            dof_adrs=np.asarray(d_adrs, dtype=np.int32),
            q_lo=np.asarray(q_lo, dtype=np.float64),
            q_hi=np.asarray(q_hi, dtype=np.float64),
        )
        return self._joint_spec

    # ------------------------------------------------------------------
    # Target marker / palm helpers
    # ------------------------------------------------------------------

    def _find_target_site_id(self) -> Optional[int]:
        if self._target_site_id is not None:
            return self._target_site_id
        sim = self.unwrapped.sim
        candidates: List[Tuple[str, int]] = []
        for name in ("IFtip_target", "target", "reach_target", "target0"):
            try:
                sid = int(sim.model.site_name2id(name))
                candidates.append((name, sid))
            except Exception:
                continue
        if not candidates and mujoco is not None:
            for sid in range(int(sim.model.nsite)):
                try:
                    nm = mujoco.mj_id2name(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, sid)
                except Exception:
                    nm = None
                nms = str(nm or "").lower()
                if "target" in nms:
                    candidates.append((nms, sid))
        if candidates:
            candidates.sort(key=lambda x: (0 if str(x[0]) == "IFtip_target" else 1, str(x[0])))
            self._target_site_id = int(candidates[0][1])
            return self._target_site_id
        return None

    def _try_apply_target_marker(self, target: np.ndarray) -> None:
        sid = self._find_target_site_id()
        if sid is None:
            return
        sim = self.unwrapped.sim
        try:
            sim.model.site_pos[sid] = np.asarray(target, dtype=np.float64)
            sim.forward()
            self._refresh_obs_cache()
        except Exception:
            pass

    def _init_palm_reference(self) -> None:
        sim = self.unwrapped.sim
        m = sim.model
        d = sim.data
        self._palm_ref_q = {}

        for jn, qref in _PALM_PRESET.items():
            try:
                jid = int(m.joint_name2id(str(jn)))
            except Exception:
                continue
            qadr = int(m.jnt_qposadr[jid])
            lo = float(m.jnt_range[jid, 0])
            hi = float(m.jnt_range[jid, 1])
            self._palm_ref_q[qadr] = float(np.clip(float(qref), lo, hi))

        if self._palm_ref_q:
            qpos_bak = np.asarray(d.qpos, dtype=np.float64).copy()
            qvel_bak = np.asarray(d.qvel, dtype=np.float64).copy()
            for qadr, qref in self._palm_ref_q.items():
                d.qpos[qadr] = float(qref)
            d.qvel[:] = 0.0
            sim.forward()
            self._refresh_obs_cache()
            n = self._palm_normal()
            if n is not None:
                self._palm_ref_normal = n.copy()
            d.qpos[:] = qpos_bak
            d.qvel[:] = qvel_bak
            sim.forward()
            self._refresh_obs_cache()

        if self._palm_ref_normal is None:
            self._palm_ref_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _apply_palm_reference(self, alpha: float) -> None:
        if alpha <= 0.0:
            return
        if not self._palm_ref_q:
            return
        sim = self.unwrapped.sim
        d = sim.data
        m = sim.model
        for qadr, qref in self._palm_ref_q.items():
            if int(qadr) >= int(m.nq):
                continue
            cur = float(d.qpos[qadr])
            d.qpos[qadr] = float((1.0 - alpha) * cur + alpha * float(qref))
        sim.forward()
        self._refresh_obs_cache()

    def _palm_normal(self) -> Optional[np.ndarray]:
        if mujoco is None:
            return None
        sim = self.unwrapped.sim
        m = sim.model
        d = sim.data
        try:
            gid2 = int(m.geom_name2id("2mcskin"))
            gid5 = int(m.geom_name2id("5mcskin"))
            sid_tip = int(m.site_name2id("IFtip"))
            p2 = np.asarray(d.geom_xpos[gid2], dtype=np.float64)
            p5 = np.asarray(d.geom_xpos[gid5], dtype=np.float64)
            tip = np.asarray(d.site_xpos[sid_tip], dtype=np.float64)
        except Exception:
            return None
        u = p5 - p2
        v = tip - 0.5 * (p2 + p5)
        n = np.cross(u, v)
        nn = float(np.linalg.norm(n))
        if nn <= 1e-9:
            return None
        return (n / nn).astype(np.float64)

    def _palm_down_error(self) -> float:
        n = self._palm_normal()
        if n is None:
            return 0.0
        ref = np.asarray(self._palm_ref_normal, dtype=np.float64).reshape(3)
        nr = float(np.linalg.norm(ref))
        if nr <= 1e-9:
            return 0.0
        ref = ref / nr
        cos = float(np.clip(np.dot(n, ref), -1.0, 1.0))
        return float(0.5 * (1.0 - cos))

    # ------------------------------------------------------------------
    # Root yaw helpers
    # ------------------------------------------------------------------

    def _init_root_yaw_state(self) -> None:
        self._root_body_id = None
        self._root_base_quat = None
        self._root_yaw_rad = 0.0
        self._root_yaw_enabled_runtime = False
        if not self.enable_root_yaw:
            return
        try:
            sim = self.unwrapped.sim
            m = sim.model
            rid = int(m.body_name2id(self.root_body_name))
            self._root_body_id = rid
            self._root_base_quat = np.asarray(m.body_quat[rid], dtype=np.float64).copy()
            self._root_yaw_enabled_runtime = True
        except Exception:
            self._root_body_id = None
            self._root_base_quat = None
            self._root_yaw_enabled_runtime = False

    def _apply_root_yaw_delta(self, a_yaw: float) -> None:
        if not self._root_yaw_enabled_runtime:
            return
        if self._root_body_id is None or self._root_base_quat is None:
            return
        yaw_cmd = float(np.clip(a_yaw, -1.0, 1.0))
        if self.root_yaw_action_mode == "absolute":
            yaw_target = float(yaw_cmd * self.root_yaw_limit_rad)
            a = float(self.root_yaw_smooth_alpha)
            self._root_yaw_rad = float((1.0 - a) * self._root_yaw_rad + a * yaw_target)
            self._root_yaw_rad = float(np.clip(self._root_yaw_rad, -self.root_yaw_limit_rad, self.root_yaw_limit_rad))
        else:
            self._root_yaw_rad = float(
                np.clip(
                    self._root_yaw_rad + self.root_yaw_step_rad * yaw_cmd,
                    -self.root_yaw_limit_rad,
                    self.root_yaw_limit_rad,
                )
            )
        q_yaw = _axis_quat(2, self._root_yaw_rad)
        q_new = _quat_normalize(_quat_mul(q_yaw, self._root_base_quat))
        sim = self.unwrapped.sim
        sim.model.body_quat[self._root_body_id] = q_new
        sim.forward()
        self._refresh_obs_cache()

    def _root_heading_xy(self) -> Optional[np.ndarray]:
        if not self._root_yaw_enabled_runtime:
            return None
        if self._root_body_id is None:
            return None
        try:
            sim = self.unwrapped.sim
            R = np.asarray(sim.data.xmat[self._root_body_id], dtype=np.float64).reshape(3, 3)
            f = np.array([float(R[0, 0]), float(R[1, 0])], dtype=np.float64)
            nf = float(np.linalg.norm(f))
            if nf <= 1e-9:
                return None
            return f / nf
        except Exception:
            return None

    def _yaw_line_error(self) -> float:
        line_xy = np.asarray(self._line_dir[:2], dtype=np.float64).reshape(2)
        nl = float(np.linalg.norm(line_xy))
        if nl <= 1e-9:
            return 0.0
        line_xy = line_xy / nl
        h = self._root_heading_xy()
        if h is None:
            return 0.0
        cos = float(np.clip(np.dot(h, line_xy), -1.0, 1.0))
        return float(0.5 * (1.0 - cos))

    # ------------------------------------------------------------------
    # Target sampling / projection / line tracking
    # ------------------------------------------------------------------

    def _sample_reachable_points(self, spec: JointSpec, n_samples: int) -> np.ndarray:
        sim = self.unwrapped.sim
        d = sim.data
        m = sim.model

        qpos_bak = np.asarray(d.qpos, dtype=np.float64).copy()
        qvel_bak = np.asarray(d.qvel, dtype=np.float64).copy()
        ctrl_bak = np.asarray(d.ctrl, dtype=np.float64).copy() if hasattr(d, "ctrl") else None
        act_bak = np.asarray(d.act, dtype=np.float64).copy() if hasattr(d, "act") else None
        body_quat_bak = None
        if self._root_body_id is not None:
            body_quat_bak = np.asarray(m.body_quat[self._root_body_id], dtype=np.float64).copy()

        pts: List[np.ndarray] = []
        for _ in range(int(max(1, n_samples))):
            q_rand = self.rng.uniform(spec.q_lo, spec.q_hi)
            d.qpos[spec.q_adrs] = q_rand
            d.qvel[spec.dof_adrs] = 0.0
            if ctrl_bak is not None and hasattr(d, "ctrl"):
                d.ctrl[:] = 0.0
            if act_bak is not None and hasattr(d, "act"):
                d.act[:] = 0.0

            if self._root_yaw_enabled_runtime and self._root_body_id is not None and self._root_base_quat is not None:
                yaw = float(self.rng.uniform(-self.root_yaw_limit_rad, self.root_yaw_limit_rad))
                q_new = _quat_normalize(_quat_mul(_axis_quat(2, yaw), self._root_base_quat))
                m.body_quat[self._root_body_id] = q_new

            sim.forward()
            self._refresh_obs_cache()
            tip = self._tip_pos()
            if np.isfinite(tip).all():
                if self.target_jitter_std > 0.0:
                    tip = tip + self.rng.normal(0.0, self.target_jitter_std, size=3)
                pts.append(tip.astype(np.float64))

        d.qpos[:] = qpos_bak
        d.qvel[:] = qvel_bak
        if ctrl_bak is not None and hasattr(d, "ctrl"):
            d.ctrl[:] = ctrl_bak
        if act_bak is not None and hasattr(d, "act"):
            d.act[:] = act_bak
        if body_quat_bak is not None and self._root_body_id is not None:
            m.body_quat[self._root_body_id] = body_quat_bak
        sim.forward()
        self._refresh_obs_cache()

        if not pts:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    def _sample_reachable_target(self, from_tip: np.ndarray) -> np.ndarray:
        from_tip = np.asarray(from_tip, dtype=np.float64).reshape(3)
        pts = np.asarray(self._reachable_points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            spec = self._resolve_joint_spec()
            pts = self._sample_reachable_points(spec, self.target_rollout_samples)
            self._reachable_points = pts.copy()

        if pts.shape[0] > 0:
            dists = np.linalg.norm(pts - from_tip[None, :], axis=1)
            keep = np.logical_and(
                dists >= float(self.target_min_start_dist),
                dists <= float(self.target_max_start_dist),
            )
            if bool(np.any(keep)):
                pool = pts[keep]
            else:
                order = np.argsort(dists)
                n_tail = max(1, int(math.ceil(0.35 * len(order))))
                pool = pts[order[-n_tail:]]
            idx = int(self.rng.integers(0, pool.shape[0]))
            return np.asarray(pool[idx], dtype=np.float64).reshape(3)

        return np.asarray(from_tip, dtype=np.float64) + np.array([0.12, 0.00, -0.02], dtype=np.float64)

    def _project_target(self, target: np.ndarray) -> Tuple[np.ndarray, bool]:
        t = np.asarray(target, dtype=np.float64).reshape(3)
        if not self.project_targets_to_reachable:
            return t.copy(), False
        pts = np.asarray(self._reachable_points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return t.copy(), False
        d = np.linalg.norm(pts - t[None, :], axis=1)
        idx = int(np.argmin(d))
        proj = np.asarray(pts[idx], dtype=np.float64).reshape(3)
        moved = bool(np.linalg.norm(proj - t) > 1e-8)
        return proj, moved

    def estimate_target_bounds(self, n_samples: int = 256, margin: float = 0.10) -> Tuple[float, float, float, float]:
        pts = np.asarray(self._reachable_points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            spec = self._resolve_joint_spec()
            pts = self._sample_reachable_points(spec, int(max(1, n_samples)))
            self._reachable_points = pts.copy()
        if pts.shape[0] == 0:
            tip = self._tip_pos()
            r = 0.90
            return (float(tip[0] - r), float(tip[0] + r), float(tip[1] - r), float(tip[1] + r))
        x_lo = float(np.min(pts[:, 0]) - margin)
        x_hi = float(np.max(pts[:, 0]) + margin)
        y_lo = float(np.min(pts[:, 1]) - margin)
        y_hi = float(np.max(pts[:, 1]) + margin)
        return (x_lo, x_hi, y_lo, y_hi)

    def _set_target_and_line(self, new_target: np.ndarray, *, line_from_tip: bool = False) -> None:
        tip = self._tip_pos()
        if bool(line_from_tip):
            prev = tip.copy()
        else:
            prev = self._target_pos.copy() if np.isfinite(self._target_pos).all() and np.linalg.norm(self._target_pos) > 0 else tip.copy()
        self._prev_target_pos = prev.copy()
        self._target_pos = np.asarray(new_target, dtype=np.float64).reshape(3)
        self._line_start = prev.copy()
        self._line_end = self._target_pos.copy()
        vec = self._line_end - self._line_start
        self._line_len = float(np.linalg.norm(vec))
        if self._line_len > 1e-9:
            self._line_dir = (vec / self._line_len).astype(np.float64)
        else:
            self._line_dir = np.zeros(3, dtype=np.float64)
        self._success_bonus_given = False
        self._last_dist = float(np.linalg.norm(self._target_pos - tip))
        self._try_apply_target_marker(self._target_pos)

    def _line_deviation(self, tip: np.ndarray) -> float:
        if self._line_len <= 1e-9:
            return 0.0
        return _point_segment_distance(tip, self._line_start, self._line_end)

    def _line_progress(self, tip: np.ndarray) -> float:
        if self._line_len <= 1e-9:
            return 0.0
        rel = np.asarray(tip, dtype=np.float64).reshape(3) - self._line_start
        return float(np.clip(np.dot(rel, self._line_dir) / max(self._line_len, 1e-9), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Observation / metrics
    # ------------------------------------------------------------------

    def _joint_limit_proximity(self, q: np.ndarray, spec: JointSpec) -> float:
        span = np.maximum(spec.q_hi - spec.q_lo, 1e-6)
        margin = np.minimum(q - spec.q_lo, spec.q_hi - q)
        norm_margin = np.clip(margin / (0.20 * span), 0.0, 1.0)
        prox = 1.0 - norm_margin
        return float(np.mean(prox))

    def _directness_error(self, tip: np.ndarray) -> float:
        prev_tip = np.asarray(self._prev_tip_pos, dtype=np.float64).reshape(3)
        cur_tip = np.asarray(tip, dtype=np.float64).reshape(3)
        move = cur_tip - prev_tip
        target_vec = self._target_pos - prev_tip
        n_move = float(np.linalg.norm(move))
        n_target = float(np.linalg.norm(target_vec))
        if n_move <= 1e-8 or n_target <= 1e-8:
            return 0.0
        move = move / n_move
        target_vec = target_vec / n_target
        cos = float(np.clip(np.dot(move, target_vec), -1.0, 1.0))
        return float(0.5 * (1.0 - cos))

    def _stability_penalty(self, tip: np.ndarray, dist: float) -> float:
        prev_tip = np.asarray(self._prev_tip_pos, dtype=np.float64).reshape(3)
        cur_tip = np.asarray(tip, dtype=np.float64).reshape(3)
        speed = float(np.linalg.norm(cur_tip - prev_tip))
        near = float(np.clip(1.0 - dist / max(4.0 * self.success_radius, 1e-6), 0.0, 1.0))
        return float(speed * near)

    def _effective_palm_weight(self, dist: float) -> float:
        near = float(np.clip(1.0 - dist / self.palm_near_target_scale, 0.0, 1.0))
        return float(self.palm_weight * (0.30 + 0.70 * near))

    def _build_obs(self, spec: JointSpec) -> np.ndarray:
        sim = self.unwrapped.sim
        d = sim.data
        q = np.asarray(d.qpos[spec.q_adrs], dtype=np.float64)
        qv = np.asarray(d.qvel[spec.dof_adrs], dtype=np.float64)
        q_span = np.maximum(spec.q_hi - spec.q_lo, 1e-6)
        q_norm = np.clip(2.0 * (q - spec.q_lo) / q_span - 1.0, -1.0, 1.0)
        qv_norm = np.tanh(0.1 * qv)

        tip = self._tip_pos()
        err = self._target_pos - tip
        dist = float(np.linalg.norm(err))
        palm_err = self._palm_down_error()
        line_dev = self._line_deviation(tip)
        line_prog = self._line_progress(tip)
        yaw_err = self._yaw_line_error()
        direct_err = self._directness_error(tip)
        stability_cost = self._stability_penalty(tip, dist)
        yaw_norm = float(np.clip(self._root_yaw_rad / max(self.root_yaw_limit_rad, 1e-6), -1.0, 1.0))

        obs = np.concatenate(
            [
                q_norm.astype(np.float32),
                qv_norm.astype(np.float32),
                tip.astype(np.float32),
                self._target_pos.astype(np.float32),
                self._prev_target_pos.astype(np.float32),
                err.astype(np.float32),
                self._line_dir.astype(np.float32),
                np.array([dist, line_dev, line_prog, palm_err, yaw_err, direct_err, stability_cost, yaw_norm], dtype=np.float32),
                self._prev_action.astype(np.float32),
            ]
        )
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target(
        self,
        target_world: Sequence[float],
        *,
        project_to_reachable: bool = True,
        line_from_tip: bool = False,
    ) -> None:
        t = np.asarray(target_world, dtype=np.float64).reshape(-1)[:3]
        if t.shape[0] < 3:
            raise ValueError("target_world must have 3 elements")
        if project_to_reachable and self.project_targets_to_reachable:
            t, moved = self._project_target(t)
            self._last_target_projected = bool(moved)
        else:
            self._last_target_projected = False
        self._set_target_and_line(t, line_from_tip=bool(line_from_tip))

    def get_tip_position(self) -> np.ndarray:
        return self._tip_pos().copy()

    def get_target_position(self) -> np.ndarray:
        return self._target_pos.copy()

    def get_previous_target_position(self) -> np.ndarray:
        return self._prev_target_pos.copy()

    def get_target_line_segment(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._line_start.copy(), self._line_end.copy()

    def get_joint_names(self) -> Tuple[str, ...]:
        spec = self._resolve_joint_spec()
        return tuple(spec.names)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        _ = options
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        _, info = self.env.reset(seed=seed)
        self._refresh_obs_cache()
        spec = self._resolve_joint_spec()
        self._tip_site_id = None
        self._target_site_id = None

        self._init_root_yaw_state()
        self._init_palm_reference()

        if self.palm_lock_alpha > 0.0:
            self._apply_palm_reference(alpha=min(1.0, max(self.palm_lock_alpha, 0.30)))

        self._reachable_points = self._sample_reachable_points(spec, self.target_rollout_samples)
        tip = self._tip_pos()
        tgt = self._sample_reachable_target(from_tip=tip)
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._set_target_and_line(tgt)
        if self._camera_lookat is None or (not np.isfinite(np.asarray(self._camera_lookat, dtype=np.float64)).all()):
            self._camera_lookat = (0.5 * (self._tip_pos() + self._target_pos)).astype(np.float64)

        self._step_count = 0
        act_dim = int(spec.q_adrs.shape[0]) + (1 if self._root_yaw_enabled_runtime else 0)
        self._prev_action = np.zeros(act_dim, dtype=np.float64)
        self._prev_tip_pos = self._tip_pos().copy()
        self._success_bonus_given = False

        obs = self._build_obs(spec)
        out_info = dict(info)
        out_info.update(
            {
                "target_pos": self._target_pos.copy(),
                "prev_target_pos": self._prev_target_pos.copy(),
                "tip_pos": self._tip_pos().copy(),
                "distance_error": float(self._last_dist),
                "line_deviation": float(self._line_deviation(self._tip_pos())),
                "line_progress": float(self._line_progress(self._tip_pos())),
                "yaw_line_error": float(self._yaw_line_error()),
                "palm_down_error": float(self._palm_down_error()),
                "directness_error": float(0.0),
                "stability_penalty": float(0.0),
                "joint_names": tuple(spec.names),
                "root_yaw_deg": float(np.rad2deg(self._root_yaw_rad)),
                "root_yaw_action_mode": str(self.root_yaw_action_mode),
            }
        )
        return obs, out_info

    def step(self, action: np.ndarray):
        spec = self._resolve_joint_spec()
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        expected_dim = int(spec.q_adrs.shape[0]) + (1 if self._root_yaw_enabled_runtime else 0)
        if a.shape[0] != expected_dim:
            raise ValueError(f"Action dim mismatch: got {a.shape[0]} expected {expected_dim}")
        a = np.clip(a, -1.0, 1.0)

        sim = self.unwrapped.sim
        d = sim.data
        m = sim.model

        qpos_bak = np.asarray(d.qpos, dtype=np.float64).copy()
        qvel_bak = np.asarray(d.qvel, dtype=np.float64).copy()
        body_quat_bak = None
        if self._root_body_id is not None:
            body_quat_bak = np.asarray(m.body_quat[self._root_body_id], dtype=np.float64).copy()
        yaw_bak = float(self._root_yaw_rad)

        # Joint action
        joint_act = a[: spec.q_adrs.shape[0]]
        q_cur = np.asarray(d.qpos[spec.q_adrs], dtype=np.float64)
        q_next = np.clip(q_cur + self.action_max_delta_rad * joint_act, spec.q_lo, spec.q_hi)
        d.qpos[spec.q_adrs] = q_next
        d.qvel[spec.dof_adrs] = 0.0

        # Root-yaw action (last dim)
        if self._root_yaw_enabled_runtime:
            self._apply_root_yaw_delta(float(a[-1]))

        if hasattr(d, "ctrl"):
            d.ctrl[:] = 0.0
        if hasattr(d, "act"):
            d.act[:] = 0.0
        sim.forward()

        if self.palm_lock_alpha > 0.0:
            self._apply_palm_reference(alpha=self.palm_lock_alpha)
        self._refresh_obs_cache()

        tip = self._tip_pos()
        state_bad = not np.isfinite(tip).all() or (not np.isfinite(d.qpos).all()) or (not np.isfinite(d.qvel).all())
        out_of_control = False
        if state_bad:
            out_of_control = True
            d.qpos[:] = qpos_bak
            d.qvel[:] = qvel_bak
            if body_quat_bak is not None and self._root_body_id is not None:
                m.body_quat[self._root_body_id] = body_quat_bak
            self._root_yaw_rad = yaw_bak
            sim.forward()
            self._refresh_obs_cache()
            tip = self._tip_pos()

        retargeted = False
        if (not out_of_control) and self.retarget_interval_steps > 0 and self._step_count > 0:
            if (self._step_count % self.retarget_interval_steps) == 0:
                new_t = self._sample_reachable_target(from_tip=tip)
                self._set_target_and_line(new_t)
                retargeted = True

        dist = float(np.linalg.norm(self._target_pos - tip))
        progress = float(self._last_dist - dist)
        palm_err = float(self._palm_down_error())
        line_dev = float(self._line_deviation(tip))
        yaw_err = float(self._yaw_line_error())
        direct_err = float(self._directness_error(tip))
        stability_cost = float(self._stability_penalty(tip, dist))
        away_cost = float(max(-progress, 0.0))
        palm_w_eff = float(self._effective_palm_weight(dist))
        action_cost = float(np.mean(np.abs(a)))
        smooth_cost = float(np.mean((a - self._prev_action) ** 2))
        joint_limit_cost = float(self._joint_limit_proximity(np.asarray(d.qpos[spec.q_adrs], dtype=np.float64), spec))

        reward = 0.0
        reward -= self.distance_weight * dist
        reward += self.progress_weight * progress
        reward -= self.line_deviation_weight * line_dev
        reward -= self.yaw_line_weight * yaw_err
        reward -= palm_w_eff * palm_err
        reward -= self.directness_weight * direct_err
        reward -= self.stability_weight * stability_cost
        reward -= self.away_penalty_weight * away_cost
        reward -= self.action_cost_weight * action_cost
        reward -= self.smoothness_weight * smooth_cost
        reward -= self.joint_limit_weight * joint_limit_cost
        reward -= self.time_penalty
        if out_of_control:
            reward -= 2.0

        reached = bool(dist <= self.success_radius)
        if reached and (not self._success_bonus_given):
            reward += self.success_bonus
            self._success_bonus_given = True

        if self.reward_clip > 0.0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self._step_count += 1
        terminated = bool(reached and self.terminate_on_success)
        truncated = bool(self._step_count >= self.max_steps)
        if out_of_control:
            truncated = True

        retargeted_after_reward = False
        # Optional auto-retarget while keeping episode alive after success.
        if reached and (not self.terminate_on_success) and self.auto_retarget_on_success:
            new_t = self._sample_reachable_target(from_tip=tip)
            self._set_target_and_line(new_t)
            retargeted = True
            retargeted_after_reward = True

        if not retargeted_after_reward:
            self._last_dist = dist
        self._prev_action = a.copy()
        self._prev_tip_pos = tip.copy()

        obs = self._build_obs(spec)
        info = {
            "target_pos": self._target_pos.copy(),
            "prev_target_pos": self._prev_target_pos.copy(),
            "tip_pos": tip.copy(),
            "distance_error": float(dist),
            "distance_progress": float(progress),
            "line_deviation": float(line_dev),
            "line_progress": float(self._line_progress(tip)),
            "yaw_line_error": float(yaw_err),
            "palm_down_error": float(palm_err),
            "palm_weight_effective": float(palm_w_eff),
            "directness_error": float(direct_err),
            "stability_penalty": float(stability_cost),
            "away_penalty": float(away_cost),
            "action_cost": float(action_cost),
            "smoothness_cost": float(smooth_cost),
            "joint_limit_cost": float(joint_limit_cost),
            "root_yaw_deg": float(np.rad2deg(self._root_yaw_rad)),
            "root_yaw_action_mode": str(self.root_yaw_action_mode),
            "target_projected": bool(self._last_target_projected),
            "retargeted": bool(retargeted),
            "auto_retarget_on_success": bool(self.auto_retarget_on_success),
            "out_of_control": bool(out_of_control),
            "success": bool(reached),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        try:
            self._apply_render_camera()
            return self.unwrapped.mj_render()
        except Exception:
            return None

    def render_rgb_array(self, width: int = 640, height: int = 480) -> np.ndarray:
        self._apply_render_camera()
        cam = self._offscreen_camera if self._offscreen_camera is not None else -1
        frame = self.unwrapped.sim.renderer.render_offscreen(
            width=int(width),
            height=int(height),
            camera_id=cam,
            rgb=True,
            depth=False,
            segmentation=False,
        )
        return np.asarray(frame).copy()

    def close(self) -> None:
        self.env.close()
