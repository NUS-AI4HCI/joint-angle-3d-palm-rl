"""Simple 3D reaching env (joint-angle control, MyoSuite backend)."""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

warnings.filterwarnings(
    "ignore",
    message=r".*Overriding environment .* already in registry.*",
    category=UserWarning,
)

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import myosuite  # noqa: F401
except Exception:
    myosuite = None  # type: ignore

try:
    import mujoco
except Exception:
    mujoco = None  # type: ignore


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


class SimpleJointReach3DEnv(gym.Env):
    """Position-only reaching with joint action and flat vector observation."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        env_id: str = "myoArmReachRandom-v0",
        env_make_kwargs: Optional[Dict[str, object]] = None,
        joint_profile: str = "opensim_arm_wrist",
        joint_names: Optional[Sequence[str]] = None,
        action_mode: str = "delta",
        action_max_delta_deg: float = 5.0,
        target_rollout_samples: int = 128,
        target_min_start_dist: float = 0.08,
        target_max_start_dist: float = 0.55,
        randomize_start_pose: bool = True,
        start_pose_trials: int = 64,
        project_external_targets: bool = True,
        success_radius: float = 0.03,
        success_steps_required: int = 10,
        max_steps: int = 200,
        reward_mode: str = "distance",
        action_penalty: float = 1e-3,
        palm_weight: float = 0.20,
        palm_near_target_scale: float = 0.12,
        success_bonus: float = 2.0,
        terminate_on_success: bool = True,
        hide_background: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if myosuite is None:
            raise RuntimeError("myosuite is required for SimpleJointReach3DEnv")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.env = gym.make(env_id, **dict(env_make_kwargs or {}))
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

        self.joint_profile = str(joint_profile).strip().lower()
        self.joint_names_override = tuple(str(n) for n in joint_names) if joint_names else None
        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in {"delta", "absolute"}:
            self.action_mode = "delta"
        self.action_max_delta_rad = float(np.deg2rad(max(1e-4, action_max_delta_deg)))

        self.target_rollout_samples = int(max(16, target_rollout_samples))
        self.target_min_start_dist = float(max(0.0, target_min_start_dist))
        self.target_max_start_dist = float(max(self.target_min_start_dist + 1e-6, target_max_start_dist))
        self.randomize_start_pose = bool(randomize_start_pose)
        self.start_pose_trials = int(max(1, start_pose_trials))
        self.project_external_targets = bool(project_external_targets)

        self.success_radius = float(max(1e-4, success_radius))
        self.success_steps_required = int(max(1, success_steps_required))
        self.max_steps = int(max(1, max_steps))

        self.reward_mode = str(reward_mode).strip().lower()
        if self.reward_mode not in {"distance", "progress"}:
            self.reward_mode = "distance"
        self.action_penalty = float(max(0.0, action_penalty))
        self.palm_weight = float(max(0.0, palm_weight))
        self.palm_near_target_scale = float(max(1e-4, palm_near_target_scale))
        self.success_bonus = float(success_bonus)
        self.terminate_on_success = bool(terminate_on_success)
        self.hide_background = bool(hide_background)

        self._joint_spec: Optional[JointSpec] = None
        self._target_site_id: Optional[int] = None
        self._tip_site_id: Optional[int] = None
        self._reachable_points = np.zeros((0, 3), dtype=np.float64)
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._prev_dist = 0.0
        self._prev_action = np.zeros(1, dtype=np.float64)
        self._step_count = 0
        self._success_streak = 0
        self._success_bonus_given = False
        self._palm_ref_q: Dict[int, float] = {}
        self._palm_ref_normal: Optional[np.ndarray] = None
        self.render_camera_distance = 1.10
        self.render_camera_azimuth = 145.0
        self.render_camera_elevation = -22.0
        self._camera_lookat: Optional[np.ndarray] = None
        self._offscreen_camera = None
        self._background_alpha_backup: Optional[np.ndarray] = None
        self._background_geom_ids: Optional[np.ndarray] = None

        obs, _ = self.reset(seed=seed)
        act_dim = int(self._joint_spec.q_adrs.shape[0]) if self._joint_spec is not None else 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs.shape[0]),), dtype=np.float32)
        self._prev_action = np.zeros(act_dim, dtype=np.float64)

    @property
    def unwrapped(self):
        return self.env.unwrapped

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

    def _resolve_joint_names(self) -> Tuple[str, ...]:
        if self.joint_names_override and len(self.joint_names_override) > 0:
            return self.joint_names_override
        return _JOINT_PROFILES.get(self.joint_profile, _JOINT_PROFILES["opensim_arm_wrist"])

    def _resolve_joint_spec(self) -> JointSpec:
        if self._joint_spec is not None:
            return self._joint_spec
        sim = self.unwrapped.sim
        m = sim.model
        q_adrs: List[int] = []
        d_adrs: List[int] = []
        q_lo: List[float] = []
        q_hi: List[float] = []
        names: List[str] = []
        for jn in self._resolve_joint_names():
            try:
                jid = int(m.joint_name2id(str(jn)))
            except Exception:
                continue
            lo = float(m.jnt_range[jid, 0])
            hi = float(m.jnt_range[jid, 1])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            q_adrs.append(int(m.jnt_qposadr[jid]))
            d_adrs.append(int(m.jnt_dofadr[jid]))
            q_lo.append(lo)
            q_hi.append(hi)
            names.append(str(jn))
        if not q_adrs:
            raise RuntimeError("No valid joints resolved.")
        self._joint_spec = JointSpec(
            names=tuple(names),
            q_adrs=np.asarray(q_adrs, dtype=np.int32),
            dof_adrs=np.asarray(d_adrs, dtype=np.int32),
            q_lo=np.asarray(q_lo, dtype=np.float64),
            q_hi=np.asarray(q_hi, dtype=np.float64),
        )
        return self._joint_spec

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

        if not self._palm_ref_q:
            if self._palm_ref_normal is None:
                self._palm_ref_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            return

        qpos_bak = np.asarray(d.qpos, dtype=np.float64).copy()
        qvel_bak = np.asarray(d.qvel, dtype=np.float64).copy()
        for qadr, qref in self._palm_ref_q.items():
            d.qpos[int(qadr)] = float(qref)
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

    def _palm_normal(self) -> Optional[np.ndarray]:
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

    def _palm_alignment_error(self) -> float:
        n = self._palm_normal()
        ref = np.asarray(self._palm_ref_normal, dtype=np.float64).reshape(3) if self._palm_ref_normal is not None else None
        if n is None or ref is None:
            return 0.0
        nr = float(np.linalg.norm(ref))
        if nr <= 1e-9:
            return 0.0
        ref = ref / nr
        cos = float(np.clip(np.dot(n, ref), -1.0, 1.0))
        return float(0.5 * (1.0 - cos))

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

    def _background_geom_indices(self) -> np.ndarray:
        if self._background_geom_ids is not None:
            return self._background_geom_ids
        try:
            m = self.unwrapped.sim.model
            ids: List[int] = []
            for gid in range(int(m.ngeom)):
                if int(m.geom_bodyid[gid]) != 0:
                    continue
                geom_name = str(m.id2name(int(gid), "geom") or "").strip().lower()
                mesh_name = ""
                try:
                    mesh_id = int(m.geom_dataid[gid])
                    if mesh_id >= 0:
                        mesh_name = str(m.id2name(mesh_id, "mesh") or "").strip().lower()
                except Exception:
                    mesh_name = ""
                if any(tok in mesh_name for tok in ("meshscene", "myosuite_scene", "logo", "myosuite_logo")):
                    ids.append(int(gid))
                    continue
                if geom_name in {"logo", "scene", "background"}:
                    ids.append(int(gid))
            self._background_geom_ids = np.asarray(ids, dtype=np.int32)
        except Exception:
            self._background_geom_ids = np.zeros((0,), dtype=np.int32)
        return self._background_geom_ids

    def _apply_background_visibility(self) -> None:
        try:
            m = self.unwrapped.sim.model
            bg_ids = self._background_geom_indices()
            if bg_ids.size == 0:
                return
            if self._background_alpha_backup is None or int(self._background_alpha_backup.shape[0]) != int(m.ngeom):
                self._background_alpha_backup = np.asarray(m.geom_rgba[:, 3], dtype=np.float64).copy()
            if self.hide_background:
                for gid in bg_ids:
                    m.geom_rgba[int(gid), 3] = 0.0
            else:
                for gid in bg_ids:
                    m.geom_rgba[int(gid), 3] = float(self._background_alpha_backup[int(gid)])
        except Exception:
            pass

    def _find_target_site_id(self) -> Optional[int]:
        if self._target_site_id is not None:
            return self._target_site_id
        sim = self.unwrapped.sim
        for name in ("IFtip_target", "target", "reach_target", "target0"):
            try:
                self._target_site_id = int(sim.model.site_name2id(name))
                return self._target_site_id
            except Exception:
                continue
        return None

    def _apply_target_marker(self, target: np.ndarray) -> None:
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

    def _sample_reachable_points(self, spec: JointSpec, n_samples: int) -> np.ndarray:
        sim = self.unwrapped.sim
        d = sim.data
        qpos_bak = np.asarray(d.qpos, dtype=np.float64).copy()
        qvel_bak = np.asarray(d.qvel, dtype=np.float64).copy()
        ctrl_bak = np.asarray(d.ctrl, dtype=np.float64).copy() if hasattr(d, "ctrl") else None
        act_bak = np.asarray(d.act, dtype=np.float64).copy() if hasattr(d, "act") else None
        pts: List[np.ndarray] = []
        for _ in range(int(max(1, n_samples))):
            q_rand = self.rng.uniform(spec.q_lo, spec.q_hi)
            d.qpos[spec.q_adrs] = q_rand
            d.qvel[spec.dof_adrs] = 0.0
            if ctrl_bak is not None and hasattr(d, "ctrl"):
                d.ctrl[:] = 0.0
            if act_bak is not None and hasattr(d, "act"):
                d.act[:] = 0.0
            sim.forward()
            self._refresh_obs_cache()
            tip = self._tip_pos()
            if np.isfinite(tip).all():
                pts.append(tip.astype(np.float64))
        d.qpos[:] = qpos_bak
        d.qvel[:] = qvel_bak
        if ctrl_bak is not None and hasattr(d, "ctrl"):
            d.ctrl[:] = ctrl_bak
        if act_bak is not None and hasattr(d, "act"):
            d.act[:] = act_bak
        sim.forward()
        self._refresh_obs_cache()
        if not pts:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    def _sample_target(self, from_tip: np.ndarray, *, strict_distance_window: bool = False) -> Optional[np.ndarray]:
        pts = np.asarray(self._reachable_points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            spec = self._resolve_joint_spec()
            pts = self._sample_reachable_points(spec, self.target_rollout_samples)
            self._reachable_points = pts.copy()
        if pts.shape[0] == 0:
            if strict_distance_window:
                return None
            return np.asarray(from_tip, dtype=np.float64) + np.array([0.12, 0.0, -0.02], dtype=np.float64)
        d = np.linalg.norm(pts - np.asarray(from_tip, dtype=np.float64)[None, :], axis=1)
        keep = np.logical_and(d >= self.target_min_start_dist, d <= self.target_max_start_dist)
        if strict_distance_window and (not np.any(keep)):
            return None
        pool = pts[keep] if np.any(keep) else pts
        idx = int(self.rng.integers(0, pool.shape[0]))
        return np.asarray(pool[idx], dtype=np.float64).reshape(3)

    def _randomize_start_pose(self, spec: JointSpec) -> Tuple[np.ndarray, np.ndarray]:
        sim = self.unwrapped.sim
        d = sim.data
        for _ in range(self.start_pose_trials):
            q_try = self.rng.uniform(spec.q_lo, spec.q_hi)
            d.qpos[spec.q_adrs] = q_try
            d.qvel[spec.dof_adrs] = 0.0
            if hasattr(d, "ctrl"):
                d.ctrl[:] = 0.0
            if hasattr(d, "act"):
                d.act[:] = 0.0
            sim.forward()
            self._refresh_obs_cache()
            tip = self._tip_pos()
            if not np.isfinite(tip).all():
                continue
            tgt = self._sample_target(tip, strict_distance_window=True)
            if tgt is None:
                continue
            return tip.copy(), np.asarray(tgt, dtype=np.float64).reshape(3).copy()
        tip = self._tip_pos()
        tgt = self._sample_target(tip, strict_distance_window=False)
        if tgt is None:
            tgt = np.asarray(tip, dtype=np.float64) + np.array([0.12, 0.0, -0.02], dtype=np.float64)
        return tip.copy(), np.asarray(tgt, dtype=np.float64).reshape(3).copy()

    def _project_target(self, target: np.ndarray) -> np.ndarray:
        t = np.asarray(target, dtype=np.float64).reshape(3)
        pts = np.asarray(self._reachable_points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return t
        d = np.linalg.norm(pts - t[None, :], axis=1)
        return np.asarray(pts[int(np.argmin(d))], dtype=np.float64).reshape(3)

    def set_target(self, target_world: Sequence[float], *, project_to_reachable: Optional[bool] = None) -> None:
        t = np.asarray(target_world, dtype=np.float64).reshape(-1)[:3]
        if t.shape[0] < 3:
            raise ValueError("target_world must have 3 elements")
        do_proj = self.project_external_targets if project_to_reachable is None else bool(project_to_reachable)
        if do_proj:
            t = self._project_target(t)
        self._target_pos = np.asarray(t, dtype=np.float64).reshape(3)
        tip = self._tip_pos()
        self._prev_dist = float(np.linalg.norm(self._target_pos - tip))
        self._success_streak = 0
        self._success_bonus_given = False
        self._apply_target_marker(self._target_pos)

    def get_tip_position(self) -> np.ndarray:
        return self._tip_pos().copy()

    def get_target_position(self) -> np.ndarray:
        return self._target_pos.copy()

    def estimate_target_bounds(self, n_samples: int = 256, margin: float = 0.08) -> Tuple[float, float, float, float]:
        pts = np.asarray(self._reachable_points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            spec = self._resolve_joint_spec()
            pts = self._sample_reachable_points(spec, int(max(1, n_samples)))
            self._reachable_points = pts.copy()
        if pts.shape[0] == 0:
            tip = self._tip_pos()
            r = 0.8
            return (float(tip[0] - r), float(tip[0] + r), float(tip[1] - r), float(tip[1] + r))
        return (
            float(np.min(pts[:, 0]) - margin),
            float(np.max(pts[:, 0]) + margin),
            float(np.min(pts[:, 1]) - margin),
            float(np.max(pts[:, 1]) + margin),
        )

    def _build_obs(self, spec: JointSpec) -> np.ndarray:
        sim = self.unwrapped.sim
        d = sim.data
        q = np.asarray(d.qpos[spec.q_adrs], dtype=np.float64)
        return np.concatenate([q.astype(np.float32), self._target_pos.astype(np.float32)], axis=0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        _ = options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _, info = self.env.reset(seed=seed)
        self._refresh_obs_cache()
        spec = self._resolve_joint_spec()
        self._init_palm_reference()
        self._tip_site_id = None
        self._target_site_id = None

        self._reachable_points = self._sample_reachable_points(spec, self.target_rollout_samples)
        if self.randomize_start_pose:
            tip, tgt = self._randomize_start_pose(spec)
        else:
            tip = self._tip_pos()
            tgt = self._sample_target(tip, strict_distance_window=False)
            if tgt is None:
                tgt = np.asarray(tip, dtype=np.float64) + np.array([0.12, 0.0, -0.02], dtype=np.float64)
        self._target_pos = np.asarray(tgt, dtype=np.float64).reshape(3)
        self._apply_target_marker(self._target_pos)
        if self._camera_lookat is None or (not np.isfinite(np.asarray(self._camera_lookat, dtype=np.float64)).all()):
            self._camera_lookat = (0.5 * (tip + self._target_pos)).astype(np.float64)
        self._apply_background_visibility()
        self._prev_dist = float(np.linalg.norm(self._target_pos - tip))
        self._step_count = 0
        self._success_streak = 0
        self._success_bonus_given = False
        self._prev_action = np.zeros(spec.q_adrs.shape[0], dtype=np.float64)
        obs = self._build_obs(spec)

        out = dict(info)
        out.update(
            {
                "distance_error": float(self._prev_dist),
                "success": False,
                "joint_names": tuple(spec.names),
            }
        )
        return obs, out

    def step(self, action: np.ndarray):
        spec = self._resolve_joint_spec()
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != int(spec.q_adrs.shape[0]):
            raise ValueError(f"Action dim mismatch: got {a.shape[0]} expected {spec.q_adrs.shape[0]}")
        a = np.clip(a, -1.0, 1.0)

        sim = self.unwrapped.sim
        d = sim.data
        q_cur = np.asarray(d.qpos[spec.q_adrs], dtype=np.float64)
        if self.action_mode == "absolute":
            q_next = 0.5 * (a + 1.0) * (spec.q_hi - spec.q_lo) + spec.q_lo
        else:
            q_next = q_cur + self.action_max_delta_rad * a
        q_next = np.clip(q_next, spec.q_lo, spec.q_hi)
        d.qpos[spec.q_adrs] = q_next
        d.qvel[spec.dof_adrs] = 0.0
        if hasattr(d, "ctrl"):
            d.ctrl[:] = 0.0
        if hasattr(d, "act"):
            d.act[:] = 0.0
        sim.forward()
        self._refresh_obs_cache()

        tip = self._tip_pos()
        dist = float(np.linalg.norm(self._target_pos - tip))
        if self.reward_mode == "progress":
            reward = float(self._prev_dist - dist)
        else:
            reward = float(-dist)
        reward -= float(self.action_penalty * np.mean(a * a))
        palm_err = float(self._palm_alignment_error())
        near = float(np.clip(1.0 - dist / self.palm_near_target_scale, 0.0, 1.0))
        palm_weight_eff = float(self.palm_weight * near)
        reward -= float(palm_weight_eff * palm_err)

        reached = bool(dist <= self.success_radius)
        self._success_streak = (self._success_streak + 1) if reached else 0
        success = bool(self._success_streak >= self.success_steps_required)
        if success and (not self._success_bonus_given):
            reward += float(self.success_bonus)
            self._success_bonus_given = True

        self._step_count += 1
        terminated = bool(success and self.terminate_on_success)
        truncated = bool(self._step_count >= self.max_steps)

        self._prev_dist = dist
        self._prev_action = a.copy()
        obs = self._build_obs(spec)
        info = {
            "distance_error": float(dist),
            "success": bool(success),
            "is_success": bool(success),
            "hold_steps": int(self._success_streak),
            "hold_steps_required": int(self.success_steps_required),
            "palm_down_error": float(palm_err),
            "palm_weight_effective": float(palm_weight_eff),
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

    def close(self):
        self.env.close()
