# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Drop-in PolicyServer compatible with Isaac-GR00T's ZMQ REQ/REP protocol, backed by
a fine-tuned starVLA model (QwenGR00T / QwenPI / QwenOFT — auto-detected from the
checkpoint config).

Why this file exists
--------------------
starVLA ships its OWN websocket policy server (deployment/model_server/server_policy.py,
port 10093, `predict_action`). That is what LIBERO/RoboTwin/SimplerEnv eval clients speak.
But soma's whole-body deployment stack (GR00T-WholeBodyControl gear_sonic) drives BOTH the
real G1 robot AND the tele_humanoid_benchmark sim through ONE contract: a gr00t-compatible
policy server on `tcp://0.0.0.0:5550` answering `get_action`, which the model-agnostic client
`gear_sonic/scripts/run_vla_inference.py` (Isaac-GR00T PolicyClient, ZMQ REQ/REP) connects to.
gr00t serves it natively; fastwam and lingbot-va each wrap their model into it
(`fastwam_deploy/fastwam_gr00t_server.py`, `wan_va/gr00t_compat_server.py`). This file is the
starVLA equivalent, so the three starVLA frameworks deploy on the real robot / sim exactly like
the other three models — no client-side change.

What it does per get_action()
------------------------------
  1. Build a starVLA `example` = {image: [ego_view frame], lang: prompt, state: raw 43-D q}
     — same shape the RoboCasa/LIBERO eval interfaces build (model2*_interface.py). The G1
     checkpoint's action-head state_encoder is MLP(input_dim=state_dim=43), so raw 43-D state
     is the default (--state-mode; sincos->86-D and none are also available).
  2. model.predict_action([example]) -> {"normalized_actions": [1, T, 78]}  (min-max normalized).
  3. Un-normalize with the checkpoint's dataset_statistics.json (action min/max/mask), exactly
     like the eval interfaces' unnormalize_actions() — NO gripper binarization (G1 is not Franka).
  4. Split the 78-D action into gr00t/fastwam/lingbot parity keys and return them:
        action.motion_token      [H_out, 64]
        action.left_hand_joints  [H_out, 7]
        action.right_hand_joints [H_out, 7]

Model auto-detection: read_mode_config() reads <run>/config.yaml + dataset_statistics.json from
the checkpoint dir; build_framework() instantiates the framework named in that config. So the
SAME command serves a QwenGR00T, QwenPI or QwenOFT checkpoint.

Dependencies (add to the starVLA env; the native websocket server doesn't need them):
    pip install pyzmq msgpack-numpy

Usage (single GPU, on a host with the checkpoint):
    python -m deployment.model_server.starvla_gr00t_server \
        --ckpt-path /path/to/<RUN_ID>/checkpoints/steps_20000_pytorch_model.pt \
        --initial-prompt "test" --host 0.0.0.0 --port 5550

Wire protocol (matches gr00t.policy.server_client.PolicyServer / fastwam_gr00t_server.py):
    Transport : ZMQ REP socket on tcp://<host>:<port>
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except (AttributeError, ValueError):
    pass


# ── 43-D G1 proprio layout (dataset meta/modality.json / UnitreeG1SonicDataConfig) ─
# state_keys order: left_leg | right_leg | waist | left_arm | left_hand | right_arm | right_hand
PROPRIO_DIM = 43
STATE_GROUPS = [
    ("left_leg", slice(0, 6)),
    ("right_leg", slice(6, 12)),
    ("waist", slice(12, 15)),
    ("left_arm", slice(15, 22)),
    ("left_hand", slice(22, 29)),
    ("right_arm", slice(29, 36)),
    ("right_hand", slice(36, 43)),
]
LEFT_HAND_SLICE = slice(22, 29)
RIGHT_HAND_SLICE = slice(36, 43)
HAND_DIM = 7
MOTION_TOKEN_DIM = 64
ACTION_DIM = 78  # motion_token(64) + left_hand(7) + right_hand(7)


# ── starVLA policy adapter ────────────────────────────────────────────────────
class StarVLAPolicy:
    """Wraps a loaded starVLA framework so it answers Isaac-GR00T get_action() calls."""

    def __init__(
        self,
        model,
        *,
        action_norm_stats: dict,
        image_size: Optional[Any],
        state_mode: str,
        upsample_factor: int,
        upsample_mode: str,
        initial_prompt: Optional[str],
        predict_kwargs: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.action_norm_stats = action_norm_stats
        self.image_size = image_size
        self.state_mode = str(state_mode)
        self.upsample_factor = max(1, int(upsample_factor))
        self.upsample_mode = str(upsample_mode)
        self.predict_kwargs = dict(predict_kwargs or {})
        self._current_prompt: Optional[str] = initial_prompt

        # pre-parse action min/max/mask once (support both min/max and q01/q99 layouts)
        st = action_norm_stats
        lo = st.get("min", st.get("q01"))
        hi = st.get("max", st.get("q99"))
        if lo is None or hi is None:
            raise ValueError(
                f"action norm stats need min/max (or q01/q99); got keys {list(st.keys())}")
        self._act_low = np.asarray(lo, dtype=np.float32)
        self._act_high = np.asarray(hi, dtype=np.float32)
        self._act_mask = (np.asarray(st["mask"], dtype=bool)
                          if "mask" in st else np.ones_like(self._act_low, dtype=bool))
        if self._act_low.shape[0] != ACTION_DIM:
            print(f"[starvla] WARN: action stats dim={self._act_low.shape[0]} != {ACTION_DIM}; "
                  f"un-normalization assumes the first {ACTION_DIM} dims are "
                  f"motion_token(64)+hands(14).", flush=True)

    # ---- public endpoints --------------------------------------------------

    def ping(self) -> dict:
        return {"status": "ok", "message": "starvla gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self._current_prompt = prompt
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        t0 = time.monotonic()

        prompt = self._extract_prompt(observation) or self._current_prompt
        if prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt at startup, call reset(prompt=...), "
                "or include obs['language']['annotation.human.task_description'][0][0].")
        self._current_prompt = prompt

        frame = self._extract_image(observation)          # [H,W,3] uint8 (newest)
        example: dict[str, Any] = {"image": [frame], "lang": prompt}
        raw_proprio = self._extract_proprio(observation)  # [43] raw joint angles (may be None)
        if raw_proprio is not None and self.state_mode != "none":
            # The checkpoint's action head state_encoder is MLP(input_dim=config.state_dim);
            # G1 config resolves state_dim=43 (raw joints), so "raw" is the default. "sincos"
            # (per-group sin/cos -> 86-D) is kept for checkpoints trained with that transform.
            example["state"] = (self._sincos_state(raw_proprio)[None]      # [1, 86]
                                if self.state_mode == "sincos"
                                else raw_proprio.astype(np.float32)[None])  # [1, 43] raw

        t_inf = time.monotonic()
        with torch.no_grad():
            out = self.model.predict_action([example], **self.predict_kwargs)
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        norm_actions = np.asarray(out["normalized_actions"])          # [B,T,78] or [T,78]
        if norm_actions.ndim == 3:
            norm_actions = norm_actions[0]                            # [T,78]
        raw = self._unnormalize(norm_actions)                        # [T,78] real

        motion_token = self._upsample(raw[:, :MOTION_TOKEN_DIM])            # [H_out,64]
        left_hand = self._upsample(raw[:, MOTION_TOKEN_DIM:MOTION_TOKEN_DIM + HAND_DIM])
        right_hand = self._upsample(raw[:, MOTION_TOKEN_DIM + HAND_DIM:ACTION_DIM])
        h_out = motion_token.shape[0]

        print(
            f"[starvla] get_action: T={raw.shape[0]} -> H_out={h_out} "
            f"(x{self.upsample_factor} {self.upsample_mode}); infer={infer_ms:.0f} ms; "
            f"state={self.state_mode}; prompt={self._current_prompt!r}",
            flush=True,
        )

        action_dict = {
            "action.motion_token": np.ascontiguousarray(motion_token, dtype=np.float32),
            "action.left_hand_joints": np.ascontiguousarray(left_hand, dtype=np.float32),
            "action.right_hand_joints": np.ascontiguousarray(right_hand, dtype=np.float32),
        }
        info = {"inference_time_s": time.monotonic() - t0, "chunk_horizon": int(h_out)}
        return action_dict, info

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        lang = obs.get("language") or {}
        raw = lang.get("annotation.human.task_description")
        if raw is None:
            return None
        while isinstance(raw, (list, tuple)) and len(raw) > 0:
            if isinstance(raw[0], (list, tuple)):
                raw = raw[0]
            else:
                break
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw if isinstance(raw, str) else None

    def _extract_image(self, obs: dict) -> np.ndarray:
        """Return the newest ego_view frame as [H,W,3] uint8 (predict_action resizes it)."""
        video = obs.get("video") or {}
        img = None
        for k in ("ego_view", "observation.images.ego_view"):
            if k in video:
                img = video[k]
                break
        if img is None:
            if len(video) == 1:
                img = next(iter(video.values()))
            else:
                raise KeyError(f"No ego_view in obs['video']; keys={list(video.keys())}")
        img = np.asarray(img)
        while img.ndim > 3 and img.shape[0] >= 1:  # peel [B,T,..] down to the newest frame
            img = img[-1]
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"ego_view frame must be [H,W,3], got {img.shape}")
        return img.astype(np.uint8)

    def _extract_proprio(self, obs: dict) -> Optional[np.ndarray]:
        q = obs.get("q")
        if q is not None:
            q = np.asarray(q, dtype=np.float32).reshape(-1)
            if q.shape[0] != PROPRIO_DIM:
                raise ValueError(f"obs['q'] must be {PROPRIO_DIM}-dim, got {q.shape[0]}")
            return q
        state = obs.get("state") or {}
        order = ["left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand"]
        if not all(g in state for g in order):
            return None  # no proprio available -> state_mode handling decides
        parts = [np.asarray(state[g], dtype=np.float32).reshape(-1) for g in order]
        q = np.concatenate(parts)
        if q.shape[0] != PROPRIO_DIM:
            raise ValueError(f"Reconstructed proprio is {q.shape[0]}-dim, expected {PROPRIO_DIM}")
        return q

    @staticmethod
    def _sincos_state(raw_proprio: np.ndarray) -> np.ndarray:
        """Per-group [sin(joints), cos(joints)] then concat, matching
        UnitreeG1SonicDataConfig.transform() (StateActionSinCosTransform) and the
        RoboCasa interface's normalize_state(). 43 raw joints -> 86-D."""
        parts = []
        for _name, sl in STATE_GROUPS:
            g = raw_proprio[sl]
            parts.append(np.concatenate([np.sin(g), np.cos(g)]))
        return np.concatenate(parts).astype(np.float32)  # [86]

    def _unnormalize(self, norm_actions: np.ndarray) -> np.ndarray:
        """Inverse of min-max normalization: action = 0.5*(x+1)*(max-min)+min for masked dims.
        Mirrors the eval interfaces' unnormalize_actions() (no gripper binarization for G1)."""
        x = np.clip(np.asarray(norm_actions, dtype=np.float32), -1.0, 1.0)
        d = min(x.shape[-1], self._act_low.shape[0])
        out = x.copy()
        lin = 0.5 * (x[:, :d] + 1.0) * (self._act_high[:d] - self._act_low[:d]) + self._act_low[:d]
        out[:, :d] = np.where(self._act_mask[:d], lin, x[:, :d])
        return out

    def _upsample(self, tokens: np.ndarray) -> np.ndarray:
        f = self.upsample_factor
        if f <= 1:
            return tokens
        if self.upsample_mode == "repeat":
            return np.repeat(tokens, f, axis=0)
        T, D = tokens.shape
        src = np.arange(T, dtype=np.float64)
        tgt = np.clip(np.arange(T * f, dtype=np.float64) / f, 0, T - 1)
        out = np.empty((T * f, D), dtype=np.float32)
        for d in range(D):
            out[:, d] = np.interp(tgt, src, tokens[:, d])
        return out


# ── ZMQ server (mirrors gr00t.policy.server_client.PolicyServer wire format) ─
def serve(policy: StarVLAPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[starvla] PolicyServer ready on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                running = False
                resp = {"status": "killing"}
            elif endpoint == "get_action":
                action, info = policy.get_action(
                    observation=data.get("observation", {}),
                    options=data.get("options"),
                )
                resp = [action, info]
            elif endpoint == "reset":
                resp = policy.reset(options=data.get("options"))
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            else:
                resp = {"error": f"Unknown endpoint: {endpoint}"}

            sock.send(mnp.packb(resp))
        except zmq.error.ContextTerminated:
            break
        except Exception as e:  # noqa: BLE001
            print(f"[starvla] server error: {e}\n{traceback.format_exc()}", flush=True)
            try:
                sock.send(mnp.packb({"error": str(e)}))
            except Exception:
                pass

    try:
        sock.close(linger=0)
    finally:
        ctx.term()


# ── Model loading (mirrors baseframework.from_pretrained: read cfg+stats, build) ─
def _select_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> str:
    if unnorm_key is None:
        if len(norm_stats) != 1:
            raise SystemExit(
                f"--unnorm-key is required: dataset_statistics.json has multiple keys "
                f"{list(norm_stats.keys())}")
        return next(iter(norm_stats.keys()))
    if unnorm_key not in norm_stats:
        raise SystemExit(f"--unnorm-key {unnorm_key!r} not in {list(norm_stats.keys())}")
    return unnorm_key


def load_model(args):
    from starVLA.model.framework.base_framework import baseframework

    ckpt = str(Path(args.ckpt_path).expanduser().resolve())
    # from_pretrained internally read_mode_config()s config.yaml + dataset_statistics.json and
    # build_framework()s the framework named in the config -> auto QwenGR00T/QwenPI/QwenOFT.
    model = baseframework.from_pretrained(ckpt)
    if args.use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(args.device).eval()

    # Re-read stats/config for un-normalization + image size (cheap; same files).
    try:
        from starVLA.model.framework.share_tools import read_mode_config
    except Exception:  # pragma: no cover
        from starVLA.model.tools import read_mode_config
    cfg, norm_stats = read_mode_config(ckpt)
    unnorm_key = _select_unnorm_key(norm_stats, args.unnorm_key)
    action_stats = norm_stats[unnorm_key]["action"]
    # image size: config.datasets.vla_data.image_size (may be absent -> predict_action no-op)
    image_size = (((cfg or {}).get("datasets", {}) or {}).get("vla_data", {}) or {}).get("image_size")
    fw = (((cfg or {}).get("framework", {}) or {}).get("name")) if isinstance(cfg, dict) else None
    print(f"[starvla] loaded framework={fw} unnorm_key={unnorm_key!r} "
          f"action_stats_dim={len(action_stats.get('min', action_stats.get('q01', [])))} "
          f"image_size={image_size}", flush=True)
    return model, action_stats, image_size


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="starVLA policy server (Gr00t-compatible, :5550/get_action).")
    p.add_argument("--ckpt-path", required=True,
                   help="<RUN_ID>/checkpoints/steps_<N>_pytorch_model.pt "
                        "(its run dir must hold config.yaml + dataset_statistics.json)")
    p.add_argument("--starvla-repo", default=None,
                   help="Path to the starVLA repo root (prepended to sys.path); "
                        "default: infer from this file / $STARVLA_REPO")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--initial-prompt", default=None,
                   help="Serve this prompt when obs carries no language (G1 training prompt is 'test')")
    p.add_argument("--unnorm-key", default=None,
                   help="Embodiment key in dataset_statistics.json (auto if there is exactly one)")
    p.add_argument("--state-mode", default="raw", choices=["raw", "sincos", "none"],
                   help="raw (default): feed the raw 43-D proprio (matches the G1 checkpoint's "
                        "state_encoder input_dim=state_dim=43); sincos: per-group sin/cos -> 86-D "
                        "(only if the checkpoint was trained that way); none: omit state.")
    # starVLA G1 predicts a 16-step chunk (action_indices=range(16)). Keep 1 unless the C++
    # 50Hz controller needs upsampling (see fastwam_gr00t_server for the 10Hz->50Hz case).
    p.add_argument("--upsample-factor", type=int, default=1)
    p.add_argument("--upsample-mode", default="interp", choices=["interp", "repeat"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # put the starVLA repo root on sys.path (deployment.* / starVLA.* imports)
    repo = args.starvla_repo or os.environ.get("STARVLA_REPO")
    if repo is None:
        # this file is <repo>/deployment/model_server/starvla_gr00t_server.py
        repo = str(Path(__file__).resolve().parents[2])
    repo = str(Path(repo).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    model, action_stats, image_size = load_model(args)
    policy = StarVLAPolicy(
        model,
        action_norm_stats=action_stats,
        image_size=image_size,
        state_mode=args.state_mode,
        upsample_factor=args.upsample_factor,
        upsample_mode=args.upsample_mode,
        initial_prompt=args.initial_prompt,
    )
    print(
        f"[starvla] ready: port={args.port} state={args.state_mode} "
        f"x{args.upsample_factor}({args.upsample_mode}); prompt={args.initial_prompt!r}",
        flush=True,
    )
    try:
        serve(policy, args.host, args.port)
    except KeyboardInterrupt:
        print("[starvla] interrupted, shutting down.", flush=True)


if __name__ == "__main__":
    main()
