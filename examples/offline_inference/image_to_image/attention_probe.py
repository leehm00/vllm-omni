# SPDX-License-Identifier: Apache-2.0
"""
Lightweight monkey-patch utilities to probe cross-attention maps and intermediate
latents inside vllm-omni diffusion pipelines (Qwen-Image / Flux-like).

Usage sketch
------------
```python
from attention_probe import AttentionProbe, install_probe
from vllm_omni.entrypoints.omni import Omni

probe = AttentionProbe(target_tokens=[5, 6], save_interval=5)
omni = Omni(model="/data/models/Qwen-Image-Edit")
install_probe(omni.diffusion_worker.pipeline, probe)  # patch in-place

images = omni.generate(prompt="a red car")
probe.save_step_grids(omni.diffusion_worker.pipeline, out_dir="./probe_out")
# probe.save_video("./probe_out.mp4")  # optional if imageio installed
```

Notes
-----
- Non-invasive: nothing is written to site-packages; all changes are runtime monkey patches.
- Memory safe: tensors captured are immediately moved to CPU (`detach().float().cpu()`).
- Configurable: `save_interval` controls which timesteps are saved.
"""

from __future__ import annotations

import math
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

try:  # Matplotlib is only needed for heatmap coloring
    import matplotlib.cm as cm
except Exception:  # pragma: no cover - optional dependency
    cm = None

try:  # Optional for mp4 writing
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency
    imageio = None

# -------- Probe state containers --------


@dataclass
class AttentionRecord:
    step: int
    timestep: float
    layer: str
    token_index: int
    attn_map: torch.Tensor  # shape: [batch, latent_seq] on CPU
    latent_h: int
    latent_w: int


@dataclass
class LatentRecord:
    step: int
    timestep: float
    latents: torch.Tensor  # packed latents on CPU, shape [B, seq, C]
    height: int
    width: int
    vae_scale_factor: int


# -------- Main probe --------


class AttentionProbe:
    def __init__(
        self,
        target_tokens: Iterable[int],
        save_interval: int = 1,
        head_reduction: str = "mean",  # mean|max for heads aggregation
    ):
        self.target_tokens = list(target_tokens)
        self.save_interval = max(1, int(save_interval))
        self.head_reduction = head_reduction

        self._current_step: Optional[int] = None
        self._current_timestep: Optional[float] = None
        self.attn_records: List[AttentionRecord] = []
        self.latent_records: List[LatentRecord] = []

    # ---- step bookkeeping ----
    def set_step(self, step: int, timestep: float, latent_shape: tuple[int, int]):
        self._current_step = step
        self._current_timestep = float(timestep)
        self._latent_h, self._latent_w = latent_shape

    def clear(self):
        self.attn_records.clear()
        self.latent_records.clear()

    # ---- hooks ----
    def make_attention_hook(self, module_name: str):
        def _hook(module, inputs, _output):
            if self._current_step is None:
                return
            if self._current_step % self.save_interval != 0:
                return
            if not inputs:
                return

            query, key, *_ = inputs
            if query is None or key is None:
                return
            seq_len_txt = getattr(module, "_probe_seq_len_txt", None)
            if seq_len_txt is None:
                return

            # Shapes: [B, S, H, D]
            if query.dim() != 4 or key.dim() != 4:
                return

            q_txt = query[:, :seq_len_txt]  # [B, Stxt, H, D]
            k_img = key[:, seq_len_txt:]  # [B, Simg, H, D]
            if q_txt.numel() == 0 or k_img.numel() == 0:
                return

            scale = getattr(module, "softmax_scale", None)
            if scale is None:
                scale = q_txt.shape[-1] ** -0.5

            # attn_scores: [B, H, Stxt, Simg]
            attn_scores = torch.einsum("bqhd,bkhd->bhqk", q_txt, k_img) * scale
            attn_probs = attn_scores.softmax(dim=-1)

            head_reduce = attn_probs.mean(dim=1) if self.head_reduction == "mean" else attn_probs.max(dim=1).values
            for tok in self.target_tokens:
                if tok < 0 or tok >= head_reduce.shape[1]:
                    continue
                attn_slice = head_reduce[:, tok]  # [B, Simg]
                self.attn_records.append(
                    AttentionRecord(
                        step=self._current_step,
                        timestep=self._current_timestep or 0.0,
                        layer=module_name,
                        token_index=tok,
                        attn_map=attn_slice.detach().float().cpu(),
                        latent_h=self._latent_h,
                        latent_w=self._latent_w,
                    )
                )

        return _hook

    def capture_latent(self, latents: torch.Tensor, height: int, width: int, vae_scale_factor: int):
        if self._current_step is None:
            return
        if self._current_step % self.save_interval != 0:
            return
        self.latent_records.append(
            LatentRecord(
                step=self._current_step,
                timestep=self._current_timestep or 0.0,
                latents=latents.detach().float().cpu(),
                height=int(height),
                width=int(width),
                vae_scale_factor=int(vae_scale_factor),
            )
        )

    # ---- visualization ----
    def _decode_latent_record(self, pipeline, record: LatentRecord, device: torch.device):
        """Decode a single latent record into a PIL image using the pipeline VAE."""
        latents = record.latents.to(device)
        # Unpack to 5D latents
        unpacked = pipeline._unpack_latents(
            latents,
            record.height,
            record.width,
            record.vae_scale_factor,
        )
        latents_mean = torch.tensor(pipeline.vae.config.latents_mean, device=device, dtype=unpacked.dtype).view(
            1, pipeline.latent_channels, 1, 1, 1
        )
        latents_std = torch.tensor(pipeline.vae.config.latents_std, device=device, dtype=unpacked.dtype).view(
            1, pipeline.latent_channels, 1, 1, 1
        )
        latents = unpacked * latents_std + latents_mean
        decoded = pipeline.vae.decode(latents).sample
        decoded = decoded.float().clamp(-1, 1)
        images = pipeline.image_processor.postprocess(decoded)
        return [img.convert("RGB") for img in images]

    def _overlay_heatmap(self, base: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
        if cm is None:
            # Simple gray overlay fallback
            heat_rgb = np.clip(heatmap, 0, 1)
            heat_rgb = np.uint8(heat_rgb * 255)
            heat_img = Image.fromarray(np.stack([heat_rgb] * 3, axis=-1)).resize(base.size, Image.BILINEAR)
        else:
            cmap = cm.get_cmap("magma")
            colored = (cmap(np.clip(heatmap, 0, 1))[:, :, :3] * 255).astype(np.uint8)
            heat_img = Image.fromarray(colored).resize(base.size, Image.BILINEAR)
        return Image.blend(base, heat_img, alpha=alpha)

    def _make_grid(self, images: List[Image.Image]) -> Image.Image:
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        grid = Image.new("RGB", (sum(widths), max(heights)), color=(0, 0, 0))
        x = 0
        for img in images:
            grid.paste(img, (x, 0))
            x += img.width
        return grid

    def _collect_step_attn(self, step: int) -> List[AttentionRecord]:
        return [r for r in self.attn_records if r.step == step]

    def save_step_grids(
        self,
        pipeline,
        out_dir: str | os.PathLike,
        token_labels: Optional[Dict[int, str]] = None,
        alpha: float = 0.45,
    ) -> List[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        device = pipeline.device if hasattr(pipeline, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        written: List[Path] = []
        for record in self.latent_records:
            imgs = self._decode_latent_record(pipeline, record, device)
            if not imgs:
                continue
            base = imgs[0]
            attn_for_step = self._collect_step_attn(record.step)
            overlays = []
            for attn in attn_for_step:
                heat = attn.attn_map.mean(dim=0).view(attn.latent_h, attn.latent_w).numpy()
                heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
                overlays.append(
                    self._overlay_heatmap(
                        base,
                        heat,
                        alpha=alpha,
                    )
                )
            labels = ["image"] + [f"tok{r.token_index}" if token_labels is None else token_labels.get(r.token_index, f"tok{r.token_index}") for r in attn_for_step]
            grid_imgs = [base] + overlays
            grid = self._make_grid(grid_imgs)
            out_path = out_dir / f"step_{record.step:03d}_t{int(record.timestep)}.png"
            grid.save(out_path)
            written.append(out_path)
        return written

    def save_video(self, video_path: str | os.PathLike, fps: int = 4):
        if imageio is None:
            raise RuntimeError("imageio is required to save video; install with `pip install imageio`." )
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for record in sorted(self.latent_records, key=lambda r: r.step):
            matching = [r for r in self.attn_records if r.step == record.step]
            if not matching:
                continue
            # Use stored grids if already saved; otherwise, render minimal overlay stack
            base_grid = None
            # We rely on PNGs if user called save_step_grids; otherwise, build a quick grid
            # Fallback: skip if no saved grids
            png_candidate = video_path.parent / f"step_{record.step:03d}_t{int(record.timestep)}.png"
            if png_candidate.exists():
                base_grid = Image.open(png_candidate)
            if base_grid is None:
                continue
            frames.append(np.array(base_grid))
        if not frames:
            raise RuntimeError("No frames to write. Run save_step_grids first or ensure probes captured data.")
        imageio.mimwrite(video_path, frames, fps=fps)
        return video_path


# -------- Monkey patch helpers --------


def _patch_cross_attention_modules(pipeline, probe: AttentionProbe):
    """Attach forward hooks to all QwenImageCrossAttention modules."""
    try:
        from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageCrossAttention
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("QwenImageCrossAttention not found; ensure vllm-omni is installed.") from exc

    handles = []
    for name, module in pipeline.transformer.named_modules():
        if isinstance(module, QwenImageCrossAttention):
            # Patch forward to stash seq_len_txt for the hook
            orig_forward = module.forward

            def wrapped_forward(self, hidden_states, encoder_hidden_states, vid_freqs, txt_freqs):
                self.attn._probe_seq_len_txt = encoder_hidden_states.shape[1]
                return orig_forward(hidden_states, encoder_hidden_states, vid_freqs, txt_freqs)

            module.forward = types.MethodType(wrapped_forward, module)
            handles.append(module.attn.register_forward_hook(probe.make_attention_hook(name)))
    return handles


def _patch_diffuse(pipeline, probe: AttentionProbe):
    """Wrap pipeline.diffuse to capture latents every save_interval steps."""
    orig_diffuse = pipeline.diffuse

    def wrapped_diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        image_latents,
        img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        guidance,
        true_cfg_scale,
    ):
        self.scheduler.set_begin_index(0)
        vae_sf = self.vae_scale_factor if hasattr(self, "vae_scale_factor") else 8
        # Recover latent grid shape from img_shapes (first element)
        # img_shapes: List[[ (frame, h, w), ... ]]
        latent_h = img_shapes[0][0][1]
        latent_w = img_shapes[0][0][2]
        full_h = latent_h * vae_sf * 2
        full_w = latent_w * vae_sf * 2

        for i, t in enumerate(timesteps):
            if getattr(self, "interrupt", False):
                break
            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            transformer_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": prompt_embeds_mask,
                "encoder_hidden_states": prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                "attention_kwargs": self.attention_kwargs,
                "return_dict": False,
            }
            if self._cache_backend is not None:
                transformer_kwargs.update({"cache_branch": "positive"})
            probe.set_step(i, float(t.item()), (latent_h, latent_w))

            noise_pred = self.transformer(**transformer_kwargs)[0]
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                transformer_kwargs.update({
                    "encoder_hidden_states": negative_prompt_embeds,
                    "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                })
                if self._cache_backend is not None:
                    transformer_kwargs.update({"cache_branch": "negative"})
                noise_pred_neg = self.transformer(**transformer_kwargs)[0]
                noise_pred_neg = noise_pred_neg[:, : latents.size(1)]
                noise_pred = noise_pred + true_cfg_scale * (noise_pred - noise_pred_neg)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            probe.capture_latent(latents, full_h, full_w, vae_sf)

        return latents

    pipeline.diffuse = types.MethodType(wrapped_diffuse, pipeline)


def install_probe(pipeline, probe: AttentionProbe):
    """Apply all monkey patches to a QwenImageEdit-like pipeline.

    Returns a list of hook handles so the caller can optionally remove them.
    """
    handles = _patch_cross_attention_modules(pipeline, probe)
    _patch_diffuse(pipeline, probe)
    return handles


__all__ = ["AttentionProbe", "install_probe"]
