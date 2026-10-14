"""ComfyUI-ClipWaltz — talking photo on your own GPU (ClipWaltz Phase C3).

One node, **ClipWaltz Talking Photo**: a portrait (IMAGE) + speech (AUDIO) -> video frames (IMAGE) + fps, with EchoMimic v3
Flash (antgroup/echomimic_v3 @ 7e89489, Apache-2.0, vendored in ./cwtalk) on Wan2.1-Fun-V1.1-1.3B-InP. Same engine and
settings as ClipWaltz's own GPU service (aiserver/talking/service.py): 8 steps, TeaCache, overlapping chunks with an
8-frame cross-fade so long clips fit 10 GB, model CPU offload. The text prompt is fixed, so its umT5 embeddings ship
precomputed (prompt_embeds.safetensors) and the 11 GB text encoder is never loaded.

Models (ComfyUI/models/clipwaltz_talking/ — see README / download_models.py):
  Wan2.1-Fun-V1.1-1.3B-InP/{config.json, diffusion_pytorch_model.safetensors, Wan2.1_VAE.pth,
                            models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}
  echomimicv3-flash-pro/diffusion_pytorch_model.safetensors
  chinese-wav2vec2-base/{config.json, preprocessor_config.json, pytorch_model.bin}
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict

import numpy as np
import torch

import comfy.model_management as mm
import folder_paths

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(folder_paths.models_dir, "clipwaltz_talking")
folder_paths.add_model_folder_path("clipwaltz_talking", MODEL_DIR)
BASE = os.path.join(MODEL_DIR, "Wan2.1-Fun-V1.1-1.3B-InP")
FPS = 25
OVERLAP = 8

REQUIRED = [
    os.path.join(BASE, "config.json"),
    os.path.join(BASE, "diffusion_pytorch_model.safetensors"),
    os.path.join(BASE, "Wan2.1_VAE.pth"),
    os.path.join(BASE, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    os.path.join(MODEL_DIR, "echomimicv3-flash-pro", "diffusion_pytorch_model.safetensors"),
    os.path.join(MODEL_DIR, "chinese-wav2vec2-base", "config.json"),
]

_M: Dict[str, Any] = {}


def _missing() -> list:
    return [p for p in REQUIRED if not os.path.isfile(p)]


def _load(device: torch.device) -> Dict[str, Any]:
    if _M.get("device") == device:
        return _M
    miss = _missing()
    if miss:
        raise RuntimeError("ClipWaltz talking photo: missing model files — run download_models.py. Missing: " + ", ".join(miss))
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from transformers import Wav2Vec2FeatureExtractor
    from .cwtalk.src.wan_vae import AutoencoderKLWan
    from .cwtalk.src.wan_image_encoder import CLIPModel
    from .cwtalk.src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
    from .cwtalk.src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
    from .cwtalk.src.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .cwtalk.src.utils import filter_kwargs
    from .cwtalk.src.cache_utils import get_teacache_coefficients
    from .cwtalk.src.wav2vec2 import Wav2Vec2Model

    dtype = torch.bfloat16 if mm.should_use_bf16(device) else torch.float16
    cfg = OmegaConf.load(os.path.join(HERE, "cwtalk", "config", "config.yaml"))
    w2v = os.path.join(MODEL_DIR, "chinese-wav2vec2-base")
    audio_encoder = Wav2Vec2Model.from_pretrained(w2v, local_files_only=True).to("cpu").eval()
    audio_encoder.feature_extractor._freeze_parameters()
    feat = Wav2Vec2FeatureExtractor.from_pretrained(w2v, local_files_only=True)
    transformer = WanTransformerAudioMask3DModel.from_pretrained(
        os.path.join(BASE, cfg["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(cfg["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=dtype,
    )
    sd = load_file(os.path.join(MODEL_DIR, "echomimicv3-flash-pro", "diffusion_pytorch_model.safetensors"))
    transformer.load_state_dict(sd.get("state_dict", sd), strict=False)
    del sd
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(BASE, cfg["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"]),
    ).to(dtype)
    clip = CLIPModel.from_pretrained(os.path.join(BASE, cfg["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))).to(dtype).eval()
    sched_kwargs = OmegaConf.to_container(cfg["scheduler_kwargs"])
    sched_kwargs["shift"] = 1  # Flow_Unipc, as in the official Flash script
    scheduler = FlowUniPCMultistepScheduler(**filter_kwargs(FlowUniPCMultistepScheduler, sched_kwargs))
    pipe = WanFunInpaintAudioPipeline(transformer=transformer, vae=vae, tokenizer=None, text_encoder=None,
                                      scheduler=scheduler, clip_image_encoder=clip)
    pipe.enable_model_cpu_offload(device=device)
    type(pipe)._execution_device = property(lambda self: device)

    from safetensors import safe_open
    with safe_open(os.path.join(HERE, "prompt_embeds.safetensors"), framework="pt") as f:
        embeds = {k: f.get_tensor(k) for k in f.keys()}
    _M.clear()
    _M.update(device=device, dtype=dtype, pipe=pipe, audio_encoder=audio_encoder, feat=feat, embeds=embeds,
              coeffs=get_teacache_coefficients(BASE))
    return _M


def _speech_16k(audio: Dict[str, Any], max_seconds: float) -> np.ndarray:
    """ComfyUI AUDIO {waveform [B, C, S], sample_rate} -> mono float32 at 16 kHz, loudness-normalised, capped."""
    import torchaudio.functional as AF
    wav = audio["waveform"][0].float()
    wav = wav.mean(dim=0) if wav.dim() == 2 else wav
    sr = int(audio["sample_rate"])
    if sr != 16000:
        wav = AF.resample(wav, sr, 16000)
    wav = wav[: int(max_seconds * 16000)].cpu().numpy()
    try:
        import pyloudnorm as pyln
        loud = pyln.Meter(16000).integrated_loudness(wav)
        if abs(loud) <= 100:
            wav = pyln.normalize.loudness(wav, loud, -23)
    except Exception:  # noqa: BLE001 — very short clips can't be measured; keep as is
        pass
    return wav.astype(np.float32)


def _audio_embeds(m: Dict[str, Any], wav16: np.ndarray, frames: int) -> torch.Tensor:
    from einops import rearrange
    feat = np.squeeze(m["feat"](wav16, sampling_rate=16000).input_values)
    feat = torch.from_numpy(feat).float().unsqueeze(0)
    enc = m["audio_encoder"]
    outs: list = []  # transformers 5.x no longer fills hidden_states for this forward: capture each layer's output
    hooks = [l.register_forward_hook(lambda mod, i, o: outs.append(o[0] if isinstance(o, tuple) else o)) for l in enc.encoder.layers]
    with torch.no_grad():
        e = enc(feat, seq_len=int(frames), output_hidden_states=True)
    for h in hooks:
        h.remove()
    hidden = e.hidden_states[1:] if e.hidden_states is not None else outs
    emb = rearrange(torch.stack(hidden, dim=1).squeeze(0), "b s d -> s b d").cpu().to(m["device"], m["dtype"])
    idx = torch.clamp(torch.arange(0, frames).unsqueeze(1) + (torch.arange(5) - 2).unsqueeze(0), min=0, max=emb.shape[0] - 1)
    return emb[idx].unsqueeze(0)  # [1, F, 5, 12, 768]


def _chunk_plan(total: int, chunk: int) -> list:
    plan, start = [], 0
    while True:
        remaining = total - start
        length = (min(chunk, remaining) - 1) // 4 * 4 + 1
        if plan and length <= OVERLAP:
            break
        plan.append((start, length))
        if remaining <= chunk:
            break
        start += length - OVERLAP
    return plan


def _crop(img, w: int, h: int):
    """Centre-crop to w:h (kept near the top for tall photos, where faces sit)."""
    iw, ih = img.size
    target = w / h
    if iw / ih > target:
        nw = int(ih * target)
        return img.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
    nh = int(iw / target)
    top = min((ih - nh) // 2, int(ih * 0.1))
    return img.crop((0, top, iw, top + nh))


class ClipWaltzTalkingPhoto:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "width": ("INT", {"default": 448, "min": 256, "max": 1024, "step": 16}),
                "height": ("INT", {"default": 768, "min": 256, "max": 1024, "step": 16}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 2**32 - 1}),
                "max_seconds": ("FLOAT", {"default": 30.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "chunk_frames": ([97, 81, 113, 65],),
                "steps": ("INT", {"default": 8, "min": 4, "max": 20}),
                # Model status, read by the ClipWaltz Connector from /object_info: only "ready" once every file exists.
                "models": (["ready"] if not _missing() else ["missing — run download_models.py"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "run"
    CATEGORY = "ClipWaltz"

    def run(self, image, audio, width, height, seed, max_seconds, chunk_frames, steps, models=None):
        from PIL import Image
        mm.unload_all_models()  # give this job the whole card
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        m = _load(device)
        from .cwtalk.src.utils import get_image_to_video_latent3

        wav16 = _speech_16k(audio, max_seconds)
        if len(wav16) < 8000:
            raise ValueError("the speech is too short (under half a second)")
        total = (int(len(wav16) / 16000 * FPS) - 1) // 4 * 4 + 1
        plan = _chunk_plan(total, int(chunk_frames))
        w, h = int(width) // 16 * 16, int(height) // 16 * 16
        photo = Image.fromarray((image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
        photo = _crop(photo, w, h)
        embeds = _audio_embeds(m, wav16, total)
        pipe = m["pipe"]
        gen = torch.Generator(device=device).manual_seed(int(seed))
        _, _, clip_image = get_image_to_video_latent3(photo, None, video_length=plan[0][1], sample_size=[h, w])
        pos = [m["embeds"]["prompt"].to(device, m["dtype"])]
        neg = [m["embeds"]["negative"].to(device, m["dtype"])]

        # One progress bar over all chunks (ComfyUI shows it; the Connector reads it from the websocket).
        from comfy.utils import ProgressBar
        pbar = ProgressBar(len(plan) * int(steps))

        def on_step(_p, _i, _t, kw):
            mm.throw_exception_if_processing_interrupted()  # ComfyUI's Cancel / the Connector's stop
            pbar.update(1)
            return kw

        try:
            mix = torch.linspace(0, 1, steps=OVERLAP).view(1, 1, -1, 1, 1)
            video, ref = None, photo
            for start, length in plan:
                if m["coeffs"] is not None:
                    pipe.transformer.enable_teacache(m["coeffs"], int(steps), 0.1, num_skip_start_steps=5, offload=False)
                inp, mask, _ = get_image_to_video_latent3(ref, None, video_length=length, sample_size=[h, w])
                with torch.no_grad():
                    sample = pipe(
                        None, prompt_embeds=pos, negative_prompt_embeds=neg,
                        num_frames=length, audio_embeds=embeds[:, start:start + length], audio_scale=1.0,
                        ip_mask=None, use_un_ip_mask=False, height=h, width=w, generator=gen, neg_scale=1.0, neg_steps=0,
                        use_dynamic_cfg=False, use_dynamic_acfg=False, guidance_scale=6.0, audio_guidance_scale=3.0,
                        num_inference_steps=int(steps), video=inp, mask_video=mask, clip_image=clip_image,
                        cfg_skip_ratio=0.0, shift=5.0, callback_on_step_end=on_step, comfyui_progressbar=False,
                    ).videos
                if video is None:
                    video = sample
                else:
                    video[:, :, -OVERLAP:] = video[:, :, -OVERLAP:] * (1 - mix) + sample[:, :, :OVERLAP] * mix
                    video = torch.cat([video, sample[:, :, OVERLAP:]], dim=2)
                ref = [Image.fromarray((video[0, :, i].permute(1, 2, 0) * 255).numpy().astype(np.uint8)) for i in range(-OVERLAP, 0)]
                del inp, mask
                mm.soft_empty_cache()
        finally:
            # Always give the card back — also after a Cancel or an error mid-render: ComfyUI's model manager doesn't
            # know these models: after a cancelled render the next Wan job ran at 233 s/step instead of 26 (2026-10-14).
            try:
                pipe.transformer.disable_teacache()
            except Exception:  # noqa: BLE001
                pass
            pipe.maybe_free_model_hooks()
            del embeds, pos, neg
            mm.soft_empty_cache()
        frames = video[0].permute(1, 2, 3, 0).float().clamp(0, 1)  # [F, H, W, C] for ComfyUI
        return (frames, float(FPS))


NODE_CLASS_MAPPINGS = {"ClipWaltzTalkingPhoto": ClipWaltzTalkingPhoto}
NODE_DISPLAY_NAME_MAPPINGS = {"ClipWaltzTalkingPhoto": "ClipWaltz Talking Photo"}
