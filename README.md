# ComfyUI-ClipWaltz — talking photo on your own GPU

Lets the **ClipWaltz Connector** render ClipWaltz *Talking photo* videos on your NVIDIA GPU for free. Adds one ComfyUI
node, **ClipWaltz Talking Photo** (portrait + speech → video), built on EchoMimic v3 Flash — the same engine ClipWaltz
runs on its own GPUs.

**Needs:** an NVIDIA GPU with 8 GB+ (10 GB recommended), ComfyUI (Windows portable or any install), ~12 GB of disk.

## Install (Windows portable)

1. In `ComfyUI\custom_nodes`, run `git clone https://github.com/phoenixtekk/ComfyUI-ClipWaltz comfyui-clipwaltz`
   (no git? download the zip from www.clipwaltz.com/connector/comfyui-clipwaltz.zip and unzip it there).
2. In the ComfyUI_windows_portable folder, run:
   ```
   python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-clipwaltz\requirements.txt
   python_embeded\python.exe ComfyUI\custom_nodes\comfyui-clipwaltz\download_models.py
   ```
3. Restart ComfyUI. Start the ClipWaltz Connector — it reports "Talking photo … ready".

Update later with `git pull` in `custom_nodes\comfyui-clipwaltz`, then restart ComfyUI.

Other installs: same steps with the Python that runs ComfyUI.

## How it renders

8-step EchoMimic v3 Flash with TeaCache, in overlapping ~4 s chunks (8-frame cross-fade) so clips up to 30 s fit a
10 GB card, with model CPU offload. The text prompt is fixed, so its embeddings are included (`prompt_embeds.safetensors`)
and the 11 GB text encoder is not needed.

## Licences

- This pack: Apache-2.0 (see `LICENSE`).
- `cwtalk/` — EchoMimic v3 source (github.com/antgroup/echomimic_v3 @ `7e89489`), Apache-2.0, see `cwtalk/LICENSE.txt`.
  Changes: imports made package-relative; the pipeline takes its dtype from the transformer and accepts a list of
  precomputed prompt embeddings (no text encoder loaded).
- Models (downloaded by you from Hugging Face): Wan2.1-Fun-V1.1-1.3B-InP (Apache-2.0), EchoMimic v3 Flash (Apache-2.0),
  chinese-wav2vec2-base (MIT).
