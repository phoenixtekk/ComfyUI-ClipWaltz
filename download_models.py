"""Download the ClipWaltz talking-photo models (~12 GB) into ComfyUI/models/clipwaltz_talking.

Run with ComfyUI's own Python from the ComfyUI folder, e.g. (Windows portable):
    python_embeded\\python.exe ComfyUI\\custom_nodes\\comfyui-clipwaltz\\download_models.py
Every file comes from a pinned Hugging Face commit and is checked against its known hash before it is used; files
already present are checked too (and replaced if they don't match). All models are Apache-2.0 (Wan2.1-Fun, EchoMimic v3)
or MIT (chinese-wav2vec2-base).
"""
import hashlib
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = os.path.abspath(os.path.join(HERE, "..", ".."))
DEST = os.path.join(COMFY, "models", "clipwaltz_talking")

WAN = ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "fc913c34361f4ec879e2f9c78b4f11ae50a937d1")
EMV3 = ("BadToBest/EchoMimicV3", "311e176905a8c4c24b240b530488fe636ce4d249")
W2V = ("TencentGameMate/chinese-wav2vec2-base", "3991242c806928916fff4a8c0e4f76acf661b743")

# (repo, revision, file, folder, kind, hash) — kind "sha256" for LFS files, "git" (blob sha1) for small text files.
FILES = [
    (*WAN, "config.json", "Wan2.1-Fun-V1.1-1.3B-InP", "git", "8421aa8010ab06da0dbaf4a244490ca90db2b664"),
    (*WAN, "diffusion_pytorch_model.safetensors", "Wan2.1-Fun-V1.1-1.3B-InP", "sha256", "4ec199076538b946935ebcb3ba808d3c427e638f29519a3c3c98d31d821e5eed"),
    (*WAN, "Wan2.1_VAE.pth", "Wan2.1-Fun-V1.1-1.3B-InP", "sha256", "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"),
    (*WAN, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "Wan2.1-Fun-V1.1-1.3B-InP", "sha256", "628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38"),
    (*EMV3, "echomimicv3-flash-pro/diffusion_pytorch_model.safetensors", "", "sha256", "5ebdbb2fc709108bf2a1728fd92eb2874804e4bc0324e92a2cd55425968c85a4"),
    (*W2V, "config.json", "chinese-wav2vec2-base", "git", "74fe6760e497a3ce7647338c725395d7b2f7e63a"),
    (*W2V, "preprocessor_config.json", "chinese-wav2vec2-base", "git", "7d03dfc4ecba260febb718eb2ed6bb0d2da4f0f9"),
    (*W2V, "pytorch_model.bin", "chinese-wav2vec2-base", "sha256", "be2da40c9e7ae26bfc904a3ed79ebb9e8f060bec6dba85d6a6ae86114bc38901"),
]


def digest(path: str, kind: str) -> str:
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git":  # git blob id = sha1("blob <size>\0" + content)
        h.update(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    for repo, rev, name, sub, kind, want in FILES:
        target = os.path.join(DEST, sub, name)
        rel = os.path.relpath(target, DEST)
        if os.path.isfile(target) and digest(target, kind) == want:
            print("ok   ", rel)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print("fetch", repo, name, flush=True)
        path = hf_hub_download(repo_id=repo, filename=name, revision=rev)
        if digest(path, kind) != want:
            print(f"ERROR: {name} from {repo} doesn't match its known hash — not installed.", file=sys.stderr)
            return 1
        shutil.copyfile(path, target)
        print("ok   ", rel)
    print("Done. Restart ComfyUI, then run the ClipWaltz Connector (it reports what it can render).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
