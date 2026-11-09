"""Download Utonia checkpoints into third_party/Utonia/ckpt and make .venv self-contained."""
import os
import shutil
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "third_party", "Utonia", "ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)

# 1. Make .venv python3.11 self-contained inside workspace so sandbox can execute it directly
uv_py_src = "/home/projval/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu"
venv_py_dst = os.path.join(ROOT, ".venv", "cpython311")
if os.path.isdir(uv_py_src) and not os.path.isdir(venv_py_dst):
    print(f"Copying {uv_py_src} -> {venv_py_dst} ...")
    shutil.copytree(uv_py_src, venv_py_dst, symlinks=True)

venv_bin_py = os.path.join(ROOT, ".venv", "bin", "python")
local_py_bin = os.path.join(venv_py_dst, "bin", "python3.11")
if os.path.isfile(local_py_bin):
    if os.path.islink(venv_bin_py) or os.path.exists(venv_bin_py):
        os.remove(venv_bin_py)
    os.symlink(local_py_bin, venv_bin_py)
    pyvenv_cfg = os.path.join(ROOT, ".venv", "pyvenv.cfg")
    if os.path.isfile(pyvenv_cfg):
        with open(pyvenv_cfg, "r", encoding="utf-8") as f:
            cfg = f.read()
        cfg = cfg.replace(os.path.join(uv_py_src, "bin"), os.path.join(venv_py_dst, "bin"))
        with open(pyvenv_cfg, "w", encoding="utf-8") as f:
            f.write(cfg)
    print("Updated .venv/bin/python to workspace-local cpython311.")

# 2. Download Utonia checkpoints
for model_name in ["utonia_tiny", "utonia_small", "utonia"]:
    target = os.path.join(CKPT_DIR, f"{model_name}.pth")
    if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
        print(f"Already downloaded: {target} ({os.path.getsize(target)/1e6:.1f} MB)")
        continue
    print(f"Downloading {model_name}.pth from Pointcept/Utonia ...")
    path = hf_hub_download(
        repo_id="Pointcept/Utonia",
        filename=f"{model_name}.pth",
        repo_type="model",
        revision="main",
        local_dir=CKPT_DIR,
    )
    print(f"Downloaded: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
