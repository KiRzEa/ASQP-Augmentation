"""Test the offline package on a Modal A100 80GB the way the advisor's server runs it: torch 2.8.0 preinstalled, the
rest installed from the package's wheels only (pip --no-index), the model read from a local HF cache, and NO network
while the job runs (block_network=True).

    unzip -q dist/asqp_llm_package.zip -d <dir>                     # the exact zip that goes to the server
    ASQP_PKG=<dir> modal run llm/modal_app.py::download              # once: weights -> volume (see download)
    ASQP_PKG=<dir> modal run llm/modal_app.py::run --cmd "python llm/pipeline.py run --smoke"
    ASQP_PKG=<dir> modal run llm/modal_app.py::run --backend unsloth --cmd "python llm/inlg.py train ..."
    ASQP_PKG=<dir> modal run llm/modal_app.py::run --model-dir qwen2.5-14b-instruct --cmd "... --profile qwen"
    modal volume get asqp-work <path> <local>                         # fetch results

Outputs persist on the volume asqp-work (/work/pkg = package copy + outputs), so a job can resume.
"""
import os
import subprocess

import modal

PKG = os.environ.get("ASQP_PKG", "dist/pkg")
app = modal.App("asqp-llm")
models = modal.Volume.from_name("asqp-models", create_if_missing=True)
work = modal.Volume.from_name("asqp-work", create_if_missing=True)

base = modal.Image.debian_slim(python_version="3.10").apt_install("zip", "unzip").pip_install("torch==2.8.0")
hf_image = (base.add_local_dir(f"{PKG}/llm", "/wheels_src", copy=True)
            .run_commands("python -m venv --system-site-packages /venv",
                          "/venv/bin/python -m pip install --no-index --find-links /wheels_src/wheels "
                          "-r /wheels_src/requirements.txt")
            .env({"PATH": "/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
            .add_local_dir(PKG, "/pkg"))
unsloth_image = base.run_commands("pip install unsloth unsloth_zoo 'torch==2.8.0'").add_local_dir(PKG, "/pkg")
ENV = {"ASQP_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "PYTHONUNBUFFERED": "1", "UNSLOTH_DISABLE_STATISTICS": "1"}


@app.function(image=base.pip_install("huggingface_hub==0.36.2", "hf-xet==1.6.0"), volumes={"/models": models},
              timeout=3600)
def download(repo: str = "unsloth/gemma-7b", name: str = "gemma-7b"):
    """gemma-7b: weights of the ungated mirror unsloth/gemma-7b (sha256 of the 4 safetensors files = google/gemma-7b,
    checked 2026-11-11); config and tokenizer files of google/gemma-7b are uploaded separately with
    `modal volume put asqp-models <dir> /gemma-7b/`.
    qwen: modal run llm/modal_app.py::download --repo Qwen/Qwen2.5-14B-Instruct --name qwen2.5-14b-instruct"""
    from huggingface_hub import snapshot_download
    pats = ["*.safetensors"] if name == "gemma-7b" else ["*.safetensors", "*.json", "merges.txt", "vocab.json"]
    snapshot_download(repo, local_dir=f"/models/{name}", allow_patterns=pats)
    models.commit()
    print(subprocess.run(f"ls -la /models/{name}; du -sh /models/{name}", shell=True, capture_output=True,
                         text=True).stdout)


def _run(cmd, tag, model_dir):
    os.makedirs(f"/work/{tag}", exist_ok=True)
    subprocess.run(f"cp -r /pkg/. /work/{tag}/", shell=True, check=True)  # code + data; outputs on the volume kept
    env = {**os.environ, **ENV, "ASQP_MODEL": f"/models/{model_dir}"}
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True,
                         capture_output=True, text=True).stdout, flush=True)
    rc = subprocess.call(cmd, shell=True, cwd=f"/work/{tag}", env=env)
    work.commit()
    return rc


@app.function(image=hf_image, gpu="A100-80GB", volumes={"/models": models, "/work": work}, block_network=True,
              timeout=24 * 3600)
def run_hf(cmd: str, tag: str, model_dir: str):
    return _run(cmd, tag, model_dir)


@app.function(image=unsloth_image, gpu="A100-80GB", volumes={"/models": models, "/work": work},
              block_network=True, timeout=24 * 3600)
def run_unsloth(cmd: str, tag: str, model_dir: str):
    return _run(cmd, tag, model_dir)


@app.local_entrypoint()
def run(cmd: str, backend: str = "hf", tag: str = "pkg", model_dir: str = "gemma-7b"):
    """model_dir: folder in the asqp-models volume (gemma-7b, or qwen2.5-14b-instruct with --profile qwen in cmd)."""
    fn = run_unsloth if backend == "unsloth" else run_hf
    rc = fn.remote(cmd + (" --backend unsloth" if backend == "unsloth" and "--backend" not in cmd else ""), tag,
                   model_dir)
    print("exit code", rc)
