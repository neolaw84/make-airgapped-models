---
name: make-airgapped-model
description: Package a Hugging Face model and all its dependencies into a portable directory for air-gapped systems.
---

# `make-airgapped-model` Skill

Use this skill when you need to package a Hugging Face model so that it is portable and runnable in an air-gapped system.

## Action Steps

### Step 1: Prompt the User for Runtime Specifics

Before downloading anything, ask the user to clarify the target runtime specifics (or read them if they are already provided in the prompt):
- **Target OS**: Linux, Windows, or macOS.
- **Hardware Platform**: CPU-only, or GPU (CUDA version or Apple Silicon MPS).
- **Docker Availability**: Would the user prefer a Docker container setup (requires Docker on the air-gapped machine) or a local virtual environment (venv) setup?
- **Python / PyTorch Version**: Any specific version requirement for PyTorch (e.g., matching a particular CUDA version).

### Step 2: Invoke the Packaging Script

Run the helper script `download_and_package.py` to create the air-gapped model bundle. The script is located at:
`[download_and_package.py](scripts/download_and_package.py)`

**Basic Command Usage:**
```bash
python .agents/skills/make-airgapped-model/scripts/download_and_package.py \
  --model_id <HF_MODEL_ID> \
  --target_dir models/<MODEL_DIR_NAME> \
  --os <linux|windows|macos> \
  --gpu <cpu|cuda|mps> \
  --use_docker <true|false>
```

Options:
- `--model_id`: The Hugging Face repo ID (e.g. `microsoft/resnet-18`).
- `--target_dir`: Where to save the output model bundle (e.g. `models/resnet-18`).
- `--os`: Target operating system.
- `--gpu`: Target hardware engine.
- `--use_docker`: Whether to generate Docker assets (`Dockerfile`, `docker-compose.yml`) instead of virtual env scripts.
- `--conda_env`: If creating a venv, you can optionally specify a conda environment name (e.g. `py312`) to act as the base interpreter path.

### Step 3: Verify the Packaged Model

Once the script completes:
1. Ensure `models/<model-name>/model_files` contains the actual configuration, vocabulary, and weights files (no symlinks).
2. Ensure `models/<model-name>/pip_cache` contains the downloaded wheel dependencies.
3. Test local environment installation by running the offline setup script (`setup_env.sh` or `setup_env.bat`).
4. Test offline inference by running the runner script (`run.sh` or `run.bat`) to verify that the model loads locally and runs correctly without internet access.
