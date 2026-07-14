# Make Airgapped Models

This repository houses the `make-airgapped-model` AI agent skill and its underlying packaging script. The goal of this project is to take any Hugging Face model and package it along with all of its Python dependency wheels into a single, fully portable directory. This directory can then be transferred and executed on air-gapped systems with zero internet connectivity.

## Core Features

* **No Symlinks**: Hugging Face cache typically relies on shared symlinks. This packaging tool downloads real file copies into a local `model_files/` folder.
* **Offline Virtual Environments**: All required Python package wheels (`.whl` files) are pre-downloaded to a local `pip_cache/` folder matching the target OS and hardware platforms.
* **Offline Verification**: Auto-generates local inference scripts (`run_inference.py`) configured with Hugging Face offline environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `local_files_only=True`) to guarantee no network calls occur.
* **Relative Paths**: No absolute paths are embedded anywhere in the runner scripts or configuration files.

---

## Repository Structure

* **[.agents/skills/make-airgapped-model/](.agents/skills/make-airgapped-model/)**: The core agent skill directory.
  * **[SKILL.md](.agents/skills/make-airgapped-model/SKILL.md)**: Main skill instructions for AI agents.
  * **[scripts/download_and_package.py](.agents/skills/make-airgapped-model/scripts/download_and_package.py)**: The underlying packaging CLI script.
* **[docs/](docs/)**: Repository documentation.
  * **[ai-index.md](docs/ai-index.md)**: Repository index file (all links are relative to repo root).
* **[models/](models/)** (Git Ignored): Output folder containing packaged models (e.g. `models/resnet-18/`).

---

## How to Install the Skill

### Option 1: Workspace-Level (Default)
Since workspace-level agent skills are automatically discovered and loaded from the `.agents/` folder, no extra installation step is required. When working in this repository workspace, the AI assistant will automatically discover the `make-airgapped-model` skill and use it.

### Option 2: System-Wide (Global)
To install the skill system-wide so it is accessible to agents across all your project workspaces, copy the skill folder into the global customization directory:

```bash
mkdir -p ~/.gemini/config/skills/make-airgapped-model
cp -r .agents/skills/make-airgapped-model/* ~/.gemini/config/skills/make-airgapped-model/
```

---

## How to Use the Skill / Script

You (or the agent) can run the packaging script directly from the root of the repository.

### Command Syntax

```bash
python .agents/skills/make-airgapped-model/scripts/download_and_package.py \
  --model_id <HF_MODEL_ID> \
  --target_dir models/<MODEL_DIR_NAME> \
  --os <linux|windows|macos> \
  --gpu <cpu|cuda|mps> \
  [--conda_env <conda_env_name>] \
  [--use_docker]
```

### Options

* `--model_id`: The Hugging Face repo ID (e.g., `microsoft/resnet-18` or `gpt2`).
* `--target_dir`: The destination directory where the portable model package will be created (e.g., `models/resnet-18`).
* `--os`: Target operating system for dependency wheels (`linux`, `windows`, or `macos`).
* `--gpu`: Target hardware engine (`cpu`, `cuda`, or `mps`).
* `--conda_env` (optional): If specified, uses that conda environment's python path as the base to create the builder venv (e.g. `py312`).
* `--use_docker` (optional): Generates a `Dockerfile` and a `docker-compose.yml` for containerized execution on the target machine instead of Python venv scripts.

---

## What the Packager Generates

Running the tool creates a packaged directory `models/<model-name>/` containing:

1. **`model_files/`**: Configuration, tokenizer, and weight binary files from Hugging Face.
2. **`pip_cache/`**: Offline wheel library cache containing the model's required packages (including PyTorch, Transformers, Pillow, and all dependencies).
3. **`requirements.txt`**: The requirements file listing dependencies.
4. **`setup_env.sh` / `.bat`**: Sets up the local virtual environment (`venv`) and installs the dependencies entirely from `pip_cache/` with `--no-index`.
5. **`run_inference.py`**: A python wrapper that loads the model via relative paths in offline mode and runs a sample verification run.
6. **`run.sh` / `.bat`**: Activates the venv and triggers `run_inference.py`.
7. **`README.md`**: Direct instructions on how to transfer, install, and execute the package on the air-gapped target.

---

## Testing Example: ResNet-18

To package Microsoft ResNet-18 (CPU version) using a base Python 3.12 conda environment:

```bash
python .agents/skills/make-airgapped-model/scripts/download_and_package.py \
  --model_id microsoft/resnet-18 \
  --target_dir models/resnet-18 \
  --os linux \
  --gpu cpu \
  --conda_env py312
```

### Deploying & Verifying on the Air-Gapped Machine:

1. **Transfer**: Copy the `models/resnet-18` directory to the target air-gapped machine.

2. **Setup and Execution**:

#### Option A: System WITHOUT Conda (Local Virtual Environment)
* **On Linux/macOS**:
  ```bash
  cd resnet-18/
  chmod +x setup_env.sh run.sh
  ./setup_env.sh
  ./run.sh
  ```
* **On Windows**:
  ```cmd
  cd resnet-18
  setup_env.bat
  run.bat
  ```

#### Option B: System WITH Conda (Conda Environment)
If your target machine uses Conda:
1. Activate your target Conda environment:
   ```bash
   conda activate <your_conda_env>
   ```
2. Install dependencies from the local wheel cache:
   ```bash
   pip install --no-index --find-links=./pip_cache -r requirements.txt
   ```
3. Run the model wrapper:
   ```bash
   python run_inference.py
   ```
