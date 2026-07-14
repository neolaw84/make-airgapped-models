#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import shutil

def get_platform_paths(target_dir):
    """Get paths for venv and script targets."""
    return {
        "model_files": os.path.join(target_dir, "model_files"),
        "pip_cache": os.path.join(target_dir, "pip_cache"),
        "venv": os.path.join(target_dir, "venv"),
        "requirements": os.path.join(target_dir, "requirements.txt"),
        "inference": os.path.join(target_dir, "run_inference.py"),
        "setup_sh": os.path.join(target_dir, "setup_env.sh"),
        "setup_bat": os.path.join(target_dir, "setup_env.bat"),
        "run_sh": os.path.join(target_dir, "run.sh"),
        "run_bat": os.path.join(target_dir, "run.bat"),
        "readme": os.path.join(target_dir, "README.md")
    }

def run_command(cmd, cwd=None):
    """Run a shell command and print its output."""
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError(f"Command {' '.join(cmd)} failed.")
    return result.stdout

def create_venv(paths, conda_env):
    """Create Python virtual environment in target dir."""
    print("Creating virtual environment...")
    # Use conda run if environment is specified, else system python
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python", "-m", "venv", paths["venv"]]
    else:
        cmd = [sys.executable, "-m", "venv", paths["venv"]]
    run_command(cmd)

def get_venv_executables(venv_path):
    """Get correct path to python and pip executables in venv."""
    if os.name == 'nt' or sys.platform.startswith('win'):
        return {
            "python": os.path.join(venv_path, "Scripts", "python.exe"),
            "pip": os.path.join(venv_path, "Scripts", "pip.exe")
        }
    else:
        return {
            "python": os.path.join(venv_path, "bin", "python"),
            "pip": os.path.join(venv_path, "bin", "pip")
        }

def download_model(paths, model_id, venv_execs):
    """Download Hugging Face model using snapshot_download in the venv."""
    print(f"Installing huggingface_hub in temporary builder venv...")
    run_command([venv_execs["pip"], "install", "huggingface_hub"])
    
    print(f"Downloading model {model_id} from Hugging Face...")
    download_script = f"""
from huggingface_hub import snapshot_download
import sys
try:
    snapshot_download(
        repo_id="{model_id}", 
        local_dir="{paths['model_files']}", 
        local_dir_use_symlinks=False
    )
    print("Download completed successfully.")
except Exception as e:
    print(f"Error during snapshot download: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    # Run the inline Python script using the venv's python interpreter
    run_command([venv_execs["python"], "-c", download_script])

def detect_packages(model_files_path, custom_reqs, gpu_type):
    """Detect required Python packages based on config.json inside the model files."""
    packages = ["transformers", "huggingface_hub"]
    
    # Check if custom requirements were provided
    if custom_reqs:
        for r in custom_reqs.split(","):
            req = r.strip()
            if req and req not in packages:
                packages.append(req)
        return packages

    config_path = os.path.join(model_files_path, "config.json")
    model_type = ""
    architectures = []

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                model_type = config_data.get("model_type", "").lower()
                architectures = config_data.get("architectures", [])
        except Exception as e:
            print(f"Warning: Could not parse config.json to detect architecture: {e}")

    # Check for image classifications, vision, etc.
    is_vision = "vision" in model_type or any("image" in arch.lower() or "resnet" in arch.lower() or "vit" in arch.lower() for arch in architectures)
    # Check for audio/speech
    is_audio = "speech" in model_type or "audio" in model_type or any("audio" in arch.lower() or "speech" in arch.lower() for arch in architectures)

    if is_vision:
        packages.extend(["torch", "torchvision", "pillow"])
    elif is_audio:
        packages.extend(["torch", "torchaudio", "soundfile"])
    else:
        packages.append("torch")

    # Clean duplicates
    return list(dict.fromkeys(packages))

def download_wheels(paths, packages, gpu_type, os_target, venv_execs):
    """Download wheels for offline environment installation."""
    print(f"Downloading wheels into local cache for packages: {packages}...")
    
    # Save requirements.txt
    with open(paths["requirements"], "w") as f:
        for p in packages:
            f.write(f"{p}\n")

    # Set up index url for PyTorch CPU if requested
    index_url = None
    if gpu_type == "cpu":
        index_url = "https://download.pytorch.org/whl/cpu"

    # Separate torch packages from PyPI packages to use specific index if necessary
    torch_pkgs = [p for p in packages if p in ["torch", "torchvision", "torchaudio"]]
    pypi_pkgs = [p for p in packages if p not in torch_pkgs]

    # Download torch packages
    if torch_pkgs:
        cmd = [venv_execs["pip"], "download", "-d", paths["pip_cache"]]
        if index_url:
            cmd.extend(["--index-url", index_url])
        cmd.extend(torch_pkgs)
        run_command(cmd)

    # Download remaining packages
    if pypi_pkgs:
        cmd = [venv_execs["pip"], "download", "-d", paths["pip_cache"]]
        cmd.extend(pypi_pkgs)
        run_command(cmd)

def generate_inference_script(paths, model_id):
    """Generate run_inference.py template matching model architecture."""
    config_path = os.path.join(paths["model_files"], "config.json")
    model_type = ""
    architectures = []

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                model_type = config_data.get("model_type", "").lower()
                architectures = config_data.get("architectures", [])
        except Exception as e:
            pass

    is_vision = "vision" in model_type or any("image" in arch.lower() or "resnet" in arch.lower() or "vit" in arch.lower() for arch in architectures)
    is_audio = "speech" in model_type or "audio" in model_type or any("audio" in arch.lower() or "speech" in arch.lower() for arch in architectures)

    if is_vision:
        # Generate an image classification inference template
        code = f"""import os
import sys
import json
from PIL import Image
import numpy as np

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_files")
    
    print("Loading image processor and model from local path...")
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    
    # Get image path from arguments, or create a dummy image
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if image_path and os.path.exists(image_path):
        print(f"Loading input image: {{image_path}}")
        image = Image.open(image_path).convert("RGB")
    else:
        print("No input image path provided or file does not exist. Generating a dummy test image...")
        # Create a dummy image (e.g. 224x224 white image)
        image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        image.save("dummy_test.jpg")
        print("Generated 'dummy_test.jpg' for verification.")
        
    print("Preprocessing image...")
    inputs = processor(images=image, return_tensors="pt")
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    predicted_class_idx = logits.argmax(-1).item()
    print("\\n--- Inference Results ---")
    print(f"Predicted class index: {{predicted_class_idx}}")
    
    # Print label if config contains label mappings
    if hasattr(model.config, "id2label") and model.config.id2label:
        label = model.config.id2label.get(predicted_class_idx, "Unknown")
        print(f"Predicted label: {{label}}")
    print("-------------------------\\n")

if __name__ == "__main__":
    main()
"""
    elif is_audio:
        # Generate an audio classification inference template
        code = f"""import os
import sys
import numpy as np

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_files")
    
    print("Loading feature extractor and model from local path...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(model_path, local_files_only=True)
    
    # Get audio path from arguments, or create a dummy waveform
    audio_path = None
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    
    if audio_path and os.path.exists(audio_path):
        print(f"Loading input audio: {{audio_path}}")
        waveform, sampling_rate = torchaudio.load(audio_path)
        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Resample to 16000Hz if needed
        if sampling_rate != 16000:
            print(f"Resampling audio from {{sampling_rate}}Hz to 16000Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            waveform = resampler(waveform)
            sampling_rate = 16000
        audio_input = waveform.squeeze().numpy()
    else:
        print("No input audio path provided or file does not exist. Generating a dummy test waveform...")
        # Create a 1-second sine wave at 16kHz
        audio_input = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        sampling_rate = 16000
        import soundfile as sf
        sf.write("dummy_test.wav", audio_input, 16000)
        print("Generated 'dummy_test.wav' for verification.")
        
    print("Preprocessing audio...")
    inputs = feature_extractor(audio_input, sampling_rate=sampling_rate, return_tensors="pt")
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Since AudioSet models are typically multi-label classification, output probabilities with sigmoid
    probabilities = torch.sigmoid(logits).squeeze()
    
    print("\\n--- Inference Results ---")
    if hasattr(model.config, "id2label") and model.config.id2label:
        # Get top 5 classes
        top_k = min(5, len(probabilities))
        top_prob, top_indices = torch.topk(probabilities, top_k)
        for i in range(top_k):
            idx = top_indices[i].item()
            prob = top_prob[i].item()
            label = model.config.id2label.get(str(idx), model.config.id2label.get(idx, f"Class {{idx}}"))
            print(f"{{label}}: {{prob:.4f}}")
    else:
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Predicted class index: {{predicted_class_idx}}")
    print("-------------------------\\n")

if __name__ == "__main__":
    main()
"""
    else:
        # Default Text Generation/CausalLM inference template
        code = f"""import os
import sys

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_files")
    
    print("Loading tokenizer and model from local path...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    
    prompt = "Hello, I am running completely offline"
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        
    print(f"Running text generation for prompt: '{{prompt}}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\\n--- Inference Results ---")
    print(generated_text)
    print("-------------------------\\n")

if __name__ == "__main__":
    main()
"""

    with open(paths["inference"], "w") as f:
        f.write(code)
    print(f"Generated inference script: {paths['inference']}")

def generate_environment_scripts(paths, use_docker, gpu_type):
    """Generate setup_env, run, and optionally Docker configurations."""
    # Write setup_env.sh
    setup_sh_content = """#!/bin/bash
# Exit on error
set -e

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Creating offline virtual environment..."
python3 -m venv venv

echo "Installing dependencies from local pip cache..."
./venv/bin/pip install --no-index --find-links=./pip_cache -r requirements.txt

echo "Offline environment setup successfully completed!"
"""
    with open(paths["setup_sh"], "w") as f:
        f.write(setup_sh_content)
    os.chmod(paths["setup_sh"], 0o755)

    # Write setup_env.bat
    setup_bat_content = """@echo off
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

echo Creating offline virtual environment...
python -m venv venv

echo Installing dependencies from local pip cache...
.\\venv\\Scripts\\pip install --no-index --find-links=.\\pip_cache -r requirements.txt

echo Offline environment setup successfully completed!
"""
    with open(paths["setup_bat"], "w") as f:
        f.write(setup_bat_content)

    # Write run.sh
    run_sh_content = """#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
./venv/bin/python run_inference.py "$@"
"""
    with open(paths["run_sh"], "w") as f:
        f.write(run_sh_content)
    os.chmod(paths["run_sh"], 0o755)

    # Write run.bat
    run_bat_content = """@echo off
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%
.\\venv\\Scripts\\python run_inference.py %*
"""
    with open(paths["run_bat"], "w") as f:
        f.write(run_bat_content)

    # Docker Files (if requested)
    if use_docker:
        base_image = "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime" if gpu_type == "cuda" else "pytorch/pytorch:latest-cpu"
        dockerfile_content = f"""FROM {base_image}

WORKDIR /app

# Set offline environment variables
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Install requirements from pip_cache
COPY requirements.txt /app/requirements.txt
COPY pip_cache /app/pip_cache
RUN pip install --no-index --find-links=/app/pip_cache -r requirements.txt

# Copy model files and code
COPY model_files /app/model_files
COPY run_inference.py /app/run_inference.py

ENTRYPOINT ["python", "run_inference.py"]
"""
        dockerfile_path = os.path.join(os.path.dirname(paths["requirements"]), "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        docker_compose_content = """version: '3.8'
services:
  inference:
    build: .
    volumes:
      - .:/app
    # Set to true if running on GPU
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
"""
        compose_path = os.path.join(os.path.dirname(paths["requirements"]), "docker-compose.yml")
        with open(compose_path, "w") as f:
            f.write(docker_compose_content)

def generate_readme(paths, model_id):
    """Generate a README guide for the offline package."""
    readme_content = f"""# Offline Model Package: `{model_id}`

This directory contains a fully self-contained bundle for running `{model_id}` in an air-gapped system.

## Package Layout

* `model_files/`: The raw configuration and weight binaries from Hugging Face (no symlinks).
* `pip_cache/`: Offline Python package wheel (.whl) files for all dependencies.
* `requirements.txt`: Python package requirements.
* `run_inference.py`: Python wrapper to run inference locally referencing `./model_files`.
* `setup_env.sh` / `.bat`: Script to create the `venv` and install requirements offline.
* `run.sh` / `.bat`: Runner script to trigger inference locally.

## Instructions for Air-Gapped Deployment

1. **Transfer**: Compress and copy this entire directory to your target air-gapped machine.

2. **Setup and Execution**:

### Option A: System WITHOUT Conda (Local Virtual Environment)

* **On Linux/macOS**:
  ```bash
  chmod +x setup_env.sh run.sh
  ./setup_env.sh
  ./run.sh [optional_path_to_input]
  ```
* **On Windows**:
  ```cmd
  setup_env.bat
  run.bat [optional_path_to_input]
  ```

### Option B: System WITH Conda (Conda Environment)

If your air-gapped machine uses Conda:
1. Activate your target Conda environment:
   ```bash
   conda activate <your_conda_env>
   ```
2. Install the packaged dependencies offline from the local wheel cache:
   * **On Linux/macOS**:
     ```bash
     pip install --no-index --find-links=./pip_cache -r requirements.txt
     ```
   * **On Windows**:
     ```cmd
     pip install --no-index --find-links=.\\pip_cache -r requirements.txt
     ```
3. Run the inference wrapper:
   ```bash
   python run_inference.py [optional_path_to_input]
   ```
"""
    with open(paths["readme"], "w") as f:
        f.write(readme_content)

def main():
    parser = argparse.ArgumentParser(description="Package a Hugging Face model for offline usage.")
    parser.add_argument("--model_id", required=True, help="Hugging Face Model repository ID")
    parser.add_argument("--target_dir", required=True, help="Directory to save the packaged model")
    parser.add_argument("--os", choices=["linux", "windows", "macos"], default="linux", help="Target OS")
    parser.add_argument("--gpu", choices=["cpu", "cuda", "mps"], default="cpu", help="Target hardware accelerator")
    parser.add_argument("--requirements", help="Comma-separated custom package requirements")
    parser.add_argument("--use_docker", action="store_true", help="Generate Docker deployment artifacts")
    parser.add_argument("--conda_env", help="Conda environment to use when initializing local venv")
    
    args = parser.parse_args()
    
    # Create target directories
    paths = get_platform_paths(args.target_dir)
    os.makedirs(paths["model_files"], exist_ok=True)
    os.makedirs(paths["pip_cache"], exist_ok=True)
    
    try:
        # 1. Create building virtual environment
        create_venv(paths, args.conda_env)
        venv_execs = get_venv_executables(paths["venv"])
        
        # 2. Download model files to target
        download_model(paths, args.model_id, venv_execs)
        
        # 3. Detect python packages needed
        packages = detect_packages(paths["model_files"], args.requirements, args.gpu)
        
        # 4. Download wheel packages to cache
        download_wheels(paths, packages, args.gpu, args.os, venv_execs)
        
        # 5. Generate runnable inference templates
        generate_inference_script(paths, args.model_id)
        
        # 6. Generate environmental activation and runner scripts
        generate_environment_scripts(paths, args.use_docker, args.gpu)
        
        # 7. Generate README instructions
        generate_readme(paths, args.model_id)
        
        # Clean up temporary builder venv (will be rebuilt locally on air-gapped target via setup_env)
        # Note: We keep the venv during testing, but delete it here if we want portability. 
        # Actually, let's keep it for now so the test can be conducted, but we note in the README how to reconstruct it.
        
        print("\n=======================================================")
        print("SUCCESS: Offline model package generated successfully!")
        print(f"Output files stored in: {args.target_dir}")
        print("=======================================================\n")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
