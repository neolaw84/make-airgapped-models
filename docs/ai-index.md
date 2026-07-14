# Offline Models Repository Index

This repository develops and tests the `make-airgapped-model` AI agent skill. This skill packages any Hugging Face model and its execution environment into a portable, self-contained directory that can be transferred and executed on air-gapped systems.

## Repository Layout

* **[.agents/skills/make-airgapped-model/](.agents/skills/make-airgapped-model/)**: The core skill directory.
  * **[SKILL.md](.agents/skills/make-airgapped-model/SKILL.md)**: Main skill instructions for agents.
  * **[scripts/download_and_package.py](.agents/skills/make-airgapped-model/scripts/download_and_package.py)**: The underlying packaging script.
* **[docs/](docs/)**: Repository documentation.
  * **[ai-index.md](docs/ai-index.md)**: This index file.
* **[models/](models/)**: Output folder containing packaged models (e.g. `models/resnet-18/`).

## Project Core Philosophy

To achieve true air-gapped execution:
1. **No External Network Calls**: Model configurations, weights, and tokenizer files are downloaded locally to `model_files/`.
2. **Relative Paths**: No absolute paths are embedded in any script, config, or virtual environment variables.
3. **Local Dependency Cache**: All python wheels are pre-downloaded to `pip_cache/` so that target machine setup doesn't require an internet connection.
4. **Flexible Runtime**: The system prompts for runtime specifics (OS version, Docker preference, CUDA/GPU version) to bundle the correct environment artifacts.
