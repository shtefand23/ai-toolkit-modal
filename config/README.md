# Configuration Examples

This directory contains training configuration examples used by the repository.

Available files:

- `train_lora_flux_dev_modal_minimal.yaml` — Minimal FLUX LoRA training template for Modal

Usage:

1. Update the training config in [`.env`](../.env.example):

```dotenv
AI_TOOLKIT_LOCAL_CONFIG_DIR=./config
AI_TOOLKIT_TRAIN_CONFIG=train_lora_flux_dev_modal_minimal.yaml
```

2. Edit the YAML file to match your setup:

- `name`
- `datasets[0].folder_path`
- `model.name_or_path`
- `sample.prompts`

3. Run training:

```powershell
modal run .\run_ai_toolkit_train.py
```

Notes:

- The default template uses the in-container dataset path `/root/ai-toolkit/datasets/ash`
- This is a minimal starter template and may not be suitable for all GPU types and model versions
- If your GPU has less than 24GB VRAM, you'll likely need to reduce resolution, sampling frequency, or switch model configurations
