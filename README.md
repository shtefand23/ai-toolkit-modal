# AI Toolkit Modal

This repository deploys [`ostris/ai-toolkit`](https://github.com/ostris/ai-toolkit) to Modal, supporting both:

- Web UI deployment
- Config-driven training job execution

## Directory Structure

- [`ai_toolkit_common.py`](./ai_toolkit_common.py) — Shared configuration, image building, and data sync logic
- [`run_ai_toolkit_ui.py`](./run_ai_toolkit_ui.py) — UI entry script
- [`run_ai_toolkit_train.py`](./run_ai_toolkit_train.py) — Training entry script
- [`config`](./config) — Training configuration examples
- [`datasets`](./datasets) — Local sample data directory, synced to Modal dataset volume on startup
- [`.env.example`](./.env.example) — Environment variable template
- [`requirements.txt`](./requirements.txt) — Local dependencies

## Prerequisites

- Python 3.10+
- A Modal account

Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

Log in to Modal:

```powershell
python -m modal setup
```

## Configuration

A [`.env`](./.env) file is provided in the repository root.

Common configuration options:

- `AI_TOOLKIT_UI_PORT` — UI port inside the container, default `8675`
- `AI_TOOLKIT_GPU` — Modal GPU type, default `L4`
- `AI_TOOLKIT_TIMEOUT` — UI function timeout in seconds, default `86400`
- `AI_TOOLKIT_TRAIN_TIMEOUT` — Training function timeout in seconds, default `7200`
- `AI_TOOLKIT_UI_VOLUME` — UI persistent volume name
- `AI_TOOLKIT_DATA_VOLUME` — Dataset volume name
- `AI_TOOLKIT_MODEL_VOLUME` — Training output volume name
- `AI_TOOLKIT_VOLUME_COMMIT_INTERVAL` — Volume auto-commit interval
- `AI_TOOLKIT_AUTH` — Optional UI password
- `AI_TOOLKIT_LOCAL_DATA_FOLDER` — Local directory to sync to the dataset volume on startup
- `AI_TOOLKIT_LOCAL_DATASET_SOURCE` — Optional extra data directory; only imports if the target dataset doesn't exist yet
- `AI_TOOLKIT_LOCAL_CONFIG_DIR` — Training config directory, mounted to `/root/local_configs` in the container
- `AI_TOOLKIT_TRAIN_CONFIG` — Default training config file; supports comma-separated multiple configs
- `AI_TOOLKIT_TRAIN_EXTRA_ARGS` — Additional training arguments
- `AI_TOOLKIT_TRAIN_OUTPUT_DIR` — Training output directory inside the container, default `/root/ai-toolkit/modal_output`; UI maps it to `/root/ai-toolkit/output`

Notes:

- When `AI_TOOLKIT_LOCAL_DATA_FOLDER=./datasets`, the repository's [`datasets`](./datasets) folder is synced to the Modal dataset volume.
- `AI_TOOLKIT_LOCAL_DATASET_SOURCE` is useful for pointing to a larger local dataset directory; existing datasets with the same name are not overwritten.
- `AI_TOOLKIT_LOCAL_CONFIG_DIR` is where you place your own YAML config files.
- `AI_TOOLKIT_TRAIN_CONFIG` accepts either absolute container paths or paths relative to `AI_TOOLKIT_LOCAL_CONFIG_DIR`; multiple configs can be comma-separated.

## Starting the Web UI

For Windows PowerShell, enable UTF-8 first:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

Start the UI:

```powershell
modal serve .\run_ai_toolkit_ui.py
```

Modal will output a public access URL once it's running.

## Running Training Jobs

Configure the training settings in [`.env`](./.env), for example:

```dotenv
AI_TOOLKIT_LOCAL_CONFIG_DIR=./config
AI_TOOLKIT_TRAIN_CONFIG=train_lora_flux_dev_modal_minimal.yaml
AI_TOOLKIT_MODEL_VOLUME=ai-toolkit-models
```

Then run:

```powershell
modal run .\run_ai_toolkit_train.py
```

To temporarily override config files or pass extra arguments:

```powershell
modal run .\run_ai_toolkit_train.py -- --config-file-list-str=job1.yaml,job2.yaml --extra-args="--sample_every_n_steps 100"
```

The training script will:

- Sync local `data` to the dataset volume
- Read YAML configs from the local config directory
- Execute `python run.py <config>` inside the container
- Write training output to `AI_TOOLKIT_MODEL_VOLUME`

A minimal ready-to-run template is included:

- [`config/train_lora_flux_dev_modal_minimal.yaml`](./config/train_lora_flux_dev_modal_minimal.yaml)

This template is based on the upstream `ai-toolkit` FLUX LoRA example, using the `ash` dataset at `/root/ai-toolkit/datasets/ash` by default. See upstream docs and examples:

- https://github.com/ostris/ai-toolkit
- https://github.com/ostris/ai-toolkit/blob/main/run.py
- https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_lora_wan21_1b_24gb.yaml

## Data Persistence

Three Modal volumes are used:

- `AI_TOOLKIT_UI_VOLUME` — Persists the database and output directory
- `AI_TOOLKIT_DATA_VOLUME` — Persists the dataset directory
- `AI_TOOLKIT_MODEL_VOLUME` — Persists training output

Corresponding paths inside the container:

- Database: `/root/ai-toolkit/aitk_db.db`
- Output: `/root/ai-toolkit/output`
- Datasets: `/root/ai-toolkit/datasets`
- Training output: Set by `AI_TOOLKIT_TRAIN_OUTPUT_DIR`, default `/root/ai-toolkit/modal_output`

## Implementation Notes

- `.env` is automatically loaded when scripts start
- The local `data` directory is copied into the image during build and synced to the dataset volume on startup
- UI and training share the same common configuration and data sync logic

## Troubleshooting

- If `modal serve` or `modal run` fails, first make sure you've run `python -m modal setup`
- First builds are slow because PyTorch, Node.js, and AI-Toolkit dependencies need to be installed inside the container
- If local data directories aren't taking effect, verify the paths in [`.env`](./.env) actually exist
- If training configs aren't found, check `AI_TOOLKIT_LOCAL_CONFIG_DIR` and `AI_TOOLKIT_TRAIN_CONFIG` first
