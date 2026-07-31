import os
import shlex
import shutil
import subprocess
from pathlib import Path

import modal

try:
    from ai_toolkit_common import (
        DATA_MOUNT_PATH,
        GPU_TYPE,
        MODEL_VOLUME_NAME,
        PERSIST_DIR,
        TOOLKIT_ROOT,
        TRAIN_CONFIG_FILE,
        TRAIN_EXTRA_ARGS,
        TRAIN_OUTPUT_DIR,
        TRAIN_TIMEOUT_SECONDS,
        build_image,
        datasets_volume,
        model_volume,
        persist_volume,
        prepare_datasets,
        resolve_container_config_path,
        run_checked,
    )
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parent
    TOOLKIT_ROOT = "/root/ai-toolkit"
    DATA_MOUNT_PATH = f"{TOOLKIT_ROOT}/datasets"
    OUTPUT_PATH = f"{TOOLKIT_ROOT}/output"
    MODEL_MOUNT_PATH = f"{TOOLKIT_ROOT}/modal_output"
    LOCAL_DATA_MOUNT_PATH = "/root/local_data"
    LOCAL_DATASET_SOURCE_MOUNT_PATH = "/mnt/dataset_source"
    LOCAL_CONFIGS_MOUNT_PATH = "/root/local_configs"

    def load_dotenv(dotenv_path: Path) -> None:
        if not dotenv_path.exists():
            return

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            os.environ.setdefault(key, value)

    def env_int(name: str, default: int) -> int:
        raw_value = os.environ.get(name, str(default))
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc

    def existing_local_dir(path_value: str) -> str:
        if not path_value:
            return ""

        local_path = Path(path_value).expanduser()
        if not local_path.is_absolute():
            local_path = (ROOT_DIR / local_path).resolve()

        if local_path.exists() and local_path.is_dir():
            return str(local_path)

        return ""

    def resolve_local_file(path_value: str) -> str:
        if not path_value:
            return ""

        local_path = Path(path_value).expanduser()
        if not local_path.is_absolute():
            local_path = (ROOT_DIR / local_path).resolve()

        if local_path.exists() and local_path.is_file():
            return str(local_path)

        return ""

    load_dotenv(ROOT_DIR / ".env")

    GPU_TYPE = os.environ.get("AI_TOOLKIT_GPU", "L4")
    TRAIN_TIMEOUT_SECONDS = env_int("AI_TOOLKIT_TRAIN_TIMEOUT", 7200)
    PERSIST_DIR = "/root/ai-toolkit/modal_persist"
    PERSIST_VOLUME_NAME = os.environ.get("AI_TOOLKIT_UI_VOLUME", "ai-toolkit-ui-data")
    DATA_VOLUME_NAME = os.environ.get("AI_TOOLKIT_DATA_VOLUME", "ai-toolkit-datasets")
    MODEL_VOLUME_NAME = os.environ.get("AI_TOOLKIT_MODEL_VOLUME", "ai-toolkit-models")
    LOCAL_DATA_FOLDER = existing_local_dir(
        os.environ.get("AI_TOOLKIT_LOCAL_DATA_FOLDER", str(ROOT_DIR / "datasets"))
    )
    LOCAL_DATASET_SOURCE = existing_local_dir(os.environ.get("AI_TOOLKIT_LOCAL_DATASET_SOURCE", ""))
    LOCAL_CONFIG_DIR = existing_local_dir(os.environ.get("AI_TOOLKIT_LOCAL_CONFIG_DIR", ""))
    TRAIN_CONFIG_FILE = os.environ.get("AI_TOOLKIT_TRAIN_CONFIG", "")
    TRAIN_EXTRA_ARGS = os.environ.get("AI_TOOLKIT_TRAIN_EXTRA_ARGS", "")
    TRAIN_OUTPUT_DIR = os.environ.get("AI_TOOLKIT_TRAIN_OUTPUT_DIR", MODEL_MOUNT_PATH)

    persist_volume = modal.Volume.from_name(PERSIST_VOLUME_NAME, create_if_missing=True)
    datasets_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

    def build_image(include_ui_build: bool) -> modal.Image:
        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install(
                "git",
                "curl",
                "ca-certificates",
                "build-essential",
                "python3",
                "make",
                "g++",
                "libgl1",
                "libglib2.0-0",
            )
            .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "DISABLE_TELEMETRY": "YES"})
            .run_commands(
                "bash -lc 'curl -fsSL https://deb.nodesource.com/setup_20.x | bash -'",
                "bash -lc 'apt-get update && apt-get install -y nodejs'",
                "bash -lc 'rm -rf /root/ai-toolkit && git clone --recursive https://github.com/ostris/ai-toolkit.git /root/ai-toolkit'",
                "bash -lc 'cd /root/ai-toolkit && git submodule update --init --recursive'",
                "bash -lc \"python -c \\\"from pathlib import Path; p=Path('/root/ai-toolkit/ui/src/app/api/img/[...imagePath]/route.ts'); t=p.read_text(encoding='utf-8'); o='const filepath = decodeURIComponent(imagePath);'; n='const rawPath = Array.isArray(imagePath) ? imagePath.join(\\'/\\') : imagePath;\\\\n    let filepath = decodeURIComponent(rawPath);\\\\n    if (!filepath.startsWith(\\'/\\')) {\\\\n      filepath = \\'/\\' + filepath;\\\\n    }'; assert o in t, f'patch target not found: {p}'; p.write_text(t.replace(o, n, 1), encoding='utf-8')\\\"\"",
                "bash -lc 'python -m pip install --upgrade pip'",
                "bash -lc 'python -m pip install torch==2.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126'",
                "bash -lc 'python -c \"import torch; print(torch.__version__)\"'",
                "bash -lc 'python -m pip install -r /root/ai-toolkit/requirements.txt'",
                "bash -lc 'python -c \"import torch; print(torch.__version__)\"'",
            )
        )

        if include_ui_build:
            image = image.run_commands(
                "bash -lc 'cd /root/ai-toolkit/ui && npm install sqlite3@5.1.6 && npm install && npm run update_db && npm run build'"
            )

        if LOCAL_DATA_FOLDER:
            image = image.add_local_dir(LOCAL_DATA_FOLDER, LOCAL_DATA_MOUNT_PATH, copy=True)

        if LOCAL_DATASET_SOURCE:
            image = image.add_local_dir(
                LOCAL_DATASET_SOURCE,
                LOCAL_DATASET_SOURCE_MOUNT_PATH,
                copy=True,
            )

        if LOCAL_CONFIG_DIR:
            image = image.add_local_dir(LOCAL_CONFIG_DIR, LOCAL_CONFIGS_MOUNT_PATH, copy=True)

        return image

    def run_checked(cmd: list[str], cwd: str, env: dict[str, str], label: str) -> None:
        result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} failed with exit code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def sync_directory(source_root: str, target_root: str, overwrite: bool) -> None:
        if not os.path.exists(source_root):
            return

        os.makedirs(target_root, exist_ok=True)

        for item in os.listdir(source_root):
            src = os.path.join(source_root, item)
            dst = os.path.join(target_root, item)

            if os.path.isdir(src):
                if os.path.exists(dst):
                    if overwrite:
                        shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                else:
                    shutil.copytree(src, dst)
            else:
                if overwrite or not os.path.exists(dst):
                    shutil.copy2(src, dst)

    def prepare_datasets() -> None:
        if os.path.exists(LOCAL_DATA_MOUNT_PATH):
            sync_directory(LOCAL_DATA_MOUNT_PATH, DATA_MOUNT_PATH, overwrite=True)

        if os.path.exists(LOCAL_DATASET_SOURCE_MOUNT_PATH):
            sync_directory(LOCAL_DATASET_SOURCE_MOUNT_PATH, DATA_MOUNT_PATH, overwrite=False)

        try:
            datasets_volume.commit()
        except Exception as exc:
            print(f"[WARN] Could not commit datasets volume: {exc}")

    def resolve_container_config_path(config_value: str) -> str:
        if not config_value:
            return ""

        if config_value.startswith("/"):
            return config_value

        local_file = resolve_local_file(config_value)
        if local_file and LOCAL_CONFIG_DIR:
            local_config_dir_path = Path(LOCAL_CONFIG_DIR)
            try:
                relative_path = Path(local_file).relative_to(local_config_dir_path)
                return f"{LOCAL_CONFIGS_MOUNT_PATH}/{relative_path.as_posix()}"
            except ValueError:
                pass

        return f"{LOCAL_CONFIGS_MOUNT_PATH}/{Path(config_value).as_posix()}"


image = build_image(include_ui_build=False)

app = modal.App(
    name="ai-toolkit-train",
    image=image,
    volumes={
        PERSIST_DIR: persist_volume,
        DATA_MOUNT_PATH: datasets_volume,
        TRAIN_OUTPUT_DIR: model_volume,
    },
)


def normalize_config_path(config_value: str) -> str:
    resolved = resolve_container_config_path(config_value)
    if not resolved:
        raise ValueError(
            "Missing training config. Set AI_TOOLKIT_TRAIN_CONFIG in .env "
            "or pass --config-file-list-str to modal run."
        )
    return resolved


def normalize_config_list(config_value: str) -> list[str]:
    items = [item.strip() for item in config_value.replace(";", ",").split(",") if item.strip()]
    if not items:
        raise ValueError(
            "Missing training config. Set AI_TOOLKIT_TRAIN_CONFIG in .env "
            "or pass --config-file-list-str to modal run."
        )
    return [normalize_config_path(item) for item in items]


@app.function(gpu=GPU_TYPE, timeout=TRAIN_TIMEOUT_SECONDS)
def train(config_file_list_str: str = "", extra_args: str = "") -> str:
    prepare_datasets()

    config_list = normalize_config_list(config_file_list_str or TRAIN_CONFIG_FILE)

    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["DISABLE_TELEMETRY"] = "YES"
    env["AITK_OUTPUT_DIR"] = TRAIN_OUTPUT_DIR

    cmd = ["python", "run.py", *config_list]
    merged_extra_args = extra_args or TRAIN_EXTRA_ARGS
    if merged_extra_args:
        cmd.extend(shlex.split(merged_extra_args))

    run_checked(cmd, cwd=TOOLKIT_ROOT, env=env, label="AI-Toolkit training")

    try:
        model_volume.commit()
    except Exception as exc:
        print(f"[WARN] Could not commit model volume: {exc}")

    return (
        f"Training completed.\n"
        f"configs={', '.join(config_list)}\n"
        f"output_volume={MODEL_VOLUME_NAME}\n"
        f"output_dir={TRAIN_OUTPUT_DIR}"
    )


@app.local_entrypoint()
def main(config_file_list_str: str = "", extra_args: str = ""):
    result = train.remote(config_file_list_str=config_file_list_str, extra_args=extra_args)
    print(result)
