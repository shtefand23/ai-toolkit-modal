import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

try:
    from ai_toolkit_common import (
        AI_TOOLKIT_AUTH,
        COMMIT_INTERVAL_SECONDS,
        DATA_MOUNT_PATH,
        DB_PATH,
        GPU_TYPE,
        MODEL_MOUNT_PATH,
        MODEL_VOLUME_NAME,
        OUTPUT_PATH,
        PERSIST_DIR,
        UI_PORT,
        UI_ROOT,
        UI_TIMEOUT_SECONDS,
        build_image,
        datasets_volume,
        model_volume,
        persist_volume,
        prepare_datasets,
        replace_with_symlink,
        run_checked,
        spawn_logged_process,
    )
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parent
    TOOLKIT_ROOT = "/root/ai-toolkit"
    UI_ROOT = f"{TOOLKIT_ROOT}/ui"
    DATA_MOUNT_PATH = f"{TOOLKIT_ROOT}/datasets"
    OUTPUT_PATH = f"{TOOLKIT_ROOT}/output"
    MODEL_MOUNT_PATH = f"{TOOLKIT_ROOT}/modal_output"
    DB_PATH = f"{TOOLKIT_ROOT}/aitk_db.db"
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

    load_dotenv(ROOT_DIR / ".env")

    GPU_TYPE = os.environ.get("AI_TOOLKIT_GPU", "L4")
    UI_PORT = env_int("AI_TOOLKIT_UI_PORT", 8675)
    UI_TIMEOUT_SECONDS = env_int("AI_TOOLKIT_TIMEOUT", 86400)
    AI_TOOLKIT_AUTH = os.environ.get("AI_TOOLKIT_AUTH", "")
    PERSIST_DIR = "/root/ai-toolkit/modal_persist"
    PERSIST_VOLUME_NAME = os.environ.get("AI_TOOLKIT_UI_VOLUME", "ai-toolkit-ui-data")
    DATA_VOLUME_NAME = os.environ.get("AI_TOOLKIT_DATA_VOLUME", "ai-toolkit-datasets")
    MODEL_VOLUME_NAME = os.environ.get("AI_TOOLKIT_MODEL_VOLUME", "ai-toolkit-models")
    COMMIT_INTERVAL_SECONDS = env_int("AI_TOOLKIT_VOLUME_COMMIT_INTERVAL", 30)
    LOCAL_DATA_FOLDER = existing_local_dir(
        os.environ.get("AI_TOOLKIT_LOCAL_DATA_FOLDER", str(ROOT_DIR / "datasets"))
    )
    LOCAL_DATASET_SOURCE = existing_local_dir(os.environ.get("AI_TOOLKIT_LOCAL_DATASET_SOURCE", ""))
    LOCAL_CONFIG_DIR = existing_local_dir(os.environ.get("AI_TOOLKIT_LOCAL_CONFIG_DIR", ""))

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
        print(f"[INFO] {label}: {' '.join(cmd)} (cwd={cwd})", flush=True)
        result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if result.stdout:
            print(f"[{label} stdout]\n{result.stdout}", flush=True)
        if result.stderr:
            print(f"[{label} stderr]\n{result.stderr}", flush=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} failed with exit code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def spawn_logged_process(
        cmd: list[str],
        cwd: str,
        env: dict[str, str],
        label: str,
    ) -> subprocess.Popen:
        print(f"[INFO] Starting {label}: {' '.join(cmd)} (cwd={cwd})", flush=True)
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def pump_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[{label}] {line.rstrip()}", flush=True)
            return_code = process.wait()
            print(f"[INFO] {label} exited with code {return_code}", flush=True)

        threading.Thread(target=pump_output, daemon=True).start()
        return process

    def replace_with_symlink(link_path: str, target_path: str) -> None:
        if os.path.islink(link_path) or os.path.isfile(link_path):
            os.remove(link_path)
        elif os.path.isdir(link_path):
            shutil.rmtree(link_path)

        os.symlink(target_path, link_path)

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


image = build_image(include_ui_build=True)

app = modal.App(
    name="ai-toolkit-ui",
    image=image,
    volumes={
        PERSIST_DIR: persist_volume,
        DATA_MOUNT_PATH: datasets_volume,
        MODEL_MOUNT_PATH: model_volume,
    },
)


@app.function(gpu=GPU_TYPE, timeout=UI_TIMEOUT_SECONDS)
@modal.web_server(UI_PORT, startup_timeout=120)
def ui():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["DISABLE_TELEMETRY"] = "YES"
    if AI_TOOLKIT_AUTH:
        os.environ["AI_TOOLKIT_AUTH"] = AI_TOOLKIT_AUTH

    os.makedirs(PERSIST_DIR, exist_ok=True)

    os.makedirs(MODEL_MOUNT_PATH, exist_ok=True)

    persistent_db = os.path.join(PERSIST_DIR, "aitk_db.db")
    if not os.path.exists(persistent_db):
        open(persistent_db, "ab").close()

    replace_with_symlink(OUTPUT_PATH, MODEL_MOUNT_PATH)
    replace_with_symlink(DB_PATH, persistent_db)
    prepare_datasets()

    worker_env = dict(os.environ)
    worker_env["NODE_ENV"] = "production"
    worker_env["DATABASE_URL"] = f"file:{persistent_db}"
    worker_env["PYTHONUNBUFFERED"] = "1"
    worker_env["NODE_NO_WARNINGS"] = worker_env.get("NODE_NO_WARNINGS", "0")

    run_checked(["npx", "prisma", "generate"], cwd=UI_ROOT, env=worker_env, label="Prisma generate")
    run_checked(
        ["npx", "prisma", "db", "push", "--skip-generate"],
        cwd=UI_ROOT,
        env=worker_env,
        label="Prisma db push",
    )

    def commit_loop() -> None:
        while True:
            time.sleep(COMMIT_INTERVAL_SECONDS)
            try:
                persist_volume.commit()
            except Exception:
                pass
            try:
                datasets_volume.commit()
            except Exception:
                pass
            try:
                model_volume.commit()
            except Exception:
                pass

    threading.Thread(target=commit_loop, daemon=True).start()

    spawn_logged_process(["node", "dist/cron/worker.js"], cwd=UI_ROOT, env=worker_env, label="cron-worker")
    spawn_logged_process(
        ["npx", "next", "start", "--port", str(UI_PORT), "--hostname", "0.0.0.0"],
        cwd=UI_ROOT,
        env=worker_env,
        label="next-server",
    )
