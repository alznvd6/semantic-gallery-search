#!/usr/bin/env python3
"""Start the semantic photo search app

    python run_app.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "app" / "frontend"

BACKEND_HOST = "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the semantic photo search backend and frontend together."
    )
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Do not restart the backend when Python files change.",
    )
    return parser.parse_args()


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((BACKEND_HOST, port)) == 0


def backend_ready(port: int, attempts: int = 3) -> bool:
    """Return True when the API answers; retried because --reload restarts it."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                f"http://{BACKEND_HOST}:{port}/api/health", timeout=3
            ) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            if attempt + 1 < attempts:
                time.sleep(1)
    return False


def frontend_running(port: int, attempts: int = 2) -> bool:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                f"http://{BACKEND_HOST}:{port}/", timeout=3
            ) as response:
                return response.status < 500
        except (urllib.error.URLError, OSError):
            if attempt + 1 < attempts:
                time.sleep(1)
    return False


def start_backend(port: int, reload: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.backend.api:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")

    return subprocess.Popen(command, cwd=str(ROOT), env=env)


def npm_command() -> str:
    if os.name == "nt":
        return shutil.which("npm") or "npm.cmd"
    return shutil.which("npm") or "npm"


def ensure_frontend_dependencies(npm: str) -> None:
    if (FRONTEND_DIR / "node_modules").is_dir():
        return
    subprocess.run([npm, "install"], cwd=str(FRONTEND_DIR), check=True)


def start_frontend(npm: str, port: int) -> subprocess.Popen:
    command = [
        npm,
        "run",
        "dev",
        "--",
        "--port",
        str(port),
        "--strictPort",
    ]
    return subprocess.Popen(command, cwd=str(FRONTEND_DIR))


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def wait_for_backend(port: int, process: subprocess.Popen, timeout: float = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if backend_ready(port):
            return True
        if process.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def main() -> int:
    args = parse_args()

    if not FRONTEND_DIR.is_dir():
        raise FileNotFoundError(f"Frontend directory not found: {FRONTEND_DIR}")

    npm = npm_command()
    if not npm:
        raise FileNotFoundError("npm was not found on PATH; install Node.js first.")

    try:
        ensure_frontend_dependencies(npm)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"npm install failed with exit code {exc.returncode}."
        ) from exc

    backend: subprocess.Popen | None = None
    frontend: subprocess.Popen | None = None

    try:
        if args.backend_port != 8000:
            raise ValueError(
                "The Vite proxy targets port 8000; --backend-port must be 8000."
            )

        backend_already_running = backend_ready(args.backend_port)

        if not backend_already_running and port_in_use(args.backend_port):
            raise RuntimeError(
                f"Port {args.backend_port} is in use by a service that is not "
                "the app backend."
            )

        if not backend_already_running:
            backend = start_backend(args.backend_port, reload=not args.no_reload)
            if not wait_for_backend(args.backend_port, backend):
                if backend.poll() is not None:
                    raise RuntimeError(
                        f"Backend exited with code {backend.returncode}; see its output."
                    )
                raise TimeoutError("Backend did not become ready within 90 seconds.")

        frontend_already_running = frontend_running(args.frontend_port)
        if frontend_already_running and backend_already_running:
            return 0
        if not frontend_already_running:
            frontend = start_frontend(npm, args.frontend_port)

        while True:
            if backend is not None and backend.poll() is not None:
                raise RuntimeError(
                    f"Backend exited with code {backend.returncode}; see its output."
                )
            if frontend is not None and frontend.poll() is not None:
                raise RuntimeError(
                    f"Frontend exited with code {frontend.returncode}; see its output."
                )
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in (frontend, backend):
            if process is not None:
                terminate(process)


if __name__ == "__main__":
    sys.exit(main())
