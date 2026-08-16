import sys
import subprocess
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SERVER_DIR = ROOT_DIR / "server"


def _get_server_env():
    env = os.environ.copy()
    server_path = str(SERVER_DIR)
    src_path = str(SERVER_DIR / "src")
    root_path = str(ROOT_DIR)
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{server_path}:{src_path}:{root_path}:{current_pp}".rstrip(":")
    return env


def run_server():
    """Start the FastAPI backend server."""
    print("Starting Enterprise FastAPI Server...")
    subprocess.run([sys.executable, "main.py"], cwd=str(SERVER_DIR), env=_get_server_env())


def run_triton():
    """Start the Triton Inference Server using Docker Compose."""
    print("Starting Triton Inference Server via Docker Compose...")
    subprocess.run(["docker", "compose", "-f", "triton_server/docker-compose.yml", "up"], cwd=str(ROOT_DIR))


def run_celery():
    """Start the Celery background worker process."""
    print("Starting Celery Worker Process...")
    subprocess.run(["celery", "-A", "src.jobs.tasks", "worker", "--loglevel=info", "--pool=threads", "--concurrency=4"], cwd=str(SERVER_DIR), env=_get_server_env())


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [fastapi | triton | celery]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "fastapi":
        run_server()
    elif command == "triton":
        run_triton()
    elif command == "celery":
        run_celery()
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
