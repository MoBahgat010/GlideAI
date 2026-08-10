import sys
import subprocess

def run_server():
    """Start the FastAPI backend server."""
    print("Starting FastAPI Server")
    subprocess.run(["python3", "server/src/main.py"])


def run_triton():
    """Start the Triton Inference Server using Docker Compose."""
    print("Starting Triton Inference Server via Docker Compose...")
    subprocess.run(["docker", "compose", "-f", "triton_server/docker-compose.yml", "up"])


def run_celery():
    """Start the Celery background worker process."""
    print("Starting Celery Worker Process...")
    subprocess.run(["celery", "-A", "tasks", "worker", "--loglevel=info", "--pool=solo"])


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [server | triton | celery]")
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
