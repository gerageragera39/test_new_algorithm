from __future__ import annotations

import socket
import threading
import time
import webbrowser

import uvicorn


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


def main() -> None:
    from app.main import app as fastapi_app

    host = "127.0.0.1"
    port = 8765

    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if _wait_for_port(host, port):
        webbrowser.open(f"http://{host}:{port}/ui")

    thread.join()


if __name__ == "__main__":
    main()
