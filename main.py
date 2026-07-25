"""
Pixel Pilot — local GUI automation agent.

Usage:
    python main.py
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request

import config  # noqa: F401 — sets HF_HOME


def is_llm_running():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{config.LLM_PORT}/v1/models", timeout=1)
        return True
    except Exception:
        return False


def is_mcp_running():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{config.MCP_PORT}/sse", timeout=1)
        return True
    except Exception:
        return False


def get_gguf_path():
    """Download GGUF model if needed, return path."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=config.LLM_MODEL_GGUF,
        filename=config.LLM_MODEL_GGUF_FILE,
    )
    return path


def start_llm_server(backend=None):
    backend = backend or config.LLM_BACKEND

    if is_llm_running():
        print(f"LLM server already running on port {config.LLM_PORT}")
        return None

    if backend == "mlx":
        print(f"Starting MLX: {config.LLM_MODEL_MLX} on port {config.LLM_PORT}...")
        print("(first run downloads the model — this may take a few minutes)")
        cmd = [
            sys.executable, "-m", "mlx_lm", "server",
            "--model", config.LLM_MODEL_MLX,
            "--port", str(config.LLM_PORT),
        ]
    elif backend == "llama_cpp":
        print(f"Starting llama.cpp: {config.LLM_MODEL_GGUF} on port {config.LLM_PORT}...")
        print("(first run downloads the model — this may take a few minutes)")
        gguf_path = get_gguf_path()
        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", gguf_path,
            "--host", "127.0.0.1",
            "--port", str(config.LLM_PORT),
            "--n_ctx", "4096",
            "--n_gpu_layers", "-1",
            "--chat_format", "chatml",
        ]
    else:
        print(f"Unknown backend: {backend}")
        sys.exit(1)

    proc = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=None)

    while True:
        if proc.poll() is not None:
            print("LLM server exited unexpectedly.")
            sys.exit(1)
        if is_llm_running():
            print("LLM server ready.")
            return proc
        time.sleep(2)


def main():
    llm_proc = None

    def cleanup(sig=None, frame=None):
        if llm_proc:
            print("\nStopping LLM server...")
            llm_proc.terminate()
            llm_proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    print("=" * 40)
    print("  Pixel Pilot")
    print("=" * 40)
    print(f"  Backend: {config.LLM_BACKEND}")
    print()
    print("  1. Run a task (starts everything)")
    print("  2. GoClick MCP server (port %d)" % config.MCP_PORT)
    print("  3. LLM server [%s] (port %d)" % (config.LLM_BACKEND, config.LLM_PORT))
    print("  4. Inference (needs 2 + 3 running)")
    print("  5. Quit")
    print()

    choice = input("Choose [1-5]: ").strip()

    if choice == "1":
        task = input("What should I do? > ").strip()
        if not task:
            print("No task given.")
            return
        llm_proc = start_llm_server()
        try:
            from agent.run import run
            run(task, spawn_mcp=True)
        finally:
            cleanup()

    elif choice == "2":
        from mcp_server.server import mcp, _ensure_model
        print(f"Starting GoClick MCP server on http://127.0.0.1:{config.MCP_PORT}")
        _ensure_model()
        mcp.run(transport="sse", host="127.0.0.1", port=config.MCP_PORT)

    elif choice == "3":
        llm_proc = start_llm_server()
        if llm_proc:
            print(f"{config.LLM_BACKEND} LLM server running at http://127.0.0.1:{config.LLM_PORT}/v1")
            print("Press Ctrl+C to stop.")
            try:
                llm_proc.wait()
            except KeyboardInterrupt:
                cleanup()
        else:
            print("Already running. Nothing to do.")

    elif choice == "4":
        if not is_mcp_running():
            print(f"\n  GoClick MCP server not running.")
            print("  Open a terminal and run: python main.py → option 2\n")
            return
        if not is_llm_running():
            print(f"\n  LLM server not running.")
            print("  Open a terminal and run: python main.py → option 3\n")
            return

        task = input("What should I do? > ").strip()
        if not task:
            print("No task given.")
            return
        from agent.run import run
        run(task, spawn_mcp=False)

    elif choice == "5":
        return

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
