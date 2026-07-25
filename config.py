import os

PROJECT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR

# LLM
LLM_BACKEND = "mlx"  # "mlx" or "llama_cpp"
LLM_MODEL_MLX = "mlx-community/Qwen3-VL-2B-Instruct-4bit"
LLM_MODEL_GGUF = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
LLM_MODEL_GGUF_FILE = "Qwen3-VL-2B-Instruct-Q4_K_M.gguf"
LLM_PORT = 8111
LLM_BASE_URL = f"http://127.0.0.1:{LLM_PORT}/v1"

# GoClick
GOCLICK_MODEL = "HongxinLi/GoClick-Base"

# GoClick MCP
MCP_PORT = 8222
MCP_URL = f"http://127.0.0.1:{MCP_PORT}/sse"

# Agent
MAX_STEPS = 20
