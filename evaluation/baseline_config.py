"""
Shared defaults for fair baseline comparison.
Import these in baseline scripts to reduce drift over time.
"""

DEFAULT_MAX_INPUT_TOKENS = 16384
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Code2LoRA-GRU defaults
DEFAULT_GRU_HIDDEN_DIM = 1024
DEFAULT_GRU_NUM_LAYERS = 1
DEFAULT_GRU_BPTT_WINDOW = 32
DEFAULT_GRU_INIT_TYPE = "mamba2"
DEFAULT_PREAMBLE_FRAC = 0.1
DEFAULT_MAX_FILES_PER_REPO = 512
