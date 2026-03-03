#!/usr/bin/env python3
import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# -----------------------
# File filtering
# -----------------------
DEFAULT_EXTS = [".py", ".md", ".rst"]  # code + docs for repo-level semantics
TEST_HYPERNET = "TEST_HYPERNET"  # test files moved by 2_separate_tests.py
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".tox", ".mypy_cache",
    TEST_HYPERNET,  # do not embed test files
}
SKIP_FILES = {
    "REPO_METADATA.json",  # metadata file, not source
    "QNA_HYPERNET.json",   # Q&A pairs, not source
}
SKIP_FILE_PATTERNS = (".min.",)
MAX_FILE_BYTES_DEFAULT = 2_000_000  # 2MB

# lightweight path heuristics (multiplier-like bias added to score)
# (you can tweak these lists without affecting correctness)
PATH_DOWNWEIGHT = [
    r"(^|/)(__init__\.py)$",
    r"(^|/)(setup\.py)$",
    r"(^|/)(conf\.py)$",
    r"(^|/)(version\.py)$",
    r"(^|/)(constants?\.py)$",
    r"(^|/)(typing.*\.py)$",
]
PATH_UPWEIGHT = [
    r"(^|/)(src/)",
    r"(^|/)(core/)",
    r"(^|/)(lib/)",
    r"(^|/)(README\.(md|rst|txt))$",  # repo-level semantics
    r"(^|/)(CONTRIBUTING\.(md|rst))$",
]

# if file is tiny, strongly reduce its chance to dominate
MIN_LINES_FOR_FULL_WEIGHT = 20


def iter_repo_dirs(repos_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (author, repo_name, repo_path) for repositories/<author>/<repo-name>."""
    for author_dir in sorted([p for p in repos_root.iterdir() if p.is_dir()]):
        author = author_dir.name
        for repo_dir in sorted([p for p in author_dir.iterdir() if p.is_dir()]):
            repo_name = repo_dir.name
            yield author, repo_name, repo_dir


def iter_source_files(
    repo_dir: Path,
    exts: List[str],
    max_files_per_repo: int,
    max_file_bytes: int,
) -> List[Path]:
    files: List[Path] = []
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fn in filenames:
            p = Path(root) / fn
            if p.name in SKIP_FILES:
                continue
            if exts and p.suffix.lower() not in exts:
                continue

            low = p.name.lower()
            if any(pat in low for pat in SKIP_FILE_PATTERNS):
                continue

            try:
                if p.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue

            files.append(p)
            if len(files) >= max_files_per_repo:
                return files
    return files


def read_text_file(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


# -----------------------
# Token chunking
# -----------------------
def chunk_token_ids(token_ids: List[int], chunk_tokens: int, overlap: int) -> List[List[int]]:
    """Produce overlapping token windows."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap >= chunk_tokens:
        raise ValueError("chunk_overlap must be < chunk_tokens")

    chunks = []
    step = chunk_tokens - overlap
    n = len(token_ids)
    if n == 0:
        return chunks

    for start in range(0, n, step):
        end = min(start + chunk_tokens, n)
        window = token_ids[start:end]
        if len(window) < 16:
            continue
        chunks.append(window)
        if end >= n:
            break
    return chunks


def make_text_chunks(tokenizer: AutoTokenizer, text: str, chunk_tokens: int, overlap: int) -> List[str]:
    """Tokenize once, chunk in token-space, decode windows back to text."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    windows = chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=overlap)
    return [tokenizer.decode(w, skip_special_tokens=True) for w in windows]


# -----------------------
# Embedding model wrapper
# -----------------------
@torch.inference_mode()
def embed_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """
    Return embeddings [N, D] using mean pooling over last_hidden_state.
    Note: We DO NOT normalize here; leave that to later (or inside hypernet).
    """
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_vecs.append(mean.detach().cpu())
    if not all_vecs:
        return torch.empty((0, model.config.hidden_size))
    return torch.cat(all_vecs, dim=0)


# -----------------------
# Pooling: chunks -> file -> repo
# -----------------------
def pool_file_embeddings(chunk_embs: torch.Tensor) -> Optional[torch.Tensor]:
    """chunk_embs [K, D] -> file_emb [D]"""
    if chunk_embs.numel() == 0:
        return None
    return chunk_embs.mean(dim=0)


def _path_bonus(rel_path: str) -> float:
    """Small additive bias to file score based on path heuristics."""
    p = rel_path.replace("\\", "/").lower()

    bonus = 0.0
    for pat in PATH_DOWNWEIGHT:
        if re.search(pat, p):
            bonus -= 0.25
            break
    for pat in PATH_UPWEIGHT:
        if re.search(pat, p):
            bonus += 0.15
            break
    return bonus


def _path_depth_bonus(rel_path: str) -> float:
    """Bonus for shallow paths (package layout): root-level and top-level files matter more."""
    p = rel_path.replace("\\", "/").strip("/")
    depth = len([x for x in p.split("/") if x]) if p else 0
    # depth 0 (root) -> +0.2, depth 1 -> +0.1, depth 2+ -> 0
    if depth == 0:
        return 0.2
    if depth == 1:
        return 0.1
    return 0.0


def compute_file_weights(
    file_embs: torch.Tensor,          # [F, D]
    file_token_counts: torch.Tensor,  # [F]
    file_line_counts: torch.Tensor,   # [F]
    file_paths: List[str],
    a_distinct: float,
    b_size: float,
    tau: float,
) -> torch.Tensor:
    """
    Whole-repo, all-files weighting:
      distinct_i = 1 - cos(f_i, mean_f)
      size_i     = normalized log(1+tokens)
      score_i    = a_distinct * distinct_i + b_size * size_i + path_bonus_i + depth_bonus_i + tiny_file_penalty
      w          = softmax(score / tau)

    Returns: w [F] sum=1
    """
    # normalize file embeddings for cosine computations only
    f_norm = F.normalize(file_embs, p=2, dim=-1)  # [F, D]
    mean_f = f_norm.mean(dim=0, keepdim=True)     # [1, D]
    mean_f = F.normalize(mean_f, p=2, dim=-1)

    cos = (f_norm * mean_f).sum(dim=-1).clamp(-1, 1)   # [F]
    distinct = (1.0 - cos)  # higher => more distinctive within repo

    # size feature
    tok = file_token_counts.float().clamp(min=1)
    log_tok = torch.log1p(tok)  # [F]
    # normalize to [0,1]
    if log_tok.numel() > 1:
        lo = log_tok.min()
        hi = log_tok.max()
        size01 = (log_tok - lo) / (hi - lo + 1e-8)
    else:
        size01 = torch.ones_like(log_tok)

    # path bonus (CPU list -> tensor)
    pb = torch.tensor([_path_bonus(p) for p in file_paths], dtype=torch.float32)
    # depth bonus: shallow paths (root, top-level) get higher weight
    depth_bonus = torch.tensor([_path_depth_bonus(p) for p in file_paths], dtype=torch.float32)

    # tiny file penalty: smoothly downweight files with few lines
    # scale in [0..1], where < MIN_LINES_FOR_FULL_WEIGHT gets smaller
    lines = file_line_counts.float().clamp(min=1)
    tiny_scale = (lines / float(MIN_LINES_FOR_FULL_WEIGHT)).clamp(max=1.0)  # [0..1]
    # log-scale to penalize very tiny files more
    tiny_bonus = torch.log(tiny_scale + 1e-6)  # <= 0

    score = a_distinct * distinct.cpu() + b_size * size01.cpu() + pb + depth_bonus + 0.15 * tiny_bonus.cpu()

    # softmax with temperature
    w = torch.softmax(score / max(tau, 1e-6), dim=0)
    return w


def pool_repo_embeddings_weighted(
    file_embs: torch.Tensor,          # [F, D]
    file_token_counts: torch.Tensor,  # [F]
    file_line_counts: torch.Tensor,   # [F]
    file_paths: List[str],
    a_distinct: float,
    b_size: float,
    tau: float,
    alpha_mean: float,
    beta_max: float,
    aggregation: str = "concat",
) -> Optional[torch.Tensor]:
    """
    Aggregate file embeddings into repo vector.
    aggregation: "concat" -> [2D] concat(alpha_mean * wmean, beta_max * vmax). "mean_only" or "max_only" -> [D].
    NO repo-level L2 normalization.
    """
    if file_embs.numel() == 0:
        return None

    # weights computed on CPU for simplicity (file_embs are on CPU already in our pipeline)
    w = compute_file_weights(
        file_embs=file_embs,
        file_token_counts=file_token_counts,
        file_line_counts=file_line_counts,
        file_paths=file_paths,
        a_distinct=a_distinct,
        b_size=b_size,
        tau=tau,
    ).to(file_embs.dtype)

    # weighted mean
    wmean = (file_embs * w.unsqueeze(-1)).sum(dim=0)
    # max over files (captures distinctive anywhere)
    vmax = file_embs.max(dim=0).values

    if aggregation == "mean_only":
        return alpha_mean * wmean
    if aggregation == "max_only":
        return beta_max * vmax
    # concat (default)
    repo_vec = torch.cat([alpha_mean * wmean, beta_max * vmax], dim=0)
    return repo_vec


# -----------------------
# Main pipeline per repo
# -----------------------
def embed_repo(
    repo_dir: Path,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    exts: List[str],
    chunk_tokens: int,
    chunk_overlap: int,
    max_files_per_repo: int,
    max_file_bytes: int,
    batch_size: int,
    # weighting knobs
    a_distinct: float,
    b_size: float,
    tau: float,
    alpha_mean: float,
    beta_max: float,
    aggregation: str = "concat",
) -> Dict:
    files = iter_source_files(repo_dir, exts, max_files_per_repo, max_file_bytes)

    file_vectors = []
    file_meta = []
    file_token_counts = []
    file_line_counts = []

    for fp in files:
        text = read_text_file(fp)
        if not text:
            continue

        rel = str(fp.relative_to(repo_dir)).replace("\\", "/")
        lines = text.splitlines()
        n_lines = len(lines)

        # tokenize once to get token count + chunks
        ids = tokenizer.encode(text, add_special_tokens=False)
        tok_count = len(ids)

        windows = chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=chunk_overlap)
        if not windows:
            continue
        chunks = [tokenizer.decode(w, skip_special_tokens=True) for w in windows]

        chunk_embs = embed_texts(
            model=model,
            tokenizer=tokenizer,
            texts=chunks,
            device=device,
            batch_size=batch_size,
            max_length=chunk_tokens,
        )
        fvec = pool_file_embeddings(chunk_embs)
        if fvec is None:
            continue

        file_vectors.append(fvec)
        file_meta.append(rel)
        file_token_counts.append(tok_count)
        file_line_counts.append(n_lines)

    if not file_vectors:
        return {"ok": False, "reason": "no_files_embedded", "num_files_found": len(files), "num_files_used": 0}

    file_embs = torch.stack(file_vectors, dim=0)  # [F, D]
    tok_t = torch.tensor(file_token_counts, dtype=torch.int64)
    line_t = torch.tensor(file_line_counts, dtype=torch.int64)

    repo_vec = pool_repo_embeddings_weighted(
        file_embs=file_embs,
        file_token_counts=tok_t,
        file_line_counts=line_t,
        file_paths=file_meta,
        a_distinct=a_distinct,
        b_size=b_size,
        tau=tau,
        alpha_mean=alpha_mean,
        beta_max=beta_max,
        aggregation=aggregation,
    )
    if repo_vec is None:
        return {"ok": False, "reason": "pool_failed", "num_files_found": len(files), "num_files_used": len(file_vectors)}

    # Per-file embeddings: list of {path, embedding} for composability experiments
    per_file_embeddings = [
        {"path": file_meta[i], "embedding": file_vectors[i].tolist()}
        for i in range(len(file_vectors))
    ]

    return {
        "ok": True,
        "num_files_found": len(files),
        "num_files_used": len(file_vectors),
        "dim_file": int(file_embs.shape[1]),
        "dim_repo": int(repo_vec.shape[0]),
        "repo_embedding": repo_vec.tolist(),
        "file_embeddings": per_file_embeddings,
        "files_used": file_meta[:200],
        "files_used_count": len(file_meta),
    }


def main():
    ap = argparse.ArgumentParser()
    default_repos = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET", "repositories",
    )
    ap.add_argument("--repos-root", type=str, default=default_repos,
                    help="Root dir with author/repo structure (REPO_DATASET/repositories)")

    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--chunk-tokens", type=int, default=4096,
                    help="Chunk size in tokens; larger captures more context for long files")
    ap.add_argument("--chunk-overlap", type=int, default=512,
                    help="Overlap between chunks")

    ap.add_argument("--max-files-per-repo", type=int, default=4000)
    ap.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES_DEFAULT)

    ap.add_argument("--exts", type=str, default=",".join(DEFAULT_EXTS),
                    help="Comma-separated extensions (e.g. .py,.md). Use '*' or empty for all.")

    # weighting knobs
    ap.add_argument("--a-distinct", type=float, default=1.0, help="weight for distinctiveness score")
    ap.add_argument("--b-size", type=float, default=0.3, help="weight for size score (log tokens)")
    ap.add_argument("--tau", type=float, default=0.15, help="softmax temperature (lower=peakier)")

    ap.add_argument("--alpha-mean", type=float, default=0.6, help="scale for weighted mean half")
    ap.add_argument("--beta-max", type=float, default=1.4, help="scale for max half")
    ap.add_argument("--aggregation", type=str, default="concat",
                    choices=["concat", "mean_only", "max_only"],
                    help="concat=2D (mean+max), mean_only/max_only=1D (smaller)")

    ap.add_argument("--overwrite", action="store_true",
                    help="Re-embed repos that already have embedding in REPO_METADATA.json")

    args = ap.parse_args()

    repos_root = Path(args.repos_root).expanduser().resolve()

    if args.exts.strip() in ("", "*"):
        exts = []
    else:
        exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    torch_dtype = torch.float32
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name, dtype=torch_dtype)
    model.to(device)
    model.eval()

    items = list(iter_repo_dirs(repos_root))
    print(f"Found {len(items)} repos under {repos_root}")

    ok_count = 0
    fail_count = 0

    for author, repo_name, repo_path in tqdm(items, desc="Embedding repos"):
        meta_path = repo_path / "REPO_METADATA.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("embedding") and not args.overwrite:
                    continue
            except (json.JSONDecodeError, OSError):
                pass

        test_hypernet_dir = repo_path / TEST_HYPERNET
        if not test_hypernet_dir.exists() or not test_hypernet_dir.is_dir():
            continue
        if not any(test_hypernet_dir.rglob("*")):
            continue

        result = embed_repo(
            repo_dir=repo_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            exts=exts,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            max_files_per_repo=args.max_files_per_repo,
            max_file_bytes=args.max_file_bytes,
            batch_size=args.batch_size,
            a_distinct=args.a_distinct,
            b_size=args.b_size,
            tau=args.tau,
            alpha_mean=args.alpha_mean,
            beta_max=args.beta_max,
            aggregation=args.aggregation,
        )

        if not result.get("ok", False) or "repo_embedding" not in result:
            fail_count += 1
            continue

        meta_path = repo_path / "REPO_METADATA.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {
                "repo_name": repo_name,
                "repo_full_name": f"{author}/{repo_name}",
                "repo_url": f"https://github.com/{author}/{repo_name}",
            }
        meta["embedding"] = result["repo_embedding"]
        if "file_embeddings" in result:
            meta["file_embeddings"] = result["file_embeddings"]
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        ok_count += 1

    print(f"Done. ok={ok_count}, fail={fail_count}")


if __name__ == "__main__":
    main()
