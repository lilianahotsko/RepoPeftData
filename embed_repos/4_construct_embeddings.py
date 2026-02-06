
#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


DEFAULT_EXTS = [".py"]
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".tox", ".mypy_cache",
}

SKIP_FILE_PATTERNS = (
    ".min.", 
)

MAX_FILE_BYTES_DEFAULT = 2_000_000 

def iter_repo_dirs(repos_root):
    for author_dir in sorted([p for p in repos_root.iterdir() if p.is_dir()]):
        author = author_dir.name
        for repo_dir in sorted([p for p in author_dir.iterdir() if p.is_dir()]):
            repo_name = repo_dir.name
            yield author, repo_name, repo_dir


def iter_source_files(repo_dir, exts, max_files_per_repo, max_file_bytes):
    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fn in filenames:
            p = Path(root) / fn
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
            # if len(files) >= max_files_per_repo:
            #     return files
    return files


def read_text_file(p):
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None



def chunk_token_ids(token_ids, chunk_tokens, overlap):

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
        if len(window) < 8:
            continue
        chunks.append(window)
        if end >= n:
            break
    return chunks


def make_text_chunks(tokenizer, text, chunk_tokens, overlap):
    ids = tokenizer.encode(text, add_special_tokens=False)
    windows = chunk_token_ids(ids, chunk_tokens=chunk_tokens, overlap=overlap)
    return [tokenizer.decode(w, skip_special_tokens=True) for w in windows]


@torch.inference_mode()
def embed_texts(model, tokenizer, texts, device, batch_size, max_length):
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
        masked = last * mask
        summed = masked.sum(dim=1)  # [B, H]
        denom = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        mean = summed / denom
        all_vecs.append(mean.detach().cpu())
    return torch.cat(all_vecs, dim=0) if all_vecs else torch.empty((0, model.config.hidden_size))


def l2_normalize(x: torch.Tensor, eps: float = 1e-12):
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=eps))


def pool_file_embeddings(chunk_embs: torch.Tensor) -> Optional[torch.Tensor]:
    """
    chunk_embs: [K, D]
    returns [D]
    """
    if chunk_embs.numel() == 0:
        return None
    return chunk_embs.mean(dim=0)


def pool_repo_embeddings(file_embs: torch.Tensor) -> Optional[torch.Tensor]:
    """
    file_embs: [F, D]
    returns [2D] = concat(mean_pool_files, max_pool_files) then L2 normalize
    """
    if file_embs.numel() == 0:
        return None
    mean_pool = file_embs.mean(dim=0)
    max_pool = file_embs.max(dim=0).values
    repo_vec = torch.cat([mean_pool, max_pool], dim=0)
    return l2_normalize(repo_vec)

def embed_repo(repo_dir, model, tokenizer, device, exts, chunk_tokens, 
    chunk_overlap, max_files_per_repo, max_file_bytes, batch_size):
    files = iter_source_files(repo_dir, exts, max_files_per_repo, max_file_bytes)
    file_vectors = []
    file_meta = []
    for fp in files:
        text = read_text_file(fp)
        if not text:
            continue

        chunks = make_text_chunks(tokenizer, text, chunk_tokens=chunk_tokens, overlap=chunk_overlap)
        if not chunks:
            continue
        chunk_embs = embed_texts(model, tokenizer, chunks, device, batch_size, chunk_tokens)
        fvec = pool_file_embeddings(chunk_embs)
        if fvec is None:
            continue

        file_vectors.append(fvec)
        file_meta.append(str(fp.relative_to(repo_dir)).replace("\\", "/"))

    if not file_vectors:
        return {
            "ok": False,
            "reason": "no_files_embedded",
            "num_files_found": len(files),
            "num_files_used": 0,
        }

    file_embs = torch.stack(file_vectors, dim=0)  # [F, D]
    repo_vec = pool_repo_embeddings(file_embs)
    if repo_vec is None:
        return {
            "ok": False,
            "reason": "pool_failed",
            "num_files_found": len(files),
            "num_files_used": len(file_vectors),
        }

    return {
        "ok": True,
        "num_files_found": len(files),
        "num_files_used": len(file_vectors),
        "dim_file": int(file_embs.shape[1]),
        "dim_repo": int(repo_vec.shape[0]),
        "repo_embedding": repo_vec.tolist(),  # [2D]
        "files_used": file_meta[:200], 
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--chunk-overlap", type=int, default=256)
    ap.add_argument("--max-files-per-repo", type=int, default=2000)
    ap.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES_DEFAULT)
    ap.add_argument("--exts", type=str, default="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-pt", action="store_true")

    args = ap.parse_args()
    repos_root = Path(args.repos_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.exts.strip() == "*" or not args.exts.strip():
        exts = []
    else:
        exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    items = list(iter_repo_dirs(repos_root))
    print(f"Found {len(items)} repos under {repos_root}")

    ok_count = 0
    fail_count = 0

    for author, repo_name, repo_path in tqdm(items, desc="Embedding repos"):
        out_name = f"{author}__{repo_name}"
        out_json = out_dir / f"{out_name}.json"
        out_pt = out_dir / f"{out_name}.pt"

        if out_json.exists() and (not args.overwrite):
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
        )

        payload = {
            "author": author,
            "repo": repo_name,
            "repo_path": str(repo_path),
            "model_name": args.model_name,
            "chunk_tokens": args.chunk_tokens,
            "chunk_overlap": args.chunk_overlap,
            "max_files_per_repo": args.max_files_per_repo,
            "max_file_bytes": args.max_file_bytes,
            **result,
        }

        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if args.save_pt and payload.get("ok", False):
            t = torch.tensor(payload["repo_embedding"], dtype=torch.float32)
            torch.save({"repo_embedding": t, "meta": {"author": author, "repo": repo_name}}, out_pt)

        if payload.get("ok", False):
            ok_count += 1
        else:
            fail_count += 1

    print(f"Done. ok={ok_count}, fail={fail_count}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
