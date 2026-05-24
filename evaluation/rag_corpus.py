"""AST-level RAG corpus units and hybrid retrieval (WeChat-style).

Inspired by fine-grained function/class indexing for code completion RAG
(see https://arxiv.org/pdf/2507.18515): corpus units are whole Python
functions and classes (never mid-definition token windows), retrieval fuses
lexical BM25 with dense embedding similarity, and injected context is
relevance-compressed to a token budget before prompt assembly.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from evaluation.compress_context import compress_oracle_context

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+")


def tokenize_for_bm25(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _get_source_segment(source: str, node: ast.AST) -> Optional[str]:
    if hasattr(ast, "get_source_segment"):
        seg = ast.get_source_segment(source, node)
        if seg:
            return seg
    lines = source.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    if start < len(lines):
        return "\n".join(lines[start:min(end, len(lines))])
    return None


def _symbol_name(node: ast.AST, parent: Optional[str] = None) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        name = node.name
        return f"{parent}.{name}" if parent else name
    return parent or ""


def extract_ast_chunks(
    rel_path: str,
    source: str,
    *,
    max_chunk_chars: int = 12_000,
) -> List[Dict[str, str]]:
    """Return semantic chunks: one per top-level function or class."""
    source = source or ""
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        text = source.strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + "\n# ... truncated ..."
        return [{
            "text": f"# File: {rel_path}\n# Symbol: <module>\n{text}",
            "rel_path": rel_path,
            "symbol": "<module>",
        }]

    chunks: List[Dict[str, str]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        seg = _get_source_segment(source, node)
        if not seg or not seg.strip():
            continue
        sym = _symbol_name(node)
        if len(seg) > max_chunk_chars:
            seg = seg[:max_chunk_chars] + "\n# ... truncated ..."
        chunks.append({
            "text": f"# File: {rel_path}\n# Symbol: {sym}\n{seg}",
            "rel_path": rel_path,
            "symbol": sym,
        })
    return chunks


class SimpleBM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, corpus_tokens: List[List[str]], *, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus_tokens
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        lens = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(lens) / self.N) if self.N else 0.0
        self.doc_freq: Dict[str, int] = defaultdict(int)
        for doc in corpus_tokens:
            for term in set(doc):
                self.doc_freq[term] += 1

    def score(self, query_tokens: List[str]) -> List[float]:
        if not self.corpus:
            return []
        scores = [0.0] * self.N
        if not query_tokens:
            return scores
        for i, doc in enumerate(self.corpus):
            dl = len(doc) or 1
            tf_map = Counter(doc)
            for q in query_tokens:
                df = self.doc_freq.get(q, 0)
                if df == 0:
                    continue
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map.get(q, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-9)
        return scores


@torch.inference_mode()
def embed_texts_mean_pool(
    model,
    tokenizer,
    texts: Sequence[str],
    device: str,
    *,
    batch_size: int = 32,
    max_length: int = 512,
) -> torch.Tensor:
    if not texts:
        return torch.empty(0)
    out_chunks: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out_chunks.append(pooled.float().cpu())
    return torch.cat(out_chunks, dim=0)


def build_index_payload(
    chunk_texts: List[str],
    embeddings: torch.Tensor,
    *,
    repo_id: str = "",
    commit_sha: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    doc_tokens = [tokenize_for_bm25(t) for t in chunk_texts]
    payload: Dict[str, Any] = {
        "version": "ast_hybrid_v1",
        "chunk_mode": "ast",
        "chunks": chunk_texts,
        "embeddings": F.normalize(embeddings.float(), p=2, dim=-1).to(torch.float16)
        if embeddings.numel() else None,
        "doc_tokens": doc_tokens,
        "repo": repo_id,
        "commit_sha": commit_sha,
        "n_chunks": len(chunk_texts),
    }
    if extra:
        payload.update(extra)
    return payload


def load_rag_index(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a cache file into the structure used at retrieval time."""
    chunks = list(data.get("chunks") or [])
    embs = data.get("embeddings")
    if embs is not None and not isinstance(embs, torch.Tensor):
        embs = torch.as_tensor(embs)
    if embs is not None:
        embs = embs.float()
    doc_tokens = data.get("doc_tokens")
    if doc_tokens is None and chunks:
        doc_tokens = [tokenize_for_bm25(t) for t in chunks]
    bm25 = SimpleBM25(doc_tokens) if doc_tokens else None
    return {
        "chunks": chunks,
        "embeddings": embs,
        "doc_tokens": doc_tokens,
        "bm25": bm25,
        "version": data.get("version", "legacy"),
        "chunk_mode": data.get("chunk_mode", "token"),
    }


def hybrid_retrieve_topk(
    query_text: str,
    query_emb: torch.Tensor,
    index: Dict[str, Any],
    top_k: int,
    *,
    candidate_k: int = 30,
    rrf_k: int = 60,
) -> List[str]:
    """Fuse dense cosine similarity with BM25 via reciprocal rank fusion."""
    chunks = index.get("chunks") or []
    if not chunks:
        return []
    n = len(chunks)
    k_out = min(top_k, n)
    cand = min(candidate_k, n)

    dense_ranks: Dict[int, int] = {}
    embs = index.get("embeddings")
    if embs is not None and embs.numel() > 0:
        sims = (query_emb.float() @ embs.T).squeeze(0)
        _, top_idx = sims.topk(cand)
        for rank, idx in enumerate(top_idx.tolist()):
            dense_ranks[int(idx)] = rank

    bm25_ranks: Dict[int, int] = {}
    bm25 = index.get("bm25")
    if bm25 is not None:
        qtok = tokenize_for_bm25(query_text)
        scores = bm25.score(qtok)
        if scores:
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:cand]
            for rank, idx in enumerate(order):
                bm25_ranks[idx] = rank

    if not dense_ranks and not bm25_ranks:
        return []

    fused: Dict[int, float] = defaultdict(float)
    for idx, rank in dense_ranks.items():
        fused[idx] += 1.0 / (rrf_k + rank + 1)
    for idx, rank in bm25_ranks.items():
        fused[idx] += 1.0 / (rrf_k + rank + 1)

    best = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k_out]
    return [chunks[i] for i, _ in best]


def compress_retrieved_chunks(
    chunks: List[str],
    prefix: str,
    tokenizer,
    max_tokens: int,
) -> str:
    if not chunks:
        return ""
    joined = "\n\n\n".join(chunks)
    return compress_oracle_context(joined, prefix, tokenizer, max_tokens=max_tokens)


def format_rag_prompt(prefix: str, context: str) -> str:
    if not context.strip():
        return prefix
    return context + "\n\n\n" + prefix
