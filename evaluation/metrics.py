"""
Shared evaluation metrics for RepoPeftBench.
Used by all baselines and the hypernetwork evaluation scripts.
"""

import re
import warnings
from difflib import SequenceMatcher

FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")

_CODEBLEU_AVAILABLE = None
_CODEBLEU_WARNED = False

# Regex tokenizer for Python code (splits on identifiers, operators, literals)
_CODE_TOKEN_RE = re.compile(r"[a-zA-Z_]\w*|0[xXoObB][\da-fA-F_]+|\d[\d_]*\.?\d*[eE]?[\d_]*|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^\s]")


def strip_fim_tokens(s: str) -> str:
    """Remove Qwen FIM special tokens from model output."""
    for tok in FIM_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def strip_comments(s: str) -> str:
    """Remove Python comments (everything after #)."""
    return s.split("#")[0].strip()


def normalize_for_match(s: str) -> str:
    """Normalize string for exact match comparison."""
    s = s.strip().rstrip(":, \t")
    s = s.replace(", ", ",").replace(" ,", ",")
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> bool:
    """Exact match after normalization.

    Callers are expected to pass through postprocess_prediction() first,
    which handles overgeneration truncation (comma, whitespace, newline).
    """
    return normalize_for_match(pred) == normalize_for_match(ref)


def edit_similarity(pred: str, ref: str) -> float:
    """Edit similarity in [0, 1]. 1 = identical."""
    return SequenceMatcher(None, pred, ref).ratio()


def _check_codebleu():
    global _CODEBLEU_AVAILABLE, _CODEBLEU_WARNED
    if _CODEBLEU_AVAILABLE is not None:
        return _CODEBLEU_AVAILABLE
    try:
        from codebleu import calc_codebleu  # noqa: F401
        _CODEBLEU_AVAILABLE = True
    except ImportError:
        _CODEBLEU_AVAILABLE = False
        if not _CODEBLEU_WARNED:
            warnings.warn(
                "codebleu package not installed. Install with: pip install codebleu\n"
                "CodeBLEU scores will be 0.0 until installed.",
                stacklevel=3,
            )
            _CODEBLEU_WARNED = True
    return _CODEBLEU_AVAILABLE


def _tokenize_code(s: str) -> list[str]:
    """Tokenize Python code into identifiers, operators, literals."""
    return _CODE_TOKEN_RE.findall(s)


def _ngram_bleu(pred_tokens: list[str], ref_tokens: list[str], max_n: int = 4) -> float:
    """Compute smoothed BLEU on token lists (no external deps)."""
    from collections import Counter
    import math

    if not pred_tokens or not ref_tokens:
        return 0.0

    brevity = min(1.0, len(pred_tokens) / len(ref_tokens))
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = max(len(pred_tokens) - n + 1, 1)
        # +1 smoothing (Chen & Cherry, 2014)
        precision = (matches + 1) / (total + 1)
        log_bleu += math.log(precision) / max_n

    return brevity * math.exp(log_bleu)


def code_bleu_score(pred: str, ref: str, lang: str = "python") -> float:
    """
    Compute code similarity score.
    Tries codebleu package first; falls back to tokenized BLEU (no external deps).
    """
    if not pred.strip() or not ref.strip():
        return 0.0

    if pred.strip() == ref.strip():
        return 1.0

    if _check_codebleu():
        try:
            from codebleu import calc_codebleu
            result = calc_codebleu([ref], [pred], lang=lang)
            return result["codebleu"]
        except Exception:
            pass

    # Fallback: tokenized BLEU on code tokens
    return _ngram_bleu(_tokenize_code(pred), _tokenize_code(ref))


def postprocess_prediction(pred: str, target: str) -> str:
    """Standard prediction postprocessing pipeline.

    Truncates overgeneration to match the target's token-level format so that
    EM, EditSim, and CodeBLEU are all computed on the same cleaned string.
    """
    pred = strip_fim_tokens(pred)
    if "\n" not in target and "\n" in pred:
        pred = pred.split("\n")[0]
    pred = strip_comments(pred)
    ref = strip_comments(target).strip()
    if ref:
        if "," not in ref and "," in pred:
            pred = pred.split(",")[0].strip()
        if len(ref.split()) == 1 and len(pred.strip().split()) > 1:
            pred = pred.strip().split()[0]
    return pred.strip()


def compute_metrics(pred: str, ref: str) -> dict:
    """Compute all metrics for a single prediction-reference pair."""
    pred_clean = postprocess_prediction(pred, ref)
    ref_clean = strip_comments(ref)
    return {
        "exact_match": exact_match(pred_clean, ref_clean),
        "edit_similarity": edit_similarity(pred_clean, ref_clean),
        "code_bleu": code_bleu_score(pred_clean, ref_clean),
        "pred_clean": pred_clean,
        "ref_clean": ref_clean,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    samples,
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for the mean of ``samples``.

    Parameters
    ----------
    samples:
        Iterable of per-pair scalar metric values (e.g. one EM 0/1 or one
        EditSim 0..1 per QnA pair).
    n_resamples:
        Number of bootstrap resamples. 5000 is the default used everywhere in
        the paper.
    ci:
        Confidence level in (0, 1). Default 0.95 -> 2.5% / 97.5% percentiles.
    seed:
        Resample RNG seed for reproducibility.

    Returns
    -------
    dict with keys ``mean``, ``low``, ``high``, ``n``, ``n_resamples``,
    ``ci``. When ``n == 0`` returns zeros and a flag.
    """
    import numpy as np

    arr = np.asarray(list(samples), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": 0.0,
            "low": 0.0,
            "high": 0.0,
            "n": 0,
            "n_resamples": int(n_resamples),
            "ci": float(ci),
            "empty": True,
        }
    if n_resamples <= 0:
        m = float(arr.mean())
        return {
            "mean": m,
            "low": m,
            "high": m,
            "n": n,
            "n_resamples": 0,
            "ci": float(ci),
            "empty": False,
        }

    rng = np.random.default_rng(seed)
    # Vectorized resampling: sample indices (n_resamples, n) at once.
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    return {
        "mean": float(arr.mean()),
        "low": low,
        "high": high,
        "n": n,
        "n_resamples": int(n_resamples),
        "ci": float(ci),
        "empty": False,
    }


def aggregate_metrics_with_ci(
    records,
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Aggregate per-pair metric dicts (``compute_metrics`` output) into
    mean + bootstrap CI for each of EM, EditSim, CodeBLEU.

    ``records`` may be either a sequence of dicts produced by
    :func:`compute_metrics`, or a sequence of (em, edit, bleu) tuples.
    """
    em, edit, bleu = [], [], []
    for r in records:
        if isinstance(r, dict):
            em.append(1.0 if r.get("exact_match") else 0.0)
            edit.append(float(r.get("edit_similarity", 0.0)))
            bleu.append(float(r.get("code_bleu", 0.0)))
        else:
            a, b, c = r
            em.append(1.0 if a else 0.0)
            edit.append(float(b))
            bleu.append(float(c))

    return {
        "n": len(em),
        "exact_match": bootstrap_ci(em, n_resamples, ci, seed),
        "edit_similarity": bootstrap_ci(edit, n_resamples, ci, seed + 1),
        "code_bleu": bootstrap_ci(bleu, n_resamples, ci, seed + 2),
    }


def format_ci(stat: dict, pct: bool = False) -> str:
    """Pretty-print one bootstrap_ci dict as 'mean [low, high]'.

    Set ``pct=True`` to multiply by 100 (for EM displayed as percent).
    """
    if stat.get("empty"):
        return "n=0"
    scale = 100.0 if pct else 1.0
    return (
        f"{stat['mean'] * scale:.2f} "
        f"[{stat['low'] * scale:.2f}, {stat['high'] * scale:.2f}]"
    )
