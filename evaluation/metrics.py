"""
Shared evaluation metrics for RepoPeftBench.
Used by all baselines and the hypernetwork evaluation scripts.
"""

import warnings
from difflib import SequenceMatcher

FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")

_CODEBLEU_AVAILABLE = None
_CODEBLEU_WARNED = False


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


def _pred_candidates(pred: str, ref: str) -> list[str]:
    """Return candidate pred strings to try for relaxed match."""
    candidates = [normalize_for_match(pred)]
    if "," not in ref and "," in pred:
        candidates.append(normalize_for_match(pred.split(",")[0]))
    if len(ref.split()) == 1 and " " in pred:
        candidates.append(normalize_for_match(pred.split()[0]))
    return candidates


def exact_match(pred: str, ref: str) -> bool:
    """Exact match with relaxed postprocessing for common model overgeneration."""
    norm_ref = normalize_for_match(ref)
    return any(c == norm_ref for c in _pred_candidates(pred, ref))


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


def code_bleu_score(pred: str, ref: str, lang: str = "python") -> float:
    """Compute CodeBLEU. Returns 0.0 if codebleu package is unavailable."""
    if not _check_codebleu():
        return 0.0
    try:
        from codebleu import calc_codebleu
        if not pred.strip() or not ref.strip():
            return 0.0
        result = calc_codebleu([ref], [pred], lang=lang)
        return result["codebleu"]
    except Exception as e:
        warnings.warn(f"CodeBLEU computation failed: {e}", stacklevel=2)
        return 0.0


def postprocess_prediction(pred: str, target: str) -> str:
    """Standard prediction postprocessing pipeline."""
    pred = strip_fim_tokens(pred)
    if "\n" not in target and "\n" in pred:
        pred = pred.split("\n")[0]
    return strip_comments(pred)


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
