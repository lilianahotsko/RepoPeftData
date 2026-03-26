"""
Compress oracle context (DRC) to fit within a token budget.

Strategies:
  1. Class compression — keep __init__ full, other methods as signatures
  2. Relevance scoring — prioritize definitions referenced in the prefix
  3. Budget filling — greedily include highest-priority blocks first

Usage:
    from evaluation.compress_context import compress_oracle_context
    compressed = compress_oracle_context(oracle_code, prefix, tokenizer, max_tokens=6000)
"""

import ast
import re
from typing import Optional


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_node_source(source: str, node: ast.AST) -> Optional[str]:
    if hasattr(ast, "get_source_segment"):
        seg = ast.get_source_segment(source, node)
        if seg:
            return seg
    start = getattr(node, "lineno", 1) - 1
    end_line = getattr(node, "end_lineno", start + 1)
    lines = source.splitlines()
    if start < len(lines):
        return "\n".join(lines[start:min(end_line, len(lines))])
    return None


def _signature_line(source: str, node: ast.AST) -> str:
    """Return the 'def ...(args):' line for a function node."""
    lines = source.splitlines()
    start = node.lineno - 1
    # Handle multi-line signatures
    sig_lines = []
    for i in range(start, min(start + 10, len(lines))):
        sig_lines.append(lines[i])
        if ")" in lines[i] and ":" in lines[i]:
            break
    return "\n".join(sig_lines)


def _first_line_docstring(node: ast.AST) -> Optional[str]:
    """Extract first line of docstring from a function/class node."""
    if not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        val = first.value.value
        if isinstance(val, str):
            lines = val.strip().splitlines()
            if not lines:
                return None
            line = lines[0].strip()
            return line
    return None


# ---------------------------------------------------------------------------
# Class compression
# ---------------------------------------------------------------------------

def _compress_class(source: str, cls_node: ast.ClassDef, full_source: str) -> str:
    """Compress a class: keep __init__ + class attrs full, other methods as signatures."""
    parts = []

    # Class header line(s) including decorators
    lines = full_source.splitlines()
    # Add decorators
    for dec in cls_node.decorator_list:
        seg = _get_node_source(full_source, dec)
        if seg:
            parts.append(f"@{seg}")
    # Class def line
    cls_line_idx = cls_node.lineno - 1
    if cls_line_idx < len(lines):
        parts.append(lines[cls_line_idx])

    for node in cls_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("__init__", "__new__", "__post_init__"):
                # Keep initializers in full
                seg = _get_node_source(full_source, node)
                if seg:
                    parts.append(seg)
            else:
                # Signature + ellipsis
                sig = _signature_line(full_source, node)
                doc = _first_line_docstring(node)
                if doc:
                    # Determine indentation
                    indent = "        "
                    parts.append(f"{sig}\n{indent}\"\"\"{doc}\"\"\"\n{indent}...")
                else:
                    parts.append(f"{sig} ...")
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            seg = _get_node_source(full_source, node)
            if seg:
                parts.append(seg)
        elif isinstance(node, ast.ClassDef):
            # Nested class: just the header
            nested_line_idx = node.lineno - 1
            if nested_line_idx < len(lines):
                parts.append(f"{lines[nested_line_idx]} ...")

    return "\n".join(parts)


def compress_definition_block(block_code: str) -> tuple[str, str]:
    """
    Compress a single definition block.

    Returns (full_version, compressed_version).
    For classes: compressed = __init__ full + method signatures.
    For standalone functions: compressed = signature + first-line docstring.
    For other code (variables, constants): unchanged.
    """
    try:
        tree = ast.parse(block_code)
    except SyntaxError:
        return block_code, block_code

    top_nodes = list(ast.iter_child_nodes(tree))
    if not top_nodes:
        return block_code, block_code

    compressed_parts = []
    has_compression = False

    for node in top_nodes:
        if isinstance(node, ast.ClassDef):
            compressed_parts.append(_compress_class(block_code, node, block_code))
            has_compression = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            full_seg = _get_node_source(block_code, node)
            # Only compress functions longer than 5 lines
            if full_seg and full_seg.count("\n") > 5:
                sig = _signature_line(block_code, node)
                doc = _first_line_docstring(node)
                if doc:
                    compressed_parts.append(f"{sig}\n    \"\"\"{doc}\"\"\"\n    ...")
                else:
                    compressed_parts.append(f"{sig} ...")
                has_compression = True
            elif full_seg:
                compressed_parts.append(full_seg)
        else:
            seg = _get_node_source(block_code, node)
            if seg:
                compressed_parts.append(seg)

    if not has_compression:
        return block_code, block_code

    return block_code, "\n\n".join(compressed_parts)


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

# Regex to extract identifiers from Python code
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

# Patterns for function calls: name(...) or Name(...)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# Patterns for attribute access: name.attr
_ATTR_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.")


def _extract_block_names(block_code: str) -> set[str]:
    """Extract the top-level defined names from a block (class/function/variable names)."""
    names = set()
    try:
        tree = ast.parse(block_code)
    except SyntaxError:
        return names
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def score_block(block_code: str, prefix: str) -> float:
    """
    Score how relevant a definition block is to the prefix.

    Returns a float in [0.0, 1.0]:
      1.0 — name appears as a function call in prefix
      0.8 — name appears as attribute base in prefix
      0.5 — name appears as identifier in prefix
      0.1 — name not found in prefix (transitive dependency)
    """
    block_names = _extract_block_names(block_code)
    if not block_names:
        return 0.1

    call_names = set(_CALL_RE.findall(prefix))
    attr_names = set(_ATTR_RE.findall(prefix))
    all_idents = set(_IDENT_RE.findall(prefix))

    best_score = 0.1
    for name in block_names:
        if name in call_names:
            return 1.0  # Direct call — max priority
        if name in attr_names:
            best_score = max(best_score, 0.8)
        elif name in all_idents:
            best_score = max(best_score, 0.5)

    return best_score


# ---------------------------------------------------------------------------
# Main compression API
# ---------------------------------------------------------------------------

def compress_oracle_context(
    oracle_code: str,
    prefix: str,
    tokenizer,
    max_tokens: int = 6000,
) -> str:
    """
    Compress oracle context to fit within *max_tokens*.

    1. Split into definition blocks (separated by "\\n\\n\\n")
    2. Score each block by relevance to the prefix
    3. Sort by score (descending)
    4. Greedily fill budget: try full version, then compressed, else skip

    Args:
        oracle_code: Raw extracted code from v2 cache (blocks separated by \\n\\n\\n).
        prefix: The test prefix for relevance scoring.
        tokenizer: HuggingFace tokenizer for token counting.
        max_tokens: Hard token budget for the output context.

    Returns:
        Compressed oracle context string.
    """
    if not oracle_code or not oracle_code.strip():
        return ""

    raw_blocks = oracle_code.split("\n\n\n")

    # Parse each block: separate header from code, compute full/compressed forms
    scored_blocks = []
    for raw in raw_blocks:
        raw = raw.strip()
        if not raw:
            continue

        # Separate "# Source: ..." header
        lines = raw.split("\n")
        header = ""
        code = raw
        if lines[0].startswith("# Source:"):
            header = lines[0]
            code = "\n".join(lines[1:]).strip()

        if not code:
            continue

        full_code, compressed_code = compress_definition_block(code)
        score = score_block(code, prefix)

        # Re-attach header
        full_text = f"{header}\n{full_code}" if header else full_code
        comp_text = f"{header}\n{compressed_code}" if header else compressed_code

        scored_blocks.append({
            "full": full_text,
            "compressed": comp_text,
            "score": score,
        })

    # Sort by relevance (highest first)
    scored_blocks.sort(key=lambda b: b["score"], reverse=True)

    # Greedily fill budget
    result_parts = []
    budget = max_tokens

    for block in scored_blocks:
        full_tokens = len(tokenizer.encode(block["full"], add_special_tokens=False))

        if full_tokens <= budget:
            result_parts.append(block["full"])
            budget -= full_tokens
        else:
            comp_tokens = len(tokenizer.encode(block["compressed"], add_special_tokens=False))
            if comp_tokens <= budget:
                result_parts.append(block["compressed"])
                budget -= comp_tokens
            # else: skip entirely

    return "\n\n\n".join(result_parts)
