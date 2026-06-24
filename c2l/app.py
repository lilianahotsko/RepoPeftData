"""SDK-backed Gradio app for Code2LoRA.

Unlike the standalone HF Space (``hf_space_code2lora/``), this app is a *thin
surface over the c2l SDK*: it calls :class:`c2l.pipeline.AdapterGenerator` to
build a portable adapter, exports it to a standard PEFT directory, and runs
predictions through :class:`c2l.infer.HFInference` -- the exact same code paths
the CLI and API use. Pick a task and an inference backend (including 4-bit /
8-bit for low-VRAM or no-GPU machines).

Run locally::

    pip install -e .[app]
    python -m c2l.app            # http://localhost:7860

Set ``C2L_OFFLINE=1`` (plus locally cached models + checkpoint) for a fully
offline / air-gapped session.
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

from .config import load_config
from .export import export_peft_adapter
from .pipeline import AdapterGenerator
from . import tasks as T

_STATE = {"generator": None, "infer": None, "adapter_dir": None}

EXAMPLE_REPOS = [
    "https://github.com/psf/cachecontrol",
    "https://github.com/john-kurkowski/tldextract",
]


def _generator() -> AdapterGenerator:
    if _STATE["generator"] is None:
        _STATE["generator"] = AdapterGenerator(load_config())
    return _STATE["generator"]


def _qna_choices(result, task: str):
    """Held-out (val/test) examples first; mined for the chosen task."""
    items = []
    for c in result.commits:
        for qi, q in enumerate(c.qnas):
            held = c.in_repo_split in ("val", "test")
            items.append((c.commit_index, qi, q, held))
    items.sort(key=lambda x: (not x[3], x[0]))
    choices = []
    for ci, qi, q, held in items[:200]:
        tag = "held-out" if held else "train"
        snippet = " ".join(q.prefix.strip().splitlines()[-1:])[-48:]
        choices.append((f"[{tag}] {Path(q.test_file).name} L{q.lineno} — …{snippet}",
                        f"{ci}:{qi}"))
    return choices


def process_cb(repo_input: str, task: str, backend: str,
               progress=gr.Progress()):
    def _p(msg, frac):
        try:
            progress(min(max(frac, 0.0), 1.0), desc=msg)
        except Exception:
            pass

    try:
        from .infer import HFInference
        from .pipeline import resolve_repo

        cfg = load_config()
        gen = _generator()
        work = Path(tempfile.gettempdir()) / "c2l_app_repos"
        work.mkdir(parents=True, exist_ok=True)
        repo_dir, repo_id = resolve_repo(repo_input, work, progress=_p)
        adapter = gen.generate(repo_dir, repo_id, task=task, progress=_p)

        out_dir = Path(tempfile.mkdtemp(prefix="c2l_adapter_"))
        export_peft_adapter(adapter, str(out_dir))
        _STATE["adapter_dir"] = str(out_dir)

        quant = None if backend == "hf" else backend
        _STATE["infer"] = HFInference(adapter_dir=str(out_dir),
                                      quantize=quant, config=cfg)

        result = gen.last_result
        choices = _qna_choices(result, task)
        info = (f"<div style='color:#22d3a6'>Adapter ready — fingerprint "
                f"<code>{adapter.fingerprint()}</code>, walked "
                f"{adapter.n_commits_walked} commits, exported to "
                f"<code>{out_dir}</code>. Pick an assertion and predict.</div>")
        return info, gr.update(choices=choices,
                               value=(choices[0][1] if choices else None))
    except Exception as e:
        return (f"<div style='color:#ff6b6b'><b>Failed:</b> {html.escape(str(e))}</div>",
                gr.update(choices=[], value=None))


def predict_cb(selected: Optional[str]):
    if _STATE["infer"] is None:
        return "<div style='color:#ff6b6b'>Generate an adapter first.</div>"
    gen = _STATE["generator"]
    if not selected or gen is None or gen.last_result is None:
        return "<div style='color:#ff6b6b'>Pick an example.</div>"
    try:
        ci, qi = (int(x) for x in selected.split(":"))
        commit = next(c for c in gen.last_result.commits if c.commit_index == ci)
        q = commit.qnas[qi]
        eng = _STATE["infer"]
        base = eng.generate(q.prefix, use_adapter=False)
        adapted = eng.generate(q.prefix, use_adapter=True)
        from .metrics import compute_metrics
        bm = compute_metrics(base, q.target)
        am = compute_metrics(adapted, q.target)

        def block(title, body, ok):
            mark = "✓" if ok else "✗"
            return (f"<div style='flex:1;min-width:220px;background:#161922;"
                    f"border:1px solid #2a2e3a;border-radius:12px;padding:12px'>"
                    f"<div style='color:#8b93a7;font-size:12px'>{title}</div>"
                    f"<div style='font-family:monospace;font-size:15px;color:#e6e6e6'>"
                    f"{html.escape(body or '(empty)')}</div>"
                    f"<div style='font-size:12px;color:#8b93a7'>{mark} match</div></div>")

        return (f"<div style='display:flex;gap:12px;flex-wrap:wrap'>"
                f"{block('Base (no adapter)', bm['pred_clean'], bm['exact_match'])}"
                f"{block('+ Code2LoRA adapter', am['pred_clean'], am['exact_match'])}"
                f"{block('Ground truth', q.target, True)}</div>")
    except Exception as e:
        return f"<div style='color:#ff6b6b'>Prediction failed: {html.escape(str(e))}</div>"


def build_demo():
    with gr.Blocks(title="Code2LoRA (SDK)") as demo:
        gr.Markdown("# Code2LoRA — generate a repo adapter, run it anywhere")
        with gr.Row():
            repo_input = gr.Textbox(label="Git URL or local path", scale=3)
            task = gr.Dropdown(label="Task", choices=T.list_tasks(),
                               value="assert_rhs")
            backend = gr.Dropdown(label="Inference backend",
                                  choices=["hf", "4bit", "8bit"], value="hf")
            process_btn = gr.Button("Generate adapter", variant="primary")
        gr.Examples(EXAMPLE_REPOS, inputs=repo_input)
        status = gr.HTML()
        with gr.Row():
            dd = gr.Dropdown(label="Example to complete", choices=[], scale=3)
            predict_btn = gr.Button("Predict", variant="primary")
        pred = gr.HTML()

        process_btn.click(process_cb, [repo_input, task, backend], [status, dd])
        predict_btn.click(predict_cb, [dd], [pred])
    return demo


def main():
    build_demo().queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
