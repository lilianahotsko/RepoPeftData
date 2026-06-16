Repository context is the bottleneck for code LLMs: every completion needs to
know the project's imports, APIs, and conventions, and today we pay for that
context at every single query — through RAG, dependency analysis, or
ever-longer prompts — or we fine-tune a LoRA per repo and watch it go stale on
the next commit.

We ask a simple question: what if the repository *itself* were the prompt, but
only once? **Code2LoRA** is a hypernetwork that reads a repo (or its stream of
commits) and emits a LoRA adapter for a frozen code LLM in a single forward
pass. The repo's knowledge lives in the weights, and inference adds zero extra
tokens.

It comes in two flavors. **Code2LoRA-Static** snapshots a repo into an adapter
and, with no per-repo training, already matches the per-repository LoRA upper
bound on in-repo eval and beats RAG / dependency-resolved-context / FFT+RAG
cross-repo (+9.9pp EM). **Code2LoRA-Evo** is the part we're most excited about:
a GRU walks the commit history and refreshes the adapter in O(1) per commit, so
the model keeps up with active development instead of fighting it. On a
strictly post-cutoff 92-repo OOD holdout — repos the encoder has never seen —
Code2LoRA-Evo lifts a Qwen2.5-Coder backbone from 44.6% to 74.1% EM.

To make all of this measurable we also release **RepoPeftBench**: 604 Python
repos, 62K static and 400K commit-derived assertion-completion tasks, with
in-repo, cross-repo, and temporal-OOD splits. We hope it's useful as a
benchmark for repo-conditioned and evolution-aware code modeling beyond our own
setup.

Code: https://anonymous.4open.science/r/code2lora-6857 · Data & models:
https://huggingface.co/code2lora
