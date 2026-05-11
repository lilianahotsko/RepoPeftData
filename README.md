# RepoPeftData

# Dataset Collection

## Repositories collection:

Repositories with pytest methods, filtered by: 
- number of stars (6+, 20+, 50+) 
- date of the last update (2025+)
- no forks
- size(100kb-200mb)

## Splitting files into 2 groups:
- Test files (where pytest was imported / name of the file is test)
- The rest of the files 

## Creating the QnA pairs:
1. code completion:
    - selecting the cutting line, predicting the next line 

2. code execution:
    - selecting the cutting point after "assert = " to predict the value of the variable 

## Creating the Embeddings:

1. withing the file: 
    - chunking (size: 4k, with overlap 256)
    - embedding the chunks in batches
    - mean pooling of the chunks of the file + masked by attention mask [B, T, H]
    - -> [B, H] embeddings

2. within the repo:
    -  weighting the file by the size
    - weighted mean 
    - max pool
    - concatenation of mean + max -> [2xH]

## Training:

embeding + qnas -> hypernetwork -> LoRA -> Injected to Qwen -> loss 

## Evaluation:

1. code execution: 
    - exact match (assert == ....)
    - self.assertqual(...) - look into the documentatiojn 
    - calculate the distribution og the qna per repo 
    

2. test coverage:

test dataset will look like:
    - input: embedding + file/function to test + current written files + current coverage score(?)
    - target: changes of the test file 
    - result: coverage

for that, initially need to check:
- test pass rate (especially in the target lines, if the training tests are actually passing)
- what is the test coverage rate currently in the separated files (should I filter by 70%?)


# DATASET STATS

[Distribution]
  total_repos: 560
  total_qnas: 73267
  n_qnas: min=30 max=200 mean=130.8
  size_bytes: min=98580 max=119542181 mean=5101419

[Repo split] train=447 val=55 test=58
  Wrote /scratch/lhotsko/REPO_DATASET/train.json (447 repos, 46313 pairs) 80 80
  Wrote /scratch/lhotsko/REPO_DATASET/ir_val.json (447 repos, 5679 pairs) 10 
  Wrote /scratch/lhotsko/REPO_DATASET/ir_test.json (447 repos, 6046 pairs) 10 
  Wrote /scratch/lhotsko/REPO_DATASET/cr_val.json (55 repos, 7609 pairs) 10
  Wrote /scratch/lhotsko/REPO_DATASET/cr_test.json (58 repos, 7620 pairs) 10

train: 447 valid repos, 46313 valid QNA pairs
cr_val: 55 valid repos, 7609 valid QNA pairs
cr_test: 58 valid repos, 7620 valid QNA pairs

Qwen Pretrained: CR 

LoRA per repo : IR 


# QWEN PRETRAINED:

head -n 50 $SCRATCH/BASELINES/qwen_full.json (on cr_test.json)
{ 
  "exact_match_pct": 34.63254593175853,
  "exact_match_count": 2639,
  "n": 7620,
  "n_total": 7620,
  "edit_similarity": 0.5302412674736591,
}



# HYPERNETWORK:

============================================================
Results on cr_test.json
============================================================
  Exact Match:     58.02% (4421/7620)
  Edit Similarity: 0.7921

## License

MIT License - feel free to use and modify for your research needs.



# Notes:

remove the tests which start with the ,




  Repos processed: 726
  Total extracted: 441658
  Total selected:  65239
  Imports present: 64810/65239 (99.3%)
  Properly indented: 65239/65239 (100.0%)

  Difficulty distribution:
    numeric_literal: 14851 (22.8%)
    variable: 14222 (21.8%)
    string_literal: 11797 (18.1%)
    collection: 7376 (11.3%)
    complex_expr: 7168 (11.0%)
    func_call: 5909 (9.1%)
    none_literal: 2219 (3.4%)
    bool_literal: 1697 (2.6%)

  Assertion types:
    assert: 48987 (75.1%)
    self.assertEqual: 5638 (8.6%)
    assert_*: 4835 (7.4%)
    pytest.raises: 2615 (4.0%)
    self.assertTrue: 729 (1.1%)
    self.assertIn: 400 (0.6%)
    self.assertRaises: 393 (0.6%)
    self.assertFalse: 267 (0.4%)
    self.assertIsInstance: 265 (0.4%)
    self.assertIsNotNone: 169 (0.3%)
    pytest.approx: 168 (0.3%)
    self.assertIsNone: 134 (0.2%)
    self.assertListEqual: 83 (0.1%)
    self.assertNotEqual: 75 (0.1%)
    self.assertIs: 73 (0.1%)
    self.assertGreater: 61 (0.1%)
    self.assertSequenceEqual: 56 (0.1%)
    self.assertNotIn: 53 (0.1%)
    self.assertAlmostEqual: 52 (0.1%)
    self.assertLess: 27 (0.0%)
    self.assertRaisesRegex: 25 (0.0%)
    self.assertDictEqual: 24 (0.0%)
    self.assertGreaterEqual: 22 (0.0%)
    self.assertRegex: 21 (0.0%)
    self.assertIsNot: 20 (0.0%)
    self.assertLessEqual: 19 (0.0%)
    self.assertCountEqual: 18 (0.0%)
    self.assertMultiLineEqual: 4 (0.0%)
    self.assertSetEqual: 2 (0.0%)
    self.assertNotIsInstance: 2 (0.0%)
    self.assertNotAlmostEqual: 1 (0.0%)
    self.assertLogs: 1 (0.0%)
[lhotsko@g23.nibi RepoPeftData]$ 




Total pairs (after comma filter): 39612
Skipped (comma-leading): 0
Pairs with oracle context: 34723/39612 (87.7%)

=== PREFIX ONLY (tokens) ===
  min=10  max=19281  mean=357  median=221
  >  512:  6594 (16.6%)
  > 1024:  1922 (4.9%)
  > 2048:   558 (1.4%)
  > 4096:   140 (0.4%)
  > 8192:    28 (0.1%)
  >16384:     2 (0.0%)

=== PREFIX + ORACLE CONTEXT (tokens) ===
  min=10  max=576153  mean=2975  median=1160
  >  512: 28979 (73.2%)
  > 1024: 21187 (53.5%)
  > 2048: 12962 (32.7%)
  > 4096:  6632 (16.7%)
  > 8192:  2615 (6.6%)
  >16384:   821 (2.1%)

=== ORACLE CONTEXT ONLY (tokens, when present) ===
  min=12  max=574933  mean=2985  median=1074
  >  256: 28691 (82.6%)
  >  512: 23441 (67.5%)
  > 1024: 17758 (51.1%)
  > 2048: 11054 (31.8%)
  > 4096:  5764 (16.6%)
  > 8192:  2332 (6.7%)

=== TARGET (tokens) ===
  min=1  max=249  mean=5  median=3
  >   16:  1557 (3.9%)
  >   32:   628 (1.6%)
  >   64:   192 (0.5%)
  >  128:    39 (0.1%)
  >  256:     0 (0.0%)

=== FULL TRAINING SEQUENCE: prefix+oracle+target (tokens) ===
  min=12  max=576160  mean=2980  median=1164
  >  512: 29062 (73.4%)
  > 1024: 21231 (53.6%)
  > 2048: 12981 (32.8%)
  > 4096:  6643 (16.8%)
  > 8192:  2618 (6.6%)
  >16384:   822 (2.1%)
  >32768:   251 (0.6%)


  + FFT + RAG 
  + related work -  Text2 Lora baseline , Doc2Lora, 
  + merge table 1 and the ending of the 2, move the rest form the t2 to appendix
  + t2 to appendix
  + table5: remove the k
  + remove table 5 and check the fig 3 (how many repos were used for these) 
  + live code bench - analysis of the errors 
  + table 6: make better examples like in text2lora + visualize the LoRA (like a heatmap ...) - to show that fft+drc is worse 
  + add the distribution of the prefix with and without drc  (based on this - change the context length of the hypernet or lora/fft) 
  + change chunk length for rag to 500 





+ cap the drc at 6k
+ memory issue on the LoRA
+ unify everything to 8k
+ perrepo Lora+ DRC
+ add the real results from the text-to-lora
+ full repo sizes
+ in table 1: add the last 3 rows into the 3 columns  (avg prefix, repo-full and target)


---

# QnA Pair Visualization — Prefix Only vs. Prefix + DRC v3

## Example 1: `N-Wouda/ALNS` — `test_late_acceptance_hill_climbing.py::11`

**Target:** `(ValueError, TypeError))`

### Setup A: Prefix Only (Pretrained / Single-LoRA / FFT)

```python
import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import One, Two, Zero

@mark.parametrize("lookback_period", [-0.01, -10, 1.5])
def test_raises_invalid_lookback_period(lookback_period):

    with assert_raises(
        ▶ ???
```

> The model sees the import and test parameters but has **no access** to `LateAcceptanceHillClimbing`'s source.
> It must guess that negative values raise `ValueError` *and* that `1.5` raises `TypeError`.

### Setup B: Prefix + DRC v3 (Oracle Context)

```python
# ── DRC v3 Context (2500 chars, compression_ratio=1.0) ────────────────
# Source: alns/accept/LateAcceptanceHillClimbing.py
class LateAcceptanceHillClimbing:
    """
    The Late Acceptance Hill Climbing (LAHC) criterion accepts a candidate
    solution when it is better than the current solution from a number of
    iterations ago.
    ...
    Parameters
    ----------
    lookback_period: int
        Non-negative integer specifying which solution to compare against
        for late acceptance. ...
    greedy: bool
        ...
    better_history: bool
        ...
    """

    def __init__(
        self,
        lookback_period: int,
        greedy: bool = False,
        better_history: bool = False,
    ):
        self._lookback_period = lookback_period
        ...

        if lookback_period < 0:
            raise ValueError("lookback_period must be a non-negative integer.")

        self._history: deque = deque([], maxlen=lookback_period)
        # ^^^ maxlen=1.5 will raise TypeError
    ...
# ── End DRC v3 ─────────────────────────────────────────────────────────


# ── Prefix ─────────────────────────────────────────────────────────────
import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import One, Two, Zero

@mark.parametrize("lookback_period", [-0.01, -10, 1.5])
def test_raises_invalid_lookback_period(lookback_period):

    with assert_raises(
        ▶ (ValueError, TypeError))
```

> With DRC v3, the model sees: `raise ValueError(...)` for negative values,
> and `deque([], maxlen=lookback_period)` where `maxlen=1.5` triggers `TypeError`.
> The correct tuple `(ValueError, TypeError)` becomes inferable.

---

## Example 2: `QuixiAI/Hexis` — `test_brave_search.py::31`

**Target:** `ToolCategory.WEB`

### Setup A: Prefix Only

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.tools.base import ToolCategory, ToolContext, ToolErrorType, ToolExecutionContext
from core.tools.brave_search import (
    BraveSearchHandler,
    create_brave_search_tools,
)

def _make_context():
    registry = MagicMock()
    registry.pool = MagicMock()
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="test-call",
        registry=registry,
    )

class TestBraveSearchSpec:

    def test_spec_category_is_web(self):

        assert BraveSearchHandler().spec.category ==
            ▶ ???
```

> The model sees `ToolCategory` is imported but has **no access** to the enum values
> or to `BraveSearchHandler.spec` to know it returns `ToolCategory.WEB`.

### Setup B: Prefix + DRC v3 (3711 chars, compression_ratio=1.0)

```python
# ── DRC v3 Context ─────────────────────────────────────────────────────
# Source: core/tools/base.py
class ToolCategory(str, Enum):
    """Categories of tools for organization and policy."""
    MEMORY = "memory"
    WEB = "web"
    FILESYSTEM = "filesystem"
    SHELL = "shell"
    CODE = "code"
    BROWSER = "browser"
    ...

# Source: core/tools/brave_search.py
class BraveSearchHandler(ToolHandler):
    """Search the web using the Brave Search API."""
    ...

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="brave_search",
            description="Search the web using Brave Search. ...",
            parameters={...},
            category=ToolCategory.WEB,      # ◀ answer visible here
            energy_cost=2,
            is_read_only=True,
            optional=True,
        )
    ...
# ── End DRC v3 ─────────────────────────────────────────────────────────


# ── Prefix ─────────────────────────────────────────────────────────────
...
    def test_spec_category_is_web(self):
        assert BraveSearchHandler().spec.category ==
            ▶ ToolCategory.WEB
```

> The DRC v3 context provides both the `ToolCategory` enum definition and
> `BraveSearchHandler.spec` property with `category=ToolCategory.WEB` — the
> answer is directly extractable from the resolved source code.

---

## Example 3: `agronholm/anyio` — `test_deprecations.py::11`

**Target:** `anyio.BrokenWorkerInterpreter`

### Setup A: Prefix Only

```python
from __future__ import annotations

import pytest

import anyio

def test_broken_worker_interpreter_deprecation() -> None:
    with pytest.warns(DeprecationWarning):
        DeprecatedClass = anyio.BrokenWorkerIntepreter  # note: typo

    assert DeprecatedClass is
        ▶ ???
```

> The model must figure out that `BrokenWorkerIntepreter` (typo) is a deprecated
> alias that points to `BrokenWorkerInterpreter` (correct spelling).
> Without source access, the model has no way to know the target class name.

### Setup B: Prefix + DRC v3 (402 chars, compression_ratio=1.0)

```python
# ── DRC v3 Context ─────────────────────────────────────────────────────
# Source: src/anyio/__init__.py
for __value in list(locals().values()):
    if getattr(__value, "__module__", "").startswith("anyio."):
        __value.__module__ = __name__
del __value

def __getattr__(attr: str) -> type[BrokenWorkerInterpreter]:
    """Support deprecated aliases."""
    if attr == "BrokenWorkerIntepreter":
        ...
# ── End DRC v3 ─────────────────────────────────────────────────────────


# ── Prefix ─────────────────────────────────────────────────────────────
...
    assert DeprecatedClass is
        ▶ anyio.BrokenWorkerInterpreter
```

> The DRC v3 shows `__getattr__` handling the typo alias, with
> return type `type[BrokenWorkerInterpreter]`. The mapping from
> deprecated `BrokenWorkerIntepreter` → `BrokenWorkerInterpreter` is clear.

---

### Key Takeaways

| Example | Prefix only | + DRC v3 | Why DRC helps |
|---------|-------------|----------|---------------|
| ALNS `assert_raises(...)` | Must guess error types | Sees `raise ValueError` + `deque(maxlen=...)` | Constructor validation logic visible |
| Hexis `spec.category ==` | Knows `ToolCategory` exists, not its value | Sees `category=ToolCategory.WEB` in spec | Property return value directly visible |
| anyio `DeprecatedClass is` | Typo alias, no way to know correct name | Sees `__getattr__` mapping to `BrokenWorkerInterpreter` | Deprecation alias mapping visible |




+ text2lora variant
+ text2 lora and doc2lora archytecture details (in related works) 
+ explain difference between the results (why code2lora is better than text2lora L variant with our embedding)
+ for a long file - create separate loras and apply sequentially (incrementally compute the pooling of the embedding to compute the new weighted average) 

+ in the loop: adding the new files after each commit (befre the ml trunk) 
+ split 80/20 commits instead of the 80/20 test 


repository: 80(train)-20(test set)(cr)

test file level: train 80% -> 80(train)/20 (test)(ir) 


- for each commit update the gru -> generate LoRA -> compute loss on the current commit -> go to the next commit 

- add the statistics : how many version are in the each versions / how many token are in the diffs 



Used in train:

=== Lines per unified diff (production_code_diff) ===
  min:    0
  max:    216274
  mean:   200.12
  median: 23.00
  std:    1888.72
  p90:    293.0
  p95:    615.0
  p99:    2667.5

=== Tokens per unified diff (production_code_diff) ===
  tokenizer: hf:Qwen/Qwen3-Embedding-0.6B
  min:    0
  max:    3814966
  mean:   2246.65
  median: 277.00
  std:    30734.44
  p90:    3025.5
  p95:    6340.8
  p99:    27842.8



- use the latest assertions only 
- evaluation also using just the assertions 
- create the file level split for the QnAs in train and val - for consistency 
- keep the complete prefix from the test file before the cutting line 
- change the diffs generation- use just the diffs between the commits that have assertions 
- after the commit split - 

Aq1: traning on this commit structure 
Aq2: accuracy getting lower after many commits 
Aq3 : also using file qna split 



- update the stats for the papaer table 
- define which commits are kept (new qnas for commit -> added and updated qnas)

- each repo - y/x - commits - which have the test files in there  (number of qnas - y)
1) for each repo have the separate figure of how many test qnas
2) instead of the normalized timeline - use the real one 

3) new image - all the repos with dots(changing size based on how many qnas are in the commits)
- each commit => 0 test files 





- collect extra repos with start data from the 2025+ (and are not the fork of any other repos that are already in the training set) - out-of-distribution set 


# Camera-Ready Submission Notes (May 2026)

The paper in `RepoPeft_Paper/main.pdf` has been rebuilt for camera-ready
submission with the latest experimental results (24 pages, ARR template,
final mode --- the `[review]` flag has been removed in `main.tex`).

## What changed for camera-ready

- **Title** updated in `RepoPeft_Paper/macros.tex`:
  *"Code2LoRA: Hypernetworks for Repository-Conditioned and
  Commit-Streaming Adapters of Code Language Models"*.
- **Abstract & introduction** rewritten to centre the story on
  (i) hypernetwork-distilled repo-specific adapters (`Code2LoRA`,
  static `Code2LoRA-GRU_file`) and
  (ii) commit-streaming adaptation (`Code2LoRA-GRU_commit`) with
  $O(1)$ per-commit updates --- mapped onto deployment regimes
  (batch / IDE-CI / federated).
- **Table 1 (`tab:main_results`)**: the `Code2LoRA-GRU_commit` row now
  carries real numbers from the full-scale (smart-capped, 400 train
  repos / 5 epochs / `MAX_SEQ_LEN=4096`) checkpoint at
  `$CKPT_DIR/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/.../code2lora_gru_best.pt`:
  CR-test 58.9% EM `[57.7, 60.1]`, IR-test 60.0% EM `[58.7, 61.3]`.
  All other rows are unchanged from the May 6 numbers in
  `AGENT_HANDOFF.md`.
- **Table 6 (`tab:ood_results`)** is now headlined by
  `Code2LoRA-GRU_commit` at **78.9% EM** on the 92-repo OOD bench
  (180,792 deduplicated assertions; 95% bootstrap CI
  `[78.74, 79.12]`). FFT and sLoRA OOD numbers are kept with the
  prefix-shape caveat called out in `sec:ood_caveats`. Pretrained on
  identical OOD inputs is 45.6% --- a clean +33.3-point delta for the
  streaming variant.
- **Conclusion** explicitly mentions the post-cutoff OOD gain so that
  reviewers can see the headline number without flipping pages.
- All `TBD` placeholders in the body and tables have been replaced.
  `% TODO [CAMERA-READY]` markers in `text/new.tex` are kept as
  comments only (they do not render in the PDF) and are listed below
  as the open follow-up work.

## Pending review comments / follow-ups

These are notes to consider in a later revision (out of scope for the
camera-ready unless the AC explicitly requests them):

1. **Static-vs-streaming framing.** The headline numbers tell two
   stories: GRU\textsubscript{file} wins CR/IR (64.4 / 66.4),
   GRU\textsubscript{commit} wins OOD (78.9). The current paper
   credits this to (a) prefix-shape (commit-derived OOD is the native
   prefix distribution for the streaming encoder) and (b) the
   architectural property of $O(1)$ per-commit updates. A reviewer may
   reasonably ask for a controlled experiment that disentangles
   prefix shape from generalization (e.g., rebuild the OOD bank with
   short cr_test-style prefixes and re-score every method). This is
   item 4 in `AGENT_HANDOFF.md` (open issue) and is the highest-impact
   follow-up.
2. **Apples-to-apples repo coverage in Table 1.** GRU\textsubscript{commit}
   is scored on 51/52 cr_test repos and 400/409 ir_test repos because a
   handful of repos are missing from `commit_parquet_hf` (see
   `scripts/slurm/missing_parquet_v2_repos.txt`). Other rows score the
   full set. Adding a `--restrict-to-method-supported-repos` flag in
   `evaluation/run_repopeft_bench.py` and re-scoring every row on the
   intersection would close this gap (`AGENT_HANDOFF.md` item 2).
3. **OOD coverage for context-based methods.** RAG / ICL / DRC /
   `Code2LoRA-direct` / GRU\textsubscript{file} / T2L-code need
   per-OOD-repo embeddings, chunk caches, or oracle-context caches
   before they can be scored on `ood_test.json`. The unified driver
   already supports these once the caches exist; this is a one-pass
   preprocessing job (`embed_repos/4_construct_embeddings.py` over
   `$SCRATCH/REPO_DATASET/repositories_ood/`). We deliberately omit
   these rows in Table 6 rather than fill them with mismatched
   numbers (`AGENT_HANDOFF.md` item 3).
4. **OOM in the IR-test pass for GRU\textsubscript{commit}.** The
   unified driver currently materializes all train-split repos
   upfront; switch to per-repo streaming before any large rerun
   (`evaluation/run_repopeft_bench.py:533`,
   `AGENT_HANDOFF.md` item 1).
5. **`cr_val` disclosure.** A 49-repo `cr_val` parquet split exists
   but is not yet documented in §4 of the paper
   (`AGENT_HANDOFF.md` item 5). One-paragraph addition.
6. **Architecture figure for GRU.** The TikZ figure in §3 currently
   only depicts the direct-projection variant. Adding a sibling
   figure (Mamba2 preamble → GRU sequential processing → PAW
   shared-basis generator → per-layer LoRA) would strengthen §3.
   Comment marker is at `text/new.tex:354`.
7. **Ablations.** LoRA rank, hidden dim, embedding components, GRU
   init type, BPTT window, file-ordering, PAW basis count, and base
   model scaling (0.5B / 1.5B / 3B) are all listed at
   `text/new.tex:1321`. None are blocking for the current claims, but
   the rebuttal will be easier with at least a basis-count and
   BPTT-window sweep.
8. **Random seeds and variance.** Currently we report bootstrap CIs
   on EM/EditSim/CodeBLEU but not seed variance. A
   3-seed re-run for the headline CR row would close
   `text/new.tex:1425`.
9. **Per-repo top/bottom-10 table.** Comment marker at
   `text/new.tex:1534`. Would be added to the appendix.
10. **Execution-based eval beyond the pilot.** `evaluation/exec_pilot.py`
    runs pytest on a hand-picked slice. Scaling this to all 512 repos
    would let us replace EM with a functional metric, but is gated on
    each repo's pytest config (some need network, fixtures, or
    proprietary services).
11. **NSERC funding number.** Acknowledgments line in `text/new.tex`
    has a `% TODO`; fill in once funding paperwork is final.

## How to rebuild the PDF

```
cd RepoPeft_Paper
latexmk -pdf -interaction=nonstopmode main.tex
```

Output: `RepoPeft_Paper/main.pdf` (24 pages, ${\sim}680\,$KB). The
build is reproducible from the texlive shipped with Compute Canada's
StdEnv/2023; only `pdflatex` and `bibtex` are required, no shell-escape
or external Python.
