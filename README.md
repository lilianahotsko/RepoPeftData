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