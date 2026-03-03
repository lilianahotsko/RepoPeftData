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