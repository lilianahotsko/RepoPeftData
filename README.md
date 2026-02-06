# RepoPeftData

A toolkit for collecting GitHub repositories with test folders and organizing them into a reusable dataset format for Python test code generation research.

**Note**: By default, all data is stored in `/home/lhotsko/scratch`.

## Features

- 🔍 **Search GitHub repositories** with tests folders using GitHub API
- 📥 **Download repositories** with configurable shallow/full cloning
- 📊 **Organize datasets** into structured format with test-source pairs
- 🚀 **Parallel downloading** for efficient collection
- 📈 **Dataset statistics** and indexing

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd RepoPeftData
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up GitHub token for higher rate limits:
```bash
export GITHUB_TOKEN=your_github_token_here
```

To create a GitHub token:
- Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate a new token with `public_repo` scope

## Quick Start

### Step 1: Collect Repository List

Search GitHub for repositories with tests folders:

```bash
python collect_repos.py --max-results 100 --min-stars 50
```

Options:
- `--token`: GitHub personal access token (or set `GITHUB_TOKEN` env var)
- `--language`: Programming language filter (default: `python`)
- `--min-stars`: Minimum number of stars (default: `10`)
- `--max-results`: Maximum repositories to collect (default: `1000`)
- `--output`: Output JSON file (default: `repositories.json`)

Example:
```bash
python collect_repos.py \
    --language python \
    --min-stars 100 \
    --max-results 500 \
    --output my_repos.json
```

### Step 2: Download Repositories

Download the collected repositories:

```bash
python download_repos.py --repos-file repositories.json --shallow --depth 1
```

Options:
- `--repos-file`: Input JSON file with repository list (default: `repositories.json`)
- `--dataset-root`: Root directory for dataset (default: `/home/lhotsko/scratch`)
- `--shallow`: Perform shallow clone (default: `True`)
- `--no-shallow`: Perform full clone
- `--depth`: Shallow clone depth (default: `1`)
- `--max-workers`: Maximum concurrent downloads (default: `5`)
- `--create-index`: Create dataset index after downloading

Example:
```bash
python download_repos.py \
    --repos-file my_repos.json \
    --dataset-root my_dataset \
    --shallow \
    --depth 1 \
    --max-workers 10 \
    --create-index
```

### Step 3: Organize Dataset

Organize downloaded repositories into structured format:

```bash
python organize_dataset.py  # Uses /home/lhotsko/scratch by default
```

This will:
- Extract all test files into `organized/test_files/`
- Extract all source files into `organized/source_files/`
- Create test-source pairs in `organized/test_source_pairs/`
- Generate statistics in `organized/statistics.json`

Options:
- `--dataset-root`: Root directory of the dataset (default: `/home/lhotsko/scratch`)
- `--no-pairs`: Skip creating test-source pairs

## Dataset Structure

After running all scripts, your dataset will have the following structure:

```
/home/lhotsko/scratch/
├── repositories/           # Cloned repositories
│   ├── owner1_repo1/
│   ├── owner2_repo2/
│   └── ...
├── metadata/              # Repository metadata JSON files
│   ├── owner1_repo1.json
│   └── ...
├── organized/            # Organized dataset
│   ├── test_files/      # All test files organized by repo
│   ├── source_files/    # All source files organized by repo
│   ├── test_source_pairs/  # Test-source file pairs
│   └── statistics.json  # Dataset statistics
└── dataset_index.json   # Complete dataset index
```

## Usage Examples

### Collect 1000 Python repositories with at least 50 stars:

```bash
python collect_repos.py \
    --language python \
    --min-stars 50 \
    --max-results 1000 \
    --output python_repos.json
```

### Download with full history (for deeper analysis):

```bash
python download_repos.py \
    --repos-file python_repos.json \
    --no-shallow \
    --max-workers 3 \
    --create-index
```

### Organize and create pairs:

```bash
python organize_dataset.py  # Uses /home/lhotsko/scratch by default
```

## Complete Workflow

Run all steps in sequence:

```bash
# 1. Collect repositories
python collect_repos.py --max-results 500 --min-stars 20

# 2. Download repositories
python download_repos.py --create-index

# 3. Organize dataset
python organize_dataset.py
```

## Dataset Format

### Repository Metadata

Each repository has a metadata JSON file with:
- Repository name, URL, clone URLs
- Stars, forks, language
- Creation and update dates
- Local path and download timestamp

### Test-Source Pairs

Each pair JSON file contains:
- `test_file`: Path to test file
- `source_file`: Path to corresponding source file
- `test_content`: Full test file content
- `source_content`: Full source file content

### Dataset Index

The `dataset_index.json` contains:
- Complete list of all repositories
- Statistics and metadata
- Tests folder locations for each repo

## Rate Limits

GitHub API rate limits:
- **Without token**: 60 requests/hour
- **With token**: 5,000 requests/hour

The scripts automatically handle rate limiting and will wait when limits are approached.

## Tips

1. **Start small**: Test with `--max-results 10` first
2. **Use shallow clones**: Faster and sufficient for most use cases (`--shallow --depth 1`)
3. **Adjust workers**: More workers = faster but more resource usage
4. **Save progress**: The `repositories.json` file saves your search results
5. **Resume downloads**: Already downloaded repos are skipped automatically

## Troubleshooting

### Rate limit errors
- Set up a GitHub token: `export GITHUB_TOKEN=your_token`
- Reduce `--max-workers` to slow down requests

### Clone failures
- Some repositories may be private or deleted
- Check `repositories.json` for valid URLs
- Failed clones are logged and skipped

### Disk space
- Use shallow clones (`--shallow --depth 1`)
- Start with smaller `--max-results`
- Clean up `/home/lhotsko/scratch/repositories/` if needed

## License

MIT License - feel free to use and modify for your research needs.

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
