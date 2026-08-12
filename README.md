# GenPuzzle

GenPuzzle is a benchmark for evaluating visual reasoning in image generation models. A model must understand a visual or textual puzzle, reason under task-specific constraints, and generate the final answer as an image.

This GitHub repository contains the source code for generation, evaluation, validation, and reporting. The full benchmark data, generated model outputs, human/automatic evaluation results, and report artifacts are hosted on Hugging Face:

**Dataset and results:** https://huggingface.co/datasets/zhangsan672/GenPuzzle

## Repository Contents

```text
GenPuzzle/
├── code/              # generation, evaluation, reporting, providers, and task registry
├── LICENSE            # code and data license information
├── README.md          # this file
└── .gitignore
```

The full Hugging Face release contains:

```text
code/       # source code
datasets/   # 2,005 benchmark instances across 12 tracks
runs/       # generated outputs, evaluation records, human-eval tables, and reports
LICENSE
README.md
```

## Benchmark Tracks

| Track | Count | Description |
|---|---:|---|
| `figure_completion` | 394 | Missing-figure pattern completion |
| `spatial_generation` | 56 | Spatial assembly, folding, projection, and complement generation |
| `maze_beginner` | 64 | Beginner maze path generation |
| `maze_intermediate` | 64 | Intermediate maze path generation |
| `maze_advanced` | 64 | Advanced maze path generation |
| `sudoku_reasoning` | 78 | 4x4 Sudoku visual completion |
| `nonogram_reasoning` | 150 | Nonogram/Picross grid reasoning |
| `tangram_reasoning` | 150 | Tangram silhouette assembly |
| `board_game_reasoning` | 300 | Board-game state and move reasoning |
| `matchstick_reasoning` | 300 | Matchstick-equation editing |
| `orthographic_reasoning` | 90 | Cube and orthographic-view reasoning |
| `math_visual_reasoning` | 295 | Visual mathematical proof and image-conditioned reasoning |
| **Total** | **2,005** | 12 equally weighted tracks |

## Installation

Use Python 3.10 or later.

```bash
git clone https://github.com/zhaochangpeng-csu/GenPuzzle.git
cd GenPuzzle
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r code/requirements.txt
```

On Windows PowerShell:

```powershell
git clone https://github.com/zhaochangpeng-csu/GenPuzzle.git
cd GenPuzzle
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r code\requirements.txt
$env:PYTHONIOENCODING="utf-8"
```

## Download the Full Data Release

The code expects `datasets/` and `runs/` to be located at the repository root when running full validation, generation, evaluation, or reporting. Download them from Hugging Face:

```bash
# Option 1: clone with Git LFS
pip install -U huggingface_hub
huggingface-cli download zhangsan672/GenPuzzle --repo-type dataset --local-dir .
```

If you only need the code, cloning this GitHub repository is sufficient. If you want to reproduce the paper results, download the Hugging Face release so that the local layout is:

```text
GenPuzzle/
├── code/
├── datasets/
└── runs/
```

## API Credentials

No API keys are included. Set only the variables required by the provider you use.

```bash
export OPENAI_API_KEY="YOUR_KEY"      # OpenAI generation and OpenAI-compatible judges
export GEMINI_API_KEY="YOUR_KEY"      # Google/Gemini image generation
export ARK_API_KEY="YOUR_KEY"         # Ark/Seedream image generation
export OPENAI_BASE_URL="https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"  # optional
```

PowerShell equivalents:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:GEMINI_API_KEY="YOUR_KEY"
$env:ARK_API_KEY="YOUR_KEY"
$env:OPENAI_BASE_URL="https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1"
```

## Validate the Dataset

After downloading `datasets/` from Hugging Face:

```bash
python code/benchmark.py list
python code/benchmark.py validate --tasks all --output validation_summary.json
```

The expected total is 2,005 instances.

## Generate Images

Dry-run prompt construction without external API calls:

```bash
python code/benchmark.py generate --tasks all --provider openai --model gpt-image-2 --run-name dryrun --limit-per-task 1 --dry-run
```

Generate with OpenAI image models:

```bash
python code/benchmark.py generate --tasks all --provider openai --model gpt-image-2 --run-name gpt_image_2_main --workers 1 --overwrite
```

Generate with Gemini image models:

```bash
python code/benchmark.py generate --tasks all --provider google --model gemini-3.1-flash-image --run-name gemini_preview_main --workers 1 --overwrite
```

Generate with Ark/Seedream:

```bash
python code/benchmark.py generate --tasks all --provider ark --model doubao-seedream-5-0-pro-260628 --run-name seedream5_full --workers 1 --overwrite
```

Outputs are written to:

```text
runs/<run-name>/<track-name>/images/
runs/<run-name>/<track-name>/records.jsonl
```

## Evaluate Generated Images

Evaluate a run with a multimodal judge:

```bash
python code/benchmark.py evaluate --tasks all --run-name gpt_image_2_main --judge-model gpt-5.5 --reasoning-effort high --passes 1 --workers 1 --overwrite
```

For a quick command check without API calls:

```bash
python code/benchmark.py evaluate --tasks all --run-name gpt_image_2_main --judge-model gpt-5.5 --dry-run
```

Task-specific evaluators combine programmatic checks, transcription-based checks, and structured multimodal judging depending on the track.

## Build Result Reports

After downloading `runs/` from Hugging Face:

```bash
python code/benchmark.py report --runs gpt_image_2_main,gemini_preview_main,seedream5_full --judge-model gpt-5.5,gpt-5.5,gpt-5.5 --out-dir runs/report_gpt_vs_gemini_vs_seedream_gpt5.5 --count-errors-as-zero
```

Precomputed report artifacts are included in the Hugging Face `runs/` directory.

## Data Format

Most tracks use JSONL metadata. Rows typically contain an instance ID, input image path(s), reference answer image path(s), prompt text or task-specific metadata, and evaluation-related fields. Dataset JSONL files may preserve source-language prompts and formulas when they are part of the benchmark content.

Images are stored with relative paths inside each dataset directory. Run outputs follow the same task names used by `code/task_registry.py`.

## License

Source code is released under the MIT License. Benchmark data, documentation, figures, generated outputs, and result files are released under CC BY 4.0 unless a file states otherwise. See `LICENSE` for details and third-party-material notes.

## Citation

If you use GenPuzzle, please cite the paper when it becomes available. A BibTeX entry will be added after public release.
