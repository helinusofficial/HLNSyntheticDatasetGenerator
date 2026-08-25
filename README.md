# Synthetic Dataset Generator

A production-ready pipeline for generating synthetic instruction-tuning datasets from local GGUF language models using `llama.cpp`.

The generator creates diverse user-assistant conversations and automatically validates, filters, deduplicates, checkpoints, and exports accepted samples as Parquet datasets.

## Features

* Local GGUF model inference with `llama.cpp`
* Persian, English, and German dataset generation
* Configurable topics and response styles
* JSON structure validation
* Language and text validation
* Persian text normalization
* Minimum and maximum response length control
* Quality scoring and filtering
* Duplicate detection
* Automatic retry for failed generations
* Checkpoint and resume support
* Sharded Parquet output
* Optional LLM-based quality judging
* Reproducible generation with configurable seeds

## Pipeline

```text
Generate
   ↓
Parse
   ↓
Validate
   ↓
Normalize
   ↓
Quality Check
   ↓
Deduplicate
   ↓
Checkpoint
   ↓
Parquet
```

## Requirements

* Python 3.10+
* A GGUF-compatible language model
* `llama-cpp-python`
* `datasets`
* `pyarrow`

### Installation

```bash
pip install llama-cpp-python datasets pyarrow
```

## Configuration

Configure the generator through `SyntheticDatasetConfig`:

```python
class SyntheticDatasetConfig:
    model_path = r"C:\models\Qwen3-8B-Q6_K.gguf"
    output_path = r"./dataset/synthetic.parquet"

    total_samples = 10000

    n_ctx = 4096
    n_threads = 8
    n_batch = 512
    n_gpu_layers = 0

    seed = 42
    language = "fa"
    max_tokens = 1536

    shard_size = 5000
    checkpoint_interval = 1000
    max_attempts_multiplier = 15

    min_user_words = 5
    max_user_words = 180
    min_assistant_words = 20
    max_assistant_words = 900

    min_quality_score = 72

    temperature = 0.75
    top_p = 0.9
    min_p = 0.05
    repeat_penalty = 1.08

    retry_count = 4

    enable_quality_judge = False
    judge_model_path = None

    keep_shards = True
    export_final = True
    cleanup_shards = False
```

## Main Options

| Option                 | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `model_path`           | Path to the GGUF generation model                         |
| `output_path`          | Final Parquet output path                                 |
| `total_samples`        | Number of accepted samples to generate                    |
| `language`             | Dataset language: `fa`, `en`, or `de`                     |
| `n_ctx`                | Model context size                                        |
| `n_threads`            | Number of CPU threads                                     |
| `n_batch`              | Model evaluation batch size                               |
| `n_gpu_layers`         | Number of GPU-offloaded layers; use `-1` for full offload |
| `max_tokens`           | Maximum number of generated tokens                        |
| `shard_size`           | Number of samples per Parquet shard                       |
| `checkpoint_interval`  | Number of accepted samples between checkpoints            |
| `min_quality_score`    | Minimum quality score required for acceptance             |
| `temperature`          | Generation temperature                                    |
| `top_p`                | Nucleus sampling parameter                                |
| `repeat_penalty`       | Repetition penalty                                        |
| `enable_quality_judge` | Enable secondary LLM-based quality evaluation             |
| `keep_shards`          | Keep generated Parquet shards                             |
| `export_final`         | Export the final combined dataset                         |
| `cleanup_shards`       | Remove shards after final export                          |

## Usage

Initialize the generator with the configured parameters:

```python
generator = SyntheticDatasetGenerator(
    model_path=SyntheticDatasetConfig.model_path,
    output_path=SyntheticDatasetConfig.output_path,
    total_samples=SyntheticDatasetConfig.total_samples,
    n_ctx=SyntheticDatasetConfig.n_ctx,
    n_threads=SyntheticDatasetConfig.n_threads,
    n_batch=SyntheticDatasetConfig.n_batch,
    seed=SyntheticDatasetConfig.seed,
    language=SyntheticDatasetConfig.language,
    n_gpu_layers=SyntheticDatasetConfig.n_gpu_layers,
    max_tokens=SyntheticDatasetConfig.max_tokens,
    shard_size=SyntheticDatasetConfig.shard_size,
    checkpoint_interval=SyntheticDatasetConfig.checkpoint_interval,
    max_attempts_multiplier=SyntheticDatasetConfig.max_attempts_multiplier,
    min_user_words=SyntheticDatasetConfig.min_user_words,
    max_user_words=SyntheticDatasetConfig.max_user_words,
    min_assistant_words=SyntheticDatasetConfig.min_assistant_words,
    max_assistant_words=SyntheticDatasetConfig.max_assistant_words,
    min_quality_score=SyntheticDatasetConfig.min_quality_score,
    temperature=SyntheticDatasetConfig.temperature,
    top_p=SyntheticDatasetConfig.top_p,
    min_p=SyntheticDatasetConfig.min_p,
    repeat_penalty=SyntheticDatasetConfig.repeat_penalty,
    retry_count=SyntheticDatasetConfig.retry_count,
    enable_quality_judge=SyntheticDatasetConfig.enable_quality_judge,
    judge_model_path=SyntheticDatasetConfig.judge_model_path,
    keep_shards=SyntheticDatasetConfig.keep_shards,
    export_final=SyntheticDatasetConfig.export_final,
    cleanup_shards=SyntheticDatasetConfig.cleanup_shards,
)

generator.run()
```

## Example

To generate 10,000 Persian samples:

```python
total_samples = 10000
language = "fa"
```

With:

```python
shard_size = 5000
```

the generator produces approximately two Parquet shards and, when final export is enabled, a combined dataset:

```text
dataset/
├── synthetic-000000.parquet
├── synthetic-000001.parquet
└── synthetic.parquet
```

## Dataset Format

Each generated sample follows the standard conversational message format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

The resulting dataset can be used for instruction tuning and supervised fine-tuning pipelines.

## Persian Support

When `language = "fa"`, the generator applies Persian-specific validation and normalization, including:

* Persian character normalization
* Punctuation normalization
* Half-space normalization
* Detection of unwanted Arabic characters
* Detection of excessive Latin characters
* Persian language consistency checks

## Quality Control

A sample is accepted only after passing the configured validation and quality checks.

The generator validates:

* JSON structure
* Message structure
* User and assistant content
* Minimum and maximum length constraints
* Language consistency
* Repetition
* Duplicate conversations
* Duplicate user questions
* Configured quality threshold

An optional secondary LLM can also be used as a quality judge.

## Checkpointing

Long-running generation jobs use checkpoints to preserve generation state.

If a generation process is interrupted, the generator can resume from the latest checkpoint without discarding previously generated data.

For large datasets, keeping checkpoints and generated shards enabled is recommended.

## Large-Scale Generation

The same pipeline can be used for larger datasets:

```python
total_samples = 100000
```

or:

```python
total_samples = 500000
```

For large-scale generation, keep sharding and checkpointing enabled:

```python
shard_size = 5000
keep_shards = True
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
