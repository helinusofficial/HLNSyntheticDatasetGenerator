# Helinus Synthetic Dataset Generator

A production-ready pipeline for generating synthetic instruction-tuning datasets from local GGUF language models using `llama.cpp`.

The generator creates high-quality synthetic **user-assistant conversation datasets** designed for:

- Instruction Tuning
- Supervised Fine-Tuning (SFT)
- Chat model adaptation
- Domain-specific language model training

The pipeline runs fully locally with GGUF-compatible models and automatically handles:

- Dataset generation
- JSON parsing
- Schema validation
- Text normalization
- Language validation
- Quality filtering
- Duplicate detection
- Checkpointing
- Parquet export

The generator is model-agnostic and supports different GGUF models.  
The default example configuration uses:

`Qwen3-8B-Q6_K.gguf`

---

## Features

- Local GGUF inference using `llama.cpp`
- Persian and English dataset generation
- Configurable topics, tasks, styles, audiences, and question types
- Single-turn and multi-turn conversation generation
- Strict JSON output enforcement
- Message structure validation
- Language consistency checking
- Persian text normalization
- Character and punctuation normalization
- Word length constraints
- Quality scoring and filtering
- Duplicate conversation detection
- Duplicate user question detection
- Automatic retry mechanism
- Checkpoint and resume support
- Sharded Parquet export
- Optional LLM-based quality evaluation
- Reproducible generation with configurable seeds

---

## Generation Pipeline

```text
Generate Samples
        ↓
Parse JSON
        ↓
Validate Structure
        ↓
Normalize Text
        ↓
Validate Language
        ↓
Quality Scoring
        ↓
Duplicate Detection
        ↓
Save Checkpoint
        ↓
Export Dataset
```

---

## Requirements

- Python 3.10+
- GGUF-compatible language model
- llama-cpp-python
- datasets
- pyarrow

---

## Installation

```bash
pip install llama-cpp-python datasets pyarrow
```

---

## Configuration

The generator is configured through `SyntheticDatasetConfig`.

Example:

```python
class SyntheticDatasetConfig:

    model_path = r"C:\models\Qwen3-8B-Q6_K.gguf"

    output_path = "./dataset/synthetic.parquet"

    total_samples = 10000

    language = "fa"

    n_ctx = 4096
    n_threads = 8
    n_batch = 512
    n_gpu_layers = 0

    seed = 42

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

    export_final = True
    cleanup_shards = False

    multi_turn = True
    min_turns = 2
    max_turns = 5
```

---

## Configuration Options

| Option | Description |
|---|---|
| `model_path` | Path to GGUF generation model |
| `output_path` | Final Parquet output path |
| `total_samples` | Number of accepted samples |
| `language` | Dataset language (`fa` or `en`) |
| `n_ctx` | Model context size |
| `n_threads` | CPU thread count |
| `n_batch` | Inference batch size |
| `n_gpu_layers` | GPU offloaded layers |
| `max_tokens` | Maximum generated tokens |
| `shard_size` | Samples per Parquet shard |
| `checkpoint_interval` | Checkpoint frequency |
| `temperature` | Sampling temperature |
| `top_p` | Nucleus sampling parameter |
| `min_p` | Minimum probability filtering |
| `repeat_penalty` | Repetition control |
| `retry_count` | Retry count per sample |
| `min_quality_score` | Minimum accepted quality score |
| `enable_quality_judge` | Enable secondary LLM evaluation |
| `multi_turn` | Enable multi-turn conversations |
| `min_turns` | Minimum conversation turns |
| `max_turns` | Maximum conversation turns |

---

## Usage

```python
generator = SyntheticDatasetGenerator(
    model_path=SyntheticDatasetConfig.model_path,
    output_path=SyntheticDatasetConfig.output_path,
    total_samples=SyntheticDatasetConfig.total_samples,
    language=SyntheticDatasetConfig.language,
    n_ctx=SyntheticDatasetConfig.n_ctx,
    n_threads=SyntheticDatasetConfig.n_threads,
    n_batch=SyntheticDatasetConfig.n_batch,
    n_gpu_layers=SyntheticDatasetConfig.n_gpu_layers,
    seed=SyntheticDatasetConfig.seed,
    max_tokens=SyntheticDatasetConfig.max_tokens,
    topics=SyntheticDatasetConfig.topics
)

generator.run()
```

---

## Example

Generate 10,000 Persian multi-turn instruction samples:

```python
total_samples = 10000

language = "fa"

multi_turn = True

min_turns = 2
max_turns = 5
```

Output:

```text
dataset/

├── synthetic-000000.parquet
├── synthetic-000001.parquet
└── synthetic.parquet
```

---

## Dataset Format

Each sample follows the standard chat dataset format:

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

Multi-turn samples:

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
    },
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

---

## Language Support

### Persian (`fa`)

Includes:

- Persian character normalization
- Arabic character detection
- Half-space normalization
- Persian punctuation normalization
- Latin character ratio checking
- Persian language consistency validation

### English (`en`)

Includes:

- Natural English validation
- Language consistency checks
- Repetition detection
- Response quality analysis

---

## Quality Control

Every generated sample passes multiple validation stages:

- Valid JSON format
- Correct message schema
- Correct role ordering
- Word length limits
- Language validation
- Repetition detection
- Duplicate detection
- Quality threshold checking

Optional LLM-based evaluation:

```python
enable_quality_judge = True
```

---

## Checkpoint and Resume

Long generation jobs automatically create checkpoints.

If generation stops, the process can resume from the latest saved state.

Stored checkpoint information:

- Current progress
- Accepted samples
- Generation attempts
- Duplicate signatures
- Validation statistics

---

## Large Scale Generation

The pipeline supports large dataset generation:

```python
total_samples = 100000
```

or:

```python
total_samples = 500000
```

Recommended:

```python
shard_size = 5000

checkpoint_interval = 1000

cleanup_shards = False
```

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.