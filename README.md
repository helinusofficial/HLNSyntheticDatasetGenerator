# Helinus Synthetic Dataset Generator

A local synthetic conversation dataset generator for creating Persian instruction-tuning and chat datasets from GGUF-compatible language models using `llama.cpp`.

The project uses a local LLM to generate structured Persian user-assistant conversations based on configurable topics and saves the generated dataset in **Parquet** format.

The generator is designed for:

* Instruction Tuning
* Supervised Fine-Tuning (SFT)
* Persian Chat Model Training
* Domain-Specific Dataset Generation
* Synthetic Conversation Generation

The entire generation process runs locally using a GGUF model through `llama-cpp-python`.

---

## Features

* Local GGUF model inference using `llama.cpp`
* Persian conversation generation
* Configurable conversation topics
* Configurable number of conversations
* Configurable number of conversation turns
* Configurable maximum generated tokens
* Configurable temperature
* Streaming model generation
* Structured JSON output
* Conversation schema validation
* Automatic removal of Markdown code fences from model output
* Maximum message limit per conversation
* Parquet dataset export
* Automatic checkpointing after every successfully generated conversation
* Resume generation from an existing Parquet file
* Logging of generation progress and errors
* Configurable CPU and GPU inference
* Support for different GGUF-compatible models

---

## How It Works

The generator follows this workflow:

```text
Load Configuration
        ↓
Load GGUF Model
        ↓
Select Next Topic
        ↓
Build System + User Prompts
        ↓
Generate Conversation
        ↓
Parse JSON
        ↓
Validate "messages"
        ↓
Limit Messages
        ↓
Save to Parquet
        ↓
Continue / Resume
```

---

## Requirements

* Python 3.10+
* A GGUF-compatible language model
* `llama-cpp-python`
* `pandas`
* `pyarrow`

The project also uses Python standard-library modules such as:

* `json`
* `os`
* `time`
* `datetime`
* `pathlib`

---

## Installation

Install the required Python packages:

```bash
pip install llama-cpp-python pandas pyarrow
```

Depending on your hardware, `llama-cpp-python` can be installed/configured with CPU or GPU support.

For GPU inference, make sure `llama-cpp-python` is installed with the appropriate backend for your hardware.

---

# Project Structure

A typical project structure can look like:

```text
project/
│
├── main.py
├── config.py
├── generator.py
├── logger.py
├── helpers/
│   └── ...
│
├── models/
│   └── model.gguf
│
├── dataset/
│   └── ...
│
└── alllogs/
    └── logs.txt
```

The exact structure depends on how `SyntheticDatasetConfig`, `MyLogger`, and `TimeFormatHelper` are organized in your project.

---

# Configuration

The generator receives its configuration through `SyntheticDatasetConfig`.

The following configuration values are used directly by `PersianConversationGenerator`:

```python
class SyntheticDatasetConfig:

    model_path = r"C:\models\Qwen3-8B-Q6_K.gguf"

    output_file = "synthetic.parquet"

    output_temp_file = None

    topics = [
        "سلامت و پزشکی",
        "هوش مصنوعی",
        "برنامه نویسی",
        "فناوری",
        "آموزش",
    ]

    num_conversations = 10000

    n_ctx = 4096
    n_threads = 8
    n_batch = 512
    n_gpu_layers = 0

    verbose = False

    max_tokens = 1536
    max_turns = 5
    temperature = 0.75

    Show_Generated_Output = False

    system_prompt = """
    ...
    """

    conversation_prompt = """
    ...
    """
```

> The exact values are examples. Your actual configuration class can define any topics, prompts, model path, and generation parameters required by your project.

---

# Configuration Options

| Option                  | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| `model_path`            | Path to the GGUF language model                          |
| `output_file`           | Name/path of the final Parquet dataset                   |
| `output_temp_file`      | Temporary file used while saving the dataset             |
| `topics`                | List of topics used for conversation generation          |
| `num_conversations`     | Total number of conversations to generate                |
| `n_ctx`                 | Context window size used by the model                    |
| `n_threads`             | Number of CPU threads used by `llama.cpp`                |
| `n_batch`               | Batch size used during inference                         |
| `n_gpu_layers`          | Number of model layers offloaded to GPU                  |
| `verbose`               | Enables/disables verbose `llama.cpp` output              |
| `max_tokens`            | Maximum number of tokens generated for each conversation |
| `max_turns`             | Maximum number of conversation turns requested           |
| `temperature`           | Controls randomness of generation                        |
| `Show_Generated_Output` | Logs the generated conversation                          |
| `system_prompt`         | System instruction provided to the model                 |
| `conversation_prompt`   | Prompt template used to generate conversations           |

---

# Topics

Topics are provided through the configuration:

```python
topics = [
    "هوش مصنوعی",
    "پزشکی",
    "برنامه نویسی پایتون",
    "امنیت سایبری",
    "آموزش",
    "فناوری",
]
```

The generator uses `get_next_topic()` to select topics sequentially.

For example:

```text
Conversation 1 → هوش مصنوعی
Conversation 2 → پزشکی
Conversation 3 → برنامه نویسی پایتون
Conversation 4 → امنیت سایبری
Conversation 5 → آموزش
Conversation 6 → فناوری
Conversation 7 → هوش مصنوعی
...
```

When the end of the topic list is reached, the generator starts again from the first topic.

---

# Prompt Configuration

Two prompts are used during generation.

## System Prompt

The system prompt is passed as a `system` message:

```python
{
    "role": "system",
    "content": self.config.system_prompt
}
```

This prompt should define the model's overall behavior and the required output format.

For example:

```python
system_prompt = """
You are a Persian synthetic dataset generator.

Generate natural and useful Persian user-assistant conversations.

Return ONLY valid JSON.
Do not use Markdown.
Do not add explanations outside the JSON.
"""
```

---

## Conversation Prompt

The conversation prompt is formatted dynamically with:

* `topic`
* `max_turns`
* `max_messages`

Example:

```python
conversation_prompt = """
Generate a Persian conversation about the following topic:

Topic: {topic}

Generate up to {max_turns} conversation turns.
The maximum number of messages is {max_messages}.

Return the result using this JSON structure:

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
"""
```

The generator automatically replaces:

```text
{topic}
{max_turns}
{max_messages}
```

with the current values.

---

# Conversation Generation

The main generation method is:

```python
generate_conversation()
```

Example:

```python
topic, response = generator.generate_conversation(
    conversation_index=1,
    total_conversations=10000,
    max_turns=5,
    max_tokens=1536,
    temperature=0.75
)
```

The model generates its response using:

```python
self.llm.create_chat_completion(
    messages=messages,
    max_tokens=max_tokens,
    temperature=temperature,
    stream=True
)
```

Generation is performed in streaming mode.

The generated chunks are combined into a single response before JSON parsing.

---

# JSON Output

The model is expected to return JSON in the following format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "هوش مصنوعی چیست؟"
    },
    {
      "role": "assistant",
      "content": "هوش مصنوعی به مجموعه‌ای از روش‌ها و فناوری‌ها گفته می‌شود که..."
    }
  ]
}
```

For multi-turn conversations:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "هوش مصنوعی چیست؟"
    },
    {
      "role": "assistant",
      "content": "هوش مصنوعی..."
    },
    {
      "role": "user",
      "content": "چه کاربردهایی دارد؟"
    },
    {
      "role": "assistant",
      "content": "هوش مصنوعی در حوزه‌های مختلفی مانند..."
    }
  ]
}
```

---

# Output Validation

After generation, the response is cleaned before parsing.

If the model returns Markdown code fences such as:

````text
```json
{
    "messages": [...]
}
````

````

the generator removes the code fences before calling:

```python
json.loads(clean_response)
````

The generator then checks that:

1. The response is valid JSON.
2. The `messages` field exists.
3. `messages` is a list.

Invalid generations are skipped.

---

# Message Limit

The generator currently limits every conversation to a maximum of **32 messages**:

```python
conversation["messages"] = conversation["messages"][:32]
```

Therefore, even if the model generates more messages, only the first 32 messages are stored.

---

# Dataset Format

The resulting Parquet dataset contains records similar to:

```text
id
topic
messages
```

Example:

```text
id: 1

topic:
هوش مصنوعی

messages:
[
    {
        "role": "user",
        "content": "هوش مصنوعی چیست؟"
    },
    {
        "role": "assistant",
        "content": "هوش مصنوعی..."
    }
]
```

The `messages` field is stored as a JSON string inside the Parquet file.

---

# Generating a Dataset

The recommended entry point is the `main()` function.

The basic workflow is:

```python
def main():
    try:
        start_time = time.time()
        start_datetime = datetime.now()

        logger_obj = MyLogger(
            log_dir="alllogs",
            log_file_name="logs.txt"
        )

        logger, path = logger_obj.setup()

        configs = SyntheticDatasetConfig(logger)
        configs.log()

        path = Path(path)

        configs.output_temp_file = path
        configs.output_file = path / configs.output_file

        generator = PersianConversationGenerator(
            logger,
            configs
        )

        generator.generate_dataset()

    except ValueError as error:
        print(f"ERROR: {error}")
```

Then:

```python
if __name__ == "__main__":
    main()
```

Run the project with:

```bash
python main.py
```

---

# Example: Generate 10,000 Conversations

Configure:

```python
num_conversations = 10000
```

For example:

```python
max_turns = 5
max_tokens = 1536
temperature = 0.75
```

Then run:

```bash
python main.py
```

The generator will continue until the configured number of conversations has been generated or the process is stopped.

---

# Checkpoint and Resume

The generator automatically checks whether the output Parquet file already exists:

```python
if os.path.exists(self.output_file):
```

If the file exists, it loads the previously generated records:

```python
existing_df = pd.read_parquet(self.output_file)

dataset = existing_df.to_dict("records")
```

The generator then resumes from:

```python
len(dataset)
```

instead of starting from zero.

For example, if:

```text
num_conversations = 10000
```

and the existing dataset contains:

```text
3500 conversations
```

the generator continues from conversation:

```text
3501
```

until it reaches:

```text
10000
```

This makes the generator suitable for long-running dataset generation jobs.

---

# Saving Progress

After every successfully generated conversation, the dataset is written to Parquet:

```python
df = pd.DataFrame(dataset)

df.to_parquet(
    self.config.output_temp_file,
    index=False
)

os.replace(
    self.config.output_temp_file,
    self.output_file
)
```

This means progress is persisted continuously rather than only at the end of the entire generation process.

If the process stops unexpectedly, previously saved conversations can be recovered from the output file.

---

# Error Handling

The generator handles invalid JSON separately:

```python
except json.JSONDecodeError:
    self.logger.info("Error: Model returned invalid JSON.")
    self.logger.info("Conversation skipped.")
```

Other runtime errors are also caught:

```python
except Exception as e:
    self.logger.info(f"Error: {e}")
    self.logger.info("Conversation skipped.")
```

A failed conversation therefore does not necessarily terminate the entire dataset generation process.

---

# Logging

Generation progress is logged using the project's `MyLogger`.

Example logs:

```text
Loading model...
Model loaded successfully!

============================================================
Generating 10000 conversations
Output: dataset/synthetic.parquet
============================================================

Conversation [1/10000] | Topic: هوش مصنوعی

Conversation saved successfully.

Conversation [2/10000] | Topic: پزشکی

Conversation saved successfully.
```

Generation time is also recorded:

```text
Generation time: 00:42
```

The project stores logs in:

```text
alllogs/logs.txt
```

---

# Model Configuration

The generator uses `llama-cpp-python`:

```python
self.llm = Llama(
    model_path=self.config.model_path,
    n_ctx=self.config.n_ctx,
    n_threads=self.config.n_threads,
    n_batch=self.config.n_batch,
    n_gpu_layers=self.config.n_gpu_layers,
    verbose=self.config.verbose
)
```

This allows the user to control how the GGUF model is executed.

### CPU Example

```python
n_threads = 8
n_gpu_layers = 0
```

### GPU Example

```python
n_threads = 8
n_gpu_layers = -1
```

The exact GPU configuration depends on the installed `llama-cpp-python` backend and available hardware.

---

# Generation Parameters

## `max_tokens`

Controls the maximum number of tokens generated by the model.

```python
max_tokens = 1536
```

Larger values allow longer conversations but can increase generation time and memory usage.

---

## `temperature`

Controls generation randomness.

```python
temperature = 0.75
```

Typical values:

```text
0.2 → More deterministic
0.5 → Conservative
0.7 → Balanced
0.9 → More diverse
1.0+ → Highly diverse
```

For synthetic training data, moderate values are generally preferable because the generated conversations should remain coherent while still providing variation.

---

## `max_turns`

Controls the requested number of conversation turns.

```python
max_turns = 5
```

The actual number of messages returned by the model depends on the prompt and model behavior.

The generator additionally enforces the final 32-message storage limit.

---

# Persian Dataset Generation

This project is primarily designed around Persian conversation generation.

For Persian datasets, the quality of the generated data depends heavily on:

* The selected GGUF model
* System prompt
* Conversation prompt
* Topic list
* Temperature
* Maximum token count
* Context size

For best results, the prompts should explicitly instruct the model to produce natural Persian and valid JSON.

---

# Example Configuration

A practical Persian configuration can look like:

```python
class SyntheticDatasetConfig:

    model_path = r"C:\models\Qwen3-8B-Q6_K.gguf"

    output_file = "persian_conversations.parquet"

    topics = [
        "هوش مصنوعی",
        "یادگیری ماشین",
        "برنامه نویسی پایتون",
        "پزشکی",
        "سلامت",
        "فناوری",
        "آموزش",
        "امنیت اطلاعات",
        "نرم افزار",
        "علم و فناوری",
    ]

    num_conversations = 10000

    n_ctx = 4096
    n_threads = 8
    n_batch = 512
    n_gpu_layers = 0

    verbose = False

    max_turns = 5
    max_tokens = 1536
    temperature = 0.75

    Show_Generated_Output = False

    system_prompt = """
    You are a Persian synthetic conversation dataset generator.

    Generate natural, useful and coherent Persian conversations.

    Return ONLY valid JSON.
    Do not use Markdown code fences.
    """

    conversation_prompt = """
    Generate a Persian conversation about:

    {topic}

    Generate up to {max_turns} turns.
    Maximum messages: {max_messages}

    Return ONLY JSON using this format:

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
    """
```

---

# Output

After generation, the dataset may look like:

```text
dataset/
└── persian_conversations.parquet
```

The Parquet file contains records such as:

```text
+----+----------------+----------------------+
| id | topic          | messages             |
+----+----------------+----------------------+
| 1  | هوش مصنوعی     | JSON messages        |
| 2  | پزشکی          | JSON messages        |
| 3  | برنامه نویسی   | JSON messages        |
| 4  | فناوری         | JSON messages        |
+----+----------------+----------------------+
```

---

# Loading the Dataset

The generated dataset can be loaded using pandas:

```python
import pandas as pd

df = pd.read_parquet(
    "dataset/persian_conversations.parquet"
)

print(df.head())
```

The messages can then be decoded:

```python
import json

messages = json.loads(
    df.iloc[0]["messages"]
)

print(messages)
```

---

# Example Dataset Record

```python
{
    "id": 1,
    "topic": "هوش مصنوعی",
    "messages": "[{\"role\": \"user\", \"content\": \"هوش مصنوعی چیست؟\"}, ...]"
}
```

After decoding the `messages` field:

```python
[
    {
        "role": "user",
        "content": "هوش مصنوعی چیست؟"
    },
    {
        "role": "assistant",
        "content": "هوش مصنوعی..."
    }
]
```

This structure can subsequently be transformed into the format required by different SFT and instruction-tuning frameworks.

---

# Large-Scale Generation

The generator can be configured for large datasets:

```python
num_conversations = 100000
```

or:

```python
num_conversations = 500000
```

Because the dataset is saved after each successful conversation, long generation jobs can resume from the existing Parquet file.

For large datasets, consider:

```python
n_ctx = 4096
max_tokens = 1536
max_turns = 5
```

and adjust the inference parameters according to the available CPU, RAM, VRAM, and model size.

---

# Important Notes

### Model Output Must Be JSON

The generator expects the model to return valid JSON.

If the model frequently produces invalid JSON, improve the `system_prompt` and `conversation_prompt`.

### Dataset Quality Depends on the Model

This project validates the basic structure of the generated response, but it does not currently perform semantic quality scoring or automatic human-like evaluation.

Generated conversations should therefore be sampled and manually inspected before using a large dataset for fine-tuning.

### Generation Speed Depends on Hardware

Generation speed is primarily affected by:

* Model size
* Quantization
* CPU performance
* GPU acceleration
* Number of GPU layers
* Context size
* Number of generated tokens

---

# Current Validation

The current implementation validates:

* JSON syntax
* Existence of `messages`
* `messages` being a list
* Maximum stored message count

The current implementation does **not** automatically validate:

* Persian language quality
* Grammar
* Semantic quality
* Duplicate conversations
* Duplicate questions
* Toxicity
* Hallucinations
* Role ordering
* Minimum/maximum word counts
* Quality scores

These checks can be added as future extensions.

---

# Future Improvements

Possible future improvements include:

* Automatic retry for invalid JSON
* Message role validation
* Persian language validation
* Duplicate detection
* Semantic quality scoring
* LLM-based quality judging
* Automatic filtering
* Dataset sharding
* Seed-based reproducibility
* `top_p` and `min_p` sampling
* `repeat_penalty`
* Parallel generation
* Multiple model support
* Separate checkpoint metadata
* Dataset statistics
* Generation metrics
* Automatic train/validation/test splitting

---

# License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
