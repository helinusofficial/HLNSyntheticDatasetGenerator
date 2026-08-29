# Helinus Synthetic Dataset Generator

A local synthetic conversation dataset generator that uses a GGUF-compatible language model through `llama-cpp-python` to generate conversations and save them as a Parquet dataset.

---

## Features

* Load a GGUF-compatible model using `llama-cpp-python`
* Configure model inference parameters
* Configure conversation topics
* Select topics sequentially
* Automatically restart topic selection from the first topic after reaching the end of the topic list
* Configure the target number of conversations
* Configure maximum generated tokens
* Configure temperature
* Configure the requested number of conversation turns through the prompt
* Generate responses using streaming
* Collect streamed model output
* Optionally log generated model output
* Remove Markdown code fences from generated output
* Parse generated output as JSON
* Validate the existence of the `messages` field
* Validate that `messages` is a list
* Limit stored conversations to a maximum of 32 messages
* Save generated conversations to Parquet
* Save progress after every successfully generated conversation
* Resume from an existing Parquet dataset
* Log generation progress
* Log generation errors
* Measure and log generation time
* Continue generation when an individual conversation fails

---

## Requirements

* Python
* `llama-cpp-python`
* `pandas`
* `pyarrow`
* A GGUF-compatible language model

The class also uses Python standard-library modules including:

* `json`
* `os`
* `datetime`

---

## Configuration

The generator expects a configuration object containing the following values:

```python
class SyntheticDatasetConfig:

    model_path = r"C:\models\Qwen3-8B-Q6_K.gguf"

    output_file = "synthetic.parquet"
    output_temp_file = "synthetic.tmp.parquet"

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

---

## Configuration Options

| Option                  | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `model_path`            | Path to the GGUF model                            |
| `output_file`           | Path of the final Parquet dataset                 |
| `output_temp_file`      | Temporary Parquet file used during saving         |
| `topics`                | Topics used for conversation generation           |
| `num_conversations`     | Target number of conversations                    |
| `n_ctx`                 | Model context size                                |
| `n_threads`             | Number of CPU threads                             |
| `n_batch`               | Batch size                                        |
| `n_gpu_layers`          | Number of model layers assigned to GPU            |
| `verbose`               | Verbose setting passed to `Llama`                 |
| `max_tokens`            | Maximum number of generated tokens                |
| `max_turns`             | Requested conversation turns passed to the prompt |
| `temperature`           | Generation temperature                            |
| `Show_Generated_Output` | Controls whether generated output is logged       |
| `system_prompt`         | System prompt sent to the model                   |
| `conversation_prompt`   | Prompt template used for conversation generation  |

---

## Model Loading

The model is loaded when `PersianConversationGenerator` is initialized:

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

The class logs the loading process:

```text
Loading model...
Model loaded successfully!
```

---

## Topic Selection

Topics are loaded from:

```python
self.config.topics
```

The `get_next_topic()` method selects topics sequentially.

For example:

```text
Topic 1 → هوش مصنوعی
Topic 2 → پزشکی
Topic 3 → برنامه نویسی
Topic 4 → فناوری
Topic 5 → آموزش
Topic 6 → هوش مصنوعی
```

When the last topic is reached, the topic index is reset to `0`.

The class does not randomly select topics.

---

## Conversation Generation

Conversation generation is performed by:

```python
generate_conversation()
```

The method receives:

```python
conversation_index
total_conversations
max_turns
max_tokens
temperature
```

If `max_turns`, `max_tokens`, or `temperature` are not provided, their values are taken from the configuration.

The number of requested messages is calculated as:

```python
max_messages = max_turns * 2
```

The current topic is then inserted into the configured conversation prompt.

---

## Prompts

The generator sends two messages to the model:

```python
messages = [
    {
        "role": "system",
        "content": self.config.system_prompt
    },
    {
        "role": "user",
        "content": self.config.conversation_prompt.format(
            topic=topic,
            max_turns=max_turns,
            max_messages=max_messages
        )
    }
]
```

The conversation prompt can therefore use these placeholders:

```text
{topic}
{max_turns}
{max_messages}
```

---

## Streaming Generation

The model is called using:

```python
self.llm.create_chat_completion(
    messages=messages,
    max_tokens=max_tokens,
    temperature=temperature,
    stream=True
)
```

The generated chunks are collected into a single response:

```python
response = ""

for chunk in output:
    content = chunk["choices"][0]["delta"].get("content", "")

    if content:
        response += content
```

The complete generated response is then returned.

---

## Generated Output Logging

When:

```python
Show_Generated_Output = True
```

the generated response is written to the logger:

```python
if self.config.Show_Generated_Output:
    self.logger.info(response)
```

---

## JSON Processing

The generated response is stripped before parsing:

```python
clean_response = response.strip()
```

If the response starts with Markdown code fences, the class removes them:

````python
if clean_response.startswith("```"):
    clean_response = clean_response.replace("```json", "")
    clean_response = clean_response.replace("```", "")
    clean_response = clean_response.strip()
````

The cleaned response is then parsed using:

```python
conversation = json.loads(clean_response)
```

---

## Output Validation

The class performs basic structural validation.

It checks whether `messages` exists:

```python
if "messages" not in conversation:
    ...
```

It also checks whether `messages` is a list:

```python
if not isinstance(conversation["messages"], list):
    ...
```

Invalid conversations are skipped.

The class does not perform semantic or linguistic quality validation.

---

## Message Limit

Before storing a conversation, the class limits the messages to 32:

```python
conversation["messages"] = conversation["messages"][:32]
```

If the model generates more than 32 messages, only the first 32 are stored.

---

## Dataset Structure

Each successfully generated conversation is stored with:

```text
id
topic
messages
```

The record is created as:

```python
{
    "id": i + 1,
    "topic": topic,
    "messages": json.dumps(
        conversation["messages"],
        ensure_ascii=False
    )
}
```

The `messages` field is therefore stored as a JSON string.

---

## Parquet Saving

The dataset is converted to a pandas DataFrame:

```python
df = pd.DataFrame(dataset)
```

and saved to the temporary Parquet file:

```python
df.to_parquet(
    self.config.output_temp_file,
    index=False
)
```

The temporary file is then replaced with the final output file:

```python
os.replace(
    self.config.output_temp_file,
    self.output_file
)
```

---

## Checkpointing

The dataset is saved after every successfully generated conversation.

The process is:

```text
Generate
   ↓
Validate
   ↓
Append to dataset
   ↓
Write temporary Parquet
   ↓
Replace final Parquet
```

This allows the generated records to be persisted during a long-running generation process.

---

## Resume

When the output file already exists:

```python
if os.path.exists(self.output_file):
```

the class attempts to load it:

```python
existing_df = pd.read_parquet(self.output_file)
dataset = existing_df.to_dict("records")
```

Generation then starts from:

```python
len(dataset)
```

rather than starting from zero.

If loading the existing Parquet file fails, the class starts with an empty dataset:

```python
except Exception:
    dataset = []
```

---

## Error Handling

Invalid JSON is handled separately:

```python
except json.JSONDecodeError:
    self.logger.info("Error: Model returned invalid JSON.")
    self.logger.info("Conversation skipped.")
```

Other exceptions are also caught:

```python
except Exception as e:
    self.logger.info(f"Error: {e}")
    self.logger.info("Conversation skipped.")
```

Therefore, an error during one conversation does not automatically terminate the generation loop.

The failed conversation is skipped and the next iteration continues.

---

## Generation Time

Generation time is measured around the model generation process:

```python
start_time = datetime.now()

output = self.llm.create_chat_completion(...)

...

elapsed = datetime.now() - start_time
```

The elapsed time is logged in minutes and seconds:

```text
Generation time: 00:42
```

---

## Dataset Generation

The complete dataset generation is performed by:

```python
generate_dataset()
```

The method:

1. Loads an existing dataset if available.
2. Determines how many conversations already exist.
3. Generates the remaining conversations.
4. Selects a topic for each conversation.
5. Generates the model response.
6. Parses and validates the response.
7. Limits messages to 32.
8. Adds the conversation to the dataset.
9. Saves the updated dataset to Parquet.
10. Continues until the configured number of conversations is reached.

At the end, the method returns:

```python
df
```

where `df` is a pandas DataFrame containing the generated dataset.

---

## Final Dataset

The final DataFrame contains these columns:

```text
id
topic
messages
```

Example:

```text
+----+----------------+----------------------+
| id | topic          | messages             |
+----+----------------+----------------------+
| 1  | هوش مصنوعی     | JSON string          |
| 2  | پزشکی          | JSON string          |
| 3  | فناوری         | JSON string          |
+----+----------------+----------------------+
```

---

## Current Validation

The implementation currently validates only:

* JSON syntax
* Existence of `messages`
* `messages` being a list
* Maximum stored message count of 32

It does not currently validate:

* Duplicate conversations
* Duplicate questions
* Persian language quality
* Grammar
* Semantic quality
* Toxicity
* Hallucinations
* Message roles
* Message ordering
* Message content types
* Minimum or maximum message length
* Quality scores

---

## Important Behavior

### `max_turns`

`max_turns` is used to construct the generation prompt.

It is **not independently enforced after generation**.

### Topic Selection

Topics are selected sequentially rather than randomly.

### Failed Conversations

A failed conversation is skipped rather than terminating the entire generation process.

### Dataset Persistence

A successful conversation is saved immediately to the Parquet dataset.

### Duplicate Detection

The current implementation does **not** perform duplicate detection.

### Quality Evaluation

The current implementation does **not** perform semantic, linguistic, or quality evaluation.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
