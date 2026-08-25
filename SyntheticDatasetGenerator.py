import os
import json
import random
from typing import Any, Dict, List, Set
from datasets import Dataset
from llama_cpp import Llama


class SyntheticDatasetGenerator:

    def __init__(self, model_path: str, output_path: str, total_samples, n_ctx, n_threads, n_batch, seed, language):
        self.model_path = model_path
        self.output_path = output_path
        self.total_samples = total_samples
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.seed = seed
        self.random = random.Random(seed)
        self.language = language

        self.topics = [
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Programming",
            "Python",
            "Software",
            "Hardware",
            "Cybersecurity",
            "Networking",
            "Databases",
            "Software Engineering",
            "Data Science",
            "Statistics",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "General Medicine",
            "Digital Health",
            "Business",
            "Management",
            "Economics",
            "Marketing",
            "Content Creation",
            "Education",
            "History",
            "Geography",
            "Translation",
            "Text Summarization",
            "Text Analysis",
            "Problem Solving",
            "Reasoning",
            "Concept Comparison",
            "General Question Answering"
        ]

        self.styles = [
            "Short and precise answer",
            "Detailed explanatory answer",
            "Step-by-step answer",
            "Educational answer for a beginner",
            "Technical answer for an experienced user",
            "Answer with examples",
            "Comparative answer",
            "Analytical answer",
            "Practical answer",
            "Simple and easy-to-understand answer"
        ]

        self.llm = None

    def _get_language_name(self) -> str:
        languages = {
            "fa": "Persian",
            "en": "English",
            "de": "German",
            "ar": "Arabic",
            "tr": "Turkish"
        }

        if self.language not in languages:
            raise ValueError(f"Unsupported language: {self.language}")

        return languages[self.language]

    def _system_prompt(self) -> str:
        language = self._get_language_name()

        return (
            f"You are a synthetic dataset generator for training and fine-tuning "
            f"large language models. All generated content must be written in {language}. "
            f"The data must be natural, diverse, accurate, and high quality. "
            f"The user's question must be realistic and meaningful, and the assistant's "
            f"answer must be accurate, useful, and directly relevant to the question. "
            f"Avoid repetitive sentences, generic answers, fabricated information, "
            f"unnatural text, and unnecessary verbosity. Vary the length and structure "
            f"of questions and answers. Do not output anything outside the JSON."
        )

    def load_model(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False
        )

    def _build_prompt(self, topic: str, style: str, index: int) -> str:
        language = self._get_language_name()

        return f"""
Generate one high-quality synthetic instruction-tuning example in {language}.

Topic: {topic}
Response style: {style}
Variation ID: {index}

The example must be a natural conversation between a user and an assistant.
The user's question must be realistic and meaningful.
The assistant's answer must provide useful and accurate information.
Do not create repetitive or templated conversations.
Avoid repeating structures from previous examples.
Vary the length and structure of questions and answers.

Output only valid JSON with exactly this structure:

{{
  "messages": [
    {{
      "role": "system",
      "content": "..."
    }},
    {{
      "role": "user",
      "content": "..."
    }},
    {{
      "role": "assistant",
      "content": "..."
    }}
  ]
}}
"""

    def _validate(self, sample: Any) -> bool:
        if not isinstance(sample, dict):
            return False

        messages = sample.get("messages")

        if not isinstance(messages, list):
            return False

        if len(messages) < 2:
            return False

        has_user = False
        has_assistant = False

        for message in messages:
            if not isinstance(message, dict):
                return False

            role = message.get("role")
            content = message.get("content")

            if role not in {"system", "user", "assistant"}:
                return False

            if not isinstance(content, str):
                return False

            if not content.strip():
                return False

            if role == "user":
                has_user = True

            if role == "assistant":
                has_assistant = True

        return has_user and has_assistant

    def _normalize(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        messages = []

        for message in sample["messages"]:
            messages.append({
                "role": str(message["role"]).strip().lower(),
                "content": str(message["content"]).strip()
            })

        return {"messages": messages}

    def _signature(self, sample: Dict[str, Any]) -> str:
        return json.dumps(sample["messages"], ensure_ascii=False, sort_keys=True).strip().lower()

    def _generate_sample(self, index: int) -> Dict[str, Any]:
        topic = self.random.choice(self.topics)
        style = self.random.choice(self.styles)

        prompt = self._build_prompt(topic=topic, style=style, index=index)

        for _ in range(5):
            try:
                result = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=1400,
                    response_format={"type": "json_object"}
                )

                content = result["choices"][0]["message"]["content"].strip()
                sample = json.loads(content)

                if self._validate(sample):
                    return self._normalize(sample)

            except Exception:
                continue

        return {}

    def generate(self) -> Dataset:
        if self.llm is None:
            self.load_model()

        samples: List[Dict[str, Any]] = []
        signatures: Set[str] = set()
        index = 0

        while len(samples) < self.total_samples:
            sample = self._generate_sample(index)
            index += 1

            if not sample:
                continue

            signature = self._signature(sample)

            if signature in signatures:
                continue

            signatures.add(signature)
            samples.append(sample)

            print(f"Generated {len(samples)}/{self.total_samples}", end="\r", flush=True)

        return Dataset.from_list(samples)

    def save(self, dataset: Dataset):
        directory = os.path.dirname(self.output_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        dataset.to_parquet(self.output_path)

    def run(self):
        dataset = self.generate()
        self.save(dataset)

        print()
        print(f"Dataset saved: {self.output_path}")
        print(f"Samples: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")