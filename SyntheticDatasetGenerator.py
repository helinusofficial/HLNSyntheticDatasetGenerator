import os
import json
import random
from typing import Any, Dict, List, Set
from datasets import Dataset
from llama_cpp import Llama


class SyntheticDatasetGenerator:

    def __init__(
        self,
        model_path: str,
        output_path: str,
        total_samples: int = 10000,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_batch: int = 512,
        seed: int = 42
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.total_samples = total_samples
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.seed = seed
        self.random = random.Random(seed)

        self.topics = [
            "هوش مصنوعی",
            "یادگیری ماشین",
            "یادگیری عمیق",
            "برنامه نویسی",
            "پایتون",
            "نرم افزار",
            "سخت افزار",
            "امنیت سایبری",
            "شبکه و اینترنت",
            "پایگاه داده",
            "مهندسی نرم افزار",
            "علم داده",
            "آمار",
            "ریاضیات",
            "فیزیک",
            "شیمی",
            "زیست شناسی",
            "پزشکی عمومی",
            "سلامت دیجیتال",
            "کسب و کار",
            "مدیریت",
            "اقتصاد",
            "بازاریابی",
            "تولید محتوا",
            "آموزش",
            "تاریخ",
            "جغرافیا",
            "زبان فارسی",
            "ترجمه",
            "خلاصه سازی متن",
            "تحلیل متن",
            "حل مسئله",
            "استدلال",
            "مقایسه مفاهیم",
            "پرسش و پاسخ عمومی"
        ]

        self.styles = [
            "پاسخ کوتاه و دقیق",
            "پاسخ تشریحی",
            "پاسخ مرحله به مرحله",
            "پاسخ آموزشی برای فرد مبتدی",
            "پاسخ تخصصی برای فرد حرفه ای",
            "پاسخ همراه با مثال",
            "پاسخ همراه با مقایسه",
            "پاسخ تحلیلی",
            "پاسخ کاربردی",
            "پاسخ ساده و قابل فهم"
        ]

        self.llm = None

    def load_model(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

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

    def _system_prompt(self) -> str:
        return (
            "تو یک تولیدکننده دیتاست مصنوعی برای آموزش و Fine-Tuning "
            "مدل‌های زبانی هستی. داده باید کاملاً فارسی، طبیعی، متنوع، "
            "دقیق و باکیفیت باشد. سوال کاربر باید واقعی و معنادار باشد "
            "و پاسخ دستیار باید دقیق، مفید و مرتبط با سوال باشد. "
            "از جملات کلیشه‌ای، تکرار، پاسخ‌های بی‌محتوا، اطلاعات ساختگی "
            "و متن غیرطبیعی خودداری کن. طول سوال و پاسخ را متنوع کن. "
            "هیچ توضیحی خارج از JSON تولید نکن."
        )

    def _build_prompt(self, topic: str, style: str, index: int) -> str:
        return f"""
یک نمونه آموزشی فارسی باکیفیت تولید کن.

موضوع: {topic}
سبک پاسخ: {style}
شناسه تنوع: {index}

نمونه باید یک مکالمه واقعی بین کاربر و دستیار باشد.
سوال و پاسخ نباید کلیشه‌ای یا تکراری باشند.
پاسخ دستیار باید اطلاعات مفید ارائه کند.
از تکرار ساختار نمونه‌های قبلی خودداری کن.

خروجی فقط JSON معتبر و دقیقاً با ساختار زیر باشد:

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
            messages.append(
                {
                    "role": str(message["role"]).strip().lower(),
                    "content": str(message["content"]).strip()
                }
            )

        return {"messages": messages}

    def _signature(self, sample: Dict[str, Any]) -> str:
        return json.dumps(
            sample["messages"],
            ensure_ascii=False,
            sort_keys=True
        ).strip().lower()

    def _generate_sample(self, index: int) -> Dict[str, Any]:
        topic = self.random.choice(self.topics)
        style = self.random.choice(self.styles)

        prompt = self._build_prompt(
            topic=topic,
            style=style,
            index=index
        )

        for _ in range(5):
            try:
                result = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": self._system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=1400,
                    response_format={
                        "type": "json_object"
                    }
                )

                content = (
                    result["choices"][0]
                    ["message"]["content"]
                    .strip()
                )

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

            print(
                f"Generated {len(samples)}/{self.total_samples}",
                end="\r",
                flush=True
            )

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




