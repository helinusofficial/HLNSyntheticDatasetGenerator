import hashlib
import json
import logging
import os
import random
import re
import time
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple
from datasets import Dataset, load_dataset
from llama_cpp import Llama


class SyntheticDatasetGenerator:

    def __init__(self,logger, cfg):
        self.logger=logger
        self.configs=cfg
        self.max_attempts = max(1, self.configs.total_samples * int(self.configs.max_attempts_multiplier))
        self.random = random.Random(self.configs.seed)
        self.llm = None
        self.judge_llm = None
        self.start_time = None
        self.accepted = 0
        self.attempts = 0
        self.signatures: Set[str] = set()
        self.user_signatures: Set[str] = set()
        self.logger = logging.getLogger("SyntheticDatasetGenerator")
        
        if self.configs.language not in self.configs.language_configs:
            raise ValueError(f"Unsupported language: {self.configs.language}")

        self.selected_lang_config = self.configs.language_configs[self.configs.language]
        self.topics = self.configs.topics[self.configs.language]
        self.tasks = self.selected_lang_config["tasks"]
        self.styles = self.selected_lang_config["styles"]
        self.audiences = self.selected_lang_config["audiences"]
        self.question_styles = self.selected_lang_config["question_styles"]


        if not isinstance(self.topics, list) or not self.topics:
            raise ValueError("Topics configuration must be a non-empty list")

        self.topics = [str(topic).strip() for topic in self.topics if str(topic).strip()]

    def _validate_config(self) -> None:
        if not os.path.isfile(self.configs.model_path):
            raise FileNotFoundError(f"Model file not found: {self.configs.model_path}")
        if self.configs.total_samples <= 0:
            raise ValueError("total_samples must be greater than zero")
        if self.configs.n_ctx <= 0 or self.configs.n_threads <= 0 or self.configs.n_batch <= 0:
            raise ValueError("n_ctx, n_threads and n_batch must be greater than zero")
        if self.configs.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if self.configs.shard_size <= 0:
            raise ValueError("shard_size must be greater than zero")
        if self.configs.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than zero")
        if not 0.0 < self.configs.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 < self.configs.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if not 0.0 <= self.configs.min_p <= 1.0:
            raise ValueError("min_p must be between 0 and 1")
        if self.configs.min_user_words <= 0 or self.configs.max_user_words < self.configs.min_user_words:
            raise ValueError("Invalid user word limits")
        if self.configs.min_assistant_words <= 0 or self.configs.max_assistant_words < self.configs.min_assistant_words:
            raise ValueError("Invalid assistant word limits")
        if self.configs.enable_quality_judge and not self.configs.judge_model_path:
            raise ValueError("judge_model_path is required when enable_quality_judge=True")
        if self.configs.multi_turn:
            if self.configs.min_turns <= 1:
                raise ValueError("multi_turn requires min_turns > 1")

            if self.configs.max_turns < self.configs.min_turns:
                raise ValueError("max_turns must be greater than or equal to min_turns")
        else:
            self.configs.min_turns = 1
            self.configs.max_turns = 1

    def load_model(self) -> None:
        self.logger.info(f"Loading generation model: {self.configs.model_path}")
        self.llm = Llama(
            model_path=self.configs.model_path,
            n_ctx=self.configs.n_ctx,
            n_threads=self.configs.n_threads,
            n_batch=self.configs.n_batch,
            n_gpu_layers=self.configs.n_gpu_layers,
            use_mmap=self.configs.load_model_use_mmap,
            use_mlock=self.configs.load_model_use_mlock,
            verbose=self.configs.load_model_verbose,
            seed=self.configs.seed
        )
        self.logger.info("Generation model loaded successfully")

    def load_judge_model(self) -> None:
        if not self.configs.enable_quality_judge:
            return
        if not os.path.isfile(self.configs.judge_model_path):
            raise FileNotFoundError(f"Judge model not found: {self.configs.judge_model_path}")
        self.logger.info(f"Loading judge model: {self.configs.judge_model_path}")
        self.judge_llm = Llama(model_path=self.configs.judge_model_path,
                               n_ctx=self.configs.n_ctx, n_threads=self.configs.n_threads,
                               n_batch=self.configs.n_batch, n_gpu_layers=self.configs.n_gpu_layers,
                               use_mmap=True, use_mlock=False, verbose=True, seed=self.configs.seed + 1000000)
        self.logger.info("Judge model loaded successfully")

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ۀ", "ه").replace("ة", "ه").replace("ـ", "")
        text = text.replace("\u200d", "").replace("\u200e", "").replace("\u200f", "")
        text = re.sub(r"[ \t\r\n]+", " ", text)
        text = re.sub(r"\s+([،؛؟,.!?])", r"\1", text)
        text = re.sub(r"([،؛؟,.!?])\1+", r"\1", text)
        return text.strip().casefold()

    def _normalize_persian_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ۀ", "ه").replace("ة", "ه").replace("ـ", "")
        text = text.replace("‌", "\u200c")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +([،؛؟])", r"\1", text)
        text = re.sub(r"([،؛؟]) +", r"\1 ", text)
        return text.strip()

    def _words(self, text: str) -> List[str]:
        return re.findall(r"\S+", self._normalize_text(text))

    def _word_count(self, text: str) -> int:
        return len(self._words(text))

    def _persian_letter_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        persian = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
        return persian / len(letters)

    def _arabic_character_ratio(self, text: str) -> float:
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return 0.0
        arabic = sum(1 for c in chars if c in "يىكؤإأة")
        return arabic / len(chars)

    def _persian_spacing_score(self, text: str) -> float:
        if self.configs.language != "fa":
            return 1.0

        words = self._words(text)

        if len(words) < 20:
            return 1.0

        prefixes = ["می", "نمی"]
        suffixes = ["ها", "های", "تر", "ترین"]

        possible = 0
        correct = 0

        for word in words:
            for prefix in prefixes:
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    possible += 1
                    if "\u200c" in word:
                        correct += 1

            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    possible += 1
                    if "\u200c" in word:
                        correct += 1

        if possible == 0:
            return 1.0

        return correct / possible

    def _contains_bad_pattern(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        return any(self._normalize_text(pattern) in normalized for pattern in self.selected_lang_config["bad_patterns"])

    def _repetition_ratio(self, text: str) -> float:
        words = self._words(text)
        if len(words) < 20:
            return 0.0
        counts = Counter(words)
        repeated = sum(v - 1 for v in counts.values() if v > 1)
        return repeated / len(words)

    def _sentence_repetition_ratio(self, text: str) -> float:
        sentences = [self._normalize_text(x) for x in re.split(r"[.!?؟\n]+", text) if self._normalize_text(x)]
        if len(sentences) < 4:
            return 0.0
        counts = Counter(sentences)
        repeated = sum(v - 1 for v in counts.values() if v > 1)
        return repeated / len(sentences)

    def _has_excessive_latin(self, text: str) -> bool:
        if self.configs.language != "fa":
            return False
        words = self._words(text)
        if not words:
            return True
        latin_words = sum(1 for word in words if re.search(r"[a-zA-Z]", word))
        return latin_words / len(words) > 0.28

    def _has_invalid_persian_characters(self, text: str) -> bool:
        if self.configs.language != "fa":
            return False
        return any(char in text for char in ["ي", "ك", "ى", "ة"])

    def _has_bad_punctuation(self, text: str) -> bool:
        if self.configs.language != "fa":
            return False
        if "؟؟؟" in text or "!!!" in text:
            return True
        if text.count("...") > 4:
            return True
        if re.search(r" {2,}", text):
            return True
        return False

    def _language_quality(self, text: str) -> float:
        if self.configs.language != "fa":
            return 1.0
        score = self._persian_letter_ratio(text)
        if self._has_invalid_persian_characters(text):
            score -= 0.15
        if self._has_excessive_latin(text):
            score -= 0.15
        if self._has_bad_punctuation(text):
            score -= 0.08
        return max(0.0, min(1.0, score))

    def _build_prompt(self, topic: str, task: str, style: str, difficulty: str,
                      audience: str, question_style: str, index: int) -> str:

        if self.configs.multi_turn:
            intro_fa = "یک نمونه مکالمه چندمرحله‌ای باکیفیت برای دیتاست آموزش و فاین‌تیون مدل زبانی تولید کن."
            intro_en = "Generate a high-quality multi-turn instruction-tuning example."

            turn_instruction_fa = f"""
    یک گفت‌وگوی چندمرحله‌ای تولید کن.

    تعداد نوبت‌های گفت‌وگو باید بین {self.configs.min_turns} و {self.configs.max_turns} نوبت باشد.
    هر نوبت شامل یک پیام user و یک پیام assistant است.

    گفت‌وگو باید پیوستگی معنایی داشته باشد.
    هر پیام user بعدی باید بر اساس پاسخ قبلی یا context مکالمه شکل بگیرد.
    کاربر نباید بدون ارتباط موضوع را تغییر دهد.
    در برخی نوبت‌ها می‌توان از ارجاع‌های طبیعی مانند «این مورد»، «همین راهکار»، «اگر این‌طور باشد» و موارد مشابه استفاده کرد.
    دستیار باید تمام context قبلی مکالمه را در نظر بگیرد.
    از تکرار سؤال یا پاسخ قبلی خودداری کن.
    مکالمه باید شبیه یک گفت‌وگوی واقعی و طبیعی باشد.
    """

            turn_instruction_en = f"""
    Generate a multi-turn conversation.

    The conversation must contain between {self.configs.min_turns} and {self.configs.max_turns} turns.
    Each turn consists of one user message followed by one assistant message.

    The conversation must maintain semantic continuity.
    Each following user message should naturally build on previous answers or conversation context.
    Do not abruptly switch to unrelated topics.
    Some user messages may naturally refer to previous context.
    The assistant must consider the full conversation history when responding.
    Avoid repeating previous questions or answers.
    The conversation must feel realistic and natural.
    """

        else:
            intro_fa = "یک نمونه تک‌مرحله‌ای باکیفیت برای دیتاست آموزش و فاین‌تیون مدل زبانی تولید کن."
            intro_en = "Generate a high-quality single-turn instruction-tuning example."

            turn_instruction_fa = """
    فقط یک نوبت سؤال و پاسخ تولید کن.

    خروجی باید دقیقاً شامل یک پیام user و یک پیام assistant باشد.
    """

            turn_instruction_en = """
    Generate exactly one user message followed by one assistant message.

    The output must contain exactly two messages.
    """

        if self.configs.language == "fa":
            return f"""/no_think
            {intro_fa}

    زبان هدف: فارسی
    موضوع: {topic}
    نوع کار: {task}
    سطح دشواری: {difficulty}
    مخاطب: {audience}
    سبک پاسخ: {style}
    نوع سؤال: {question_style}
    شناسه تنوع: {index}

    {turn_instruction_fa}

    سؤال‌ها باید کاملاً طبیعی و شبیه سؤال‌هایی باشند که یک فارسی‌زبان واقعی می‌پرسد.
    پاسخ‌های دستیار باید دقیق، مفید، مرتبط، روان و متناسب با context مکالمه باشند.

    از ترجمه تحت‌اللفظی انگلیسی خودداری کن.
    از ساختارهای تکراری و کلیشه‌ای استفاده نکن.
    پاسخ‌ها را با عبارت‌هایی مانند «حتماً»، «البته»، «امیدوارم این پاسخ مفید باشد» به شکل تکراری شروع یا تمام نکن.

    در متن فارسی:
    - از «ی» و «ک» فارسی استفاده کن.
    - از نیم‌فاصله در موارد مناسب مانند «می‌شود»، «می‌کند»، «داده‌ها»، «نرم‌افزارها» و «بهینه‌سازی» استفاده کن.
    - از علائم نگارشی فارسی مانند «،»، «؛» و «؟» طبیعی استفاده کن.

    واژه‌های انگلیسی فقط برای اصطلاح تخصصی، نام فناوری، نام محصول، کد، زبان برنامه‌نویسی یا نام خاص مجاز هستند.

    برای موضوعات پزشکی فقط اطلاعات عمومی و آموزشی ارائه کن و تشخیص، نسخه یا تصمیم درمانی شخصی ارائه نکن.

    برای مسائل استدلالی، نتیجه و توضیح لازم برای درک پاسخ را ارائه کن اما زنجیره تفکر خصوصی را افشا نکن.

    خروجی فقط JSON معتبر باشد.

    ساختار خروجی:
    {{
      "messages": [
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

    {"در حالت چندمرحله‌ای، messages باید با همین الگو ادامه پیدا کند: user → assistant → user → assistant → ..." if self.configs.multi_turn else "در حالت تک‌مرحله‌ای دقیقاً فقط دو پیام تولید کن: user → assistant."}
    """

        return f"""/no_think
        {intro_en}

    Target language: {self.selected_lang_config['name']}
    Topic: {topic}
    Task type: {task}
    Difficulty: {difficulty}
    Audience: {audience}
    Response style: {style}
    Question style: {question_style}
    Variation ID: {index}

    {turn_instruction_en}

    The user messages must be realistic and natural.
    The assistant responses must be accurate, useful, relevant and complete.

    Avoid repetitive structures, generic filler, artificial benchmark prompts and meta commentary.

    Return only valid JSON.

    Expected structure:
    {{
      "messages": [
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

    {"For multi-turn mode, continue the same pattern: user → assistant → user → assistant → ..." if self.configs.multi_turn else "For single-turn mode, return exactly two messages: user → assistant."}
    """

    def _validate_structure(self, sample: Any) -> Tuple[bool, str]:
        if not isinstance(sample, dict):
            return False, "not_object"

        messages = sample.get("messages")

        if not isinstance(messages, list):
            return False, "invalid_messages"

        if self.configs.multi_turn:
            turns = len(messages) // 2

            if turns < self.configs.min_turns or turns > self.configs.max_turns:
                return False, "invalid_multi_turn_turn_count"
        else:
            if len(messages) != 2:
                return False, "invalid_message_count"

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                return False, "invalid_message_objects"

            if set(message.keys()) != {"role", "content"}:
                return False, "invalid_keys"

            if not isinstance(message.get("content"), str):
                return False, "invalid_content_type"

            if not message["content"].strip():
                return False, "empty_content"

            expected_role = "user" if i % 2 == 0 else "assistant"

            if message.get("role") != expected_role:
                return False, "invalid_roles"

        user_messages = messages[0::2]
        assistant_messages = messages[1::2]

        total_user_words = sum(
            self._word_count(message["content"])
            for message in user_messages
        )

        total_assistant_words = sum(
            self._word_count(message["content"])
            for message in assistant_messages
        )

        if total_user_words < self.configs.min_user_words:
            return False, "user_too_short"

        if total_user_words > self.configs.max_user_words:
            return False, "user_too_long"

        if total_assistant_words < self.configs.min_assistant_words:
            return False, "assistant_too_short"

        if total_assistant_words > self.configs.max_assistant_words:
            return False, "assistant_too_long"

        return True, "ok"

    def _validate_language(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        messages = sample["messages"]

        for i, message in enumerate(messages):
            text = message["content"]

            if self.configs.language == "fa":
                if self._persian_letter_ratio(text) < 0.58:
                    return False, f"message_{i}_not_persian"

                if self._arabic_character_ratio(text) > 0.03:
                    return False, f"message_{i}_arabic_characters"

                if self._has_excessive_latin(text):
                    return False, f"message_{i}_excessive_latin"

                if self._has_invalid_persian_characters(text):
                    return False, f"message_{i}_invalid_persian_characters"

        return True, "ok"

    def _quality_score(self, sample: Dict[str, Any]) -> int:
        messages = sample["messages"]

        score = 100

        user_messages = [
            message["content"]
            for message in messages
            if message["role"] == "user"
        ]

        assistant_messages = [
            message["content"]
            for message in messages
            if message["role"] == "assistant"
        ]

        all_assistant_text = "\n".join(assistant_messages)
        all_user_text = "\n".join(user_messages)

        assistant_words = self._word_count(all_assistant_text)
        user_words = self._word_count(all_user_text)

        unique_ratio = len(set(self._words(all_assistant_text))) / max(1, assistant_words)

        if self._repetition_ratio(all_assistant_text) > 0.22:
            score -= 15

        if self._sentence_repetition_ratio(all_assistant_text) > 0.15:
            score -= 15

        if unique_ratio < 0.42:
            score -= 12

        if self._contains_bad_pattern(all_assistant_text):
            score -= 30

        if self._has_bad_punctuation(all_assistant_text):
            score -= 8

        if self.configs.language == "fa":
            if self._language_quality(all_user_text) < 0.72:
                score -= 10

            if self._language_quality(all_assistant_text) < 0.72:
                score -= 10

            if assistant_words > 60 and self._persian_spacing_score(all_assistant_text) < 0.25:
                score -= 3

        if user_words < 8:
            score -= 5

        if assistant_words < 30:
            score -= 8

        if assistant_words > 750 * max(1, len(assistant_messages)):
            score -= 5

        return max(0, min(100, score))

    def _normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        normalized_messages = []

        for message in sample["messages"]:
            content = message["content"].strip()

            if self.configs.language == "fa":
                content = self._normalize_persian_text(content)
            else:
                content = unicodedata.normalize("NFKC", content)

            normalized_messages.append({
                "role": message["role"],
                "content": content
            })

        return {
            "messages": normalized_messages
        }

    def _signature(self, sample: Dict[str, Any]) -> str:
        parts = []

        for message in sample["messages"]:
            parts.append(
                f"{message['role']}:{self._normalize_text(message['content'])}"
            )

        return hashlib.sha256(
            "\n".join(parts).encode("utf-8")
        ).hexdigest()

    def _user_signature(self, sample: Dict[str, Any]) -> str:
        user_messages = [
            self._normalize_text(message["content"])
            for message in sample["messages"]
            if message["role"] == "user"
        ]

        return hashlib.sha256(
            "\n".join(user_messages).encode("utf-8")
        ).hexdigest()

    def _judge(self, sample: Dict[str, Any]) -> bool:
        if not self.configs.enable_quality_judge:
            return True
        prompt = f"""این نمونه دیتاست instruction-tuning را از نظر کیفیت بررسی کن.

فقط JSON معتبر برگردان:
{{
  "score": 0,
  "relevant": true,
  "accurate": true,
  "natural": true,
  "complete": true,
  "acceptable": true
}}

نمونه:
{json.dumps(sample, ensure_ascii=False)}"""
        try:
            result = self.judge_llm.create_chat_completion(messages=[{"role": "system", "content": "You are a strict dataset quality evaluator. Return only valid JSON."}, {"role": "user", "content": prompt}], temperature=0.1, top_p=0.9, max_tokens=200, response_format={"type": "json_object"}, seed=self.configs.seed + self.accepted)
            judgment = json.loads(result["choices"][0]["message"]["content"].strip())
            return bool(judgment.get("acceptable", False)) and int(judgment.get("score", 0)) >= self.configs.min_quality_score
        except Exception as exc:
            self.logger.warning("Judge failure: %s", exc)
            return False

    def _generate_sample(self, index: int) -> Dict[str, Any]:
        topic = self.random.choice(self.topics)
        task = self.random.choice(self.tasks)
        style = self.random.choice(self.styles)
        difficulty = self.random.choice(["مبتدی", "متوسط", "پیشرفته", "تخصصی"] if self.configs.language == "fa" else ["Beginner", "Intermediate", "Advanced", "Expert"])
        audience = self.random.choice(self.audiences)
        question_style = self.random.choice(self.question_styles)
        prompt = self._build_prompt(topic, task, style, difficulty, audience, question_style, index)
        self.logger.info(f"Generating sample: index={index}, topic={topic}, task={task}, retry_count={self.configs.retry_count}")

        for retry in range(self.configs.retry_count):
            try:
                self.logger.info(f"Generation attempt: index={index}, retry={retry + 1}/{self.configs.retry_count}")

                temperature = min(
                    0.95,
                    max(0.55, self.configs.temperature + self.random.uniform(-0.08, 0.08))
                )

                
                self.logger.info("=" * 80)
                self.logger.info(f"START GENERATION | index={index} | retry={retry + 1}/{self.configs.retry_count}")
                self.logger.info("=" * 80)

                generation_start = time.time()

                stream = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"{self.configs.system_prompt}\n\n"
                                f"Your target language is {self.selected_lang_config['name']}.\n"
                                f"{self.selected_lang_config['prompt']}"
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    top_p=self.configs.top_p,
                    min_p=self.configs.min_p,
                    repeat_penalty=self.configs.repeat_penalty,
                    max_tokens=self.configs.max_tokens,
                    response_format={"type": "json_object"},
                    seed=self.configs.seed + index * 100 + retry,
                    stream=True,
                )

                stream_ready_time = time.time()

                self.logger.info(f"Stream ready after: {stream_ready_time - generation_start:.2f} seconds")

                raw_parts = []
                chunk_count = 0
                first_token_time = None

                for chunk in stream:
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if not content:
                        continue
                    if first_token_time is None:
                        first_token_time = time.time()
                        
                        self.logger.info(f"First token after: {first_token_time - generation_start:.2f} seconds")
                        self.logger.info("-" * 80)

                    raw_parts.append(content)
                    chunk_count += 1

                    self.logger.info(content, end="", flush=True)
                generation_end = time.time()
                raw = "".join(raw_parts).strip()
                total_time = generation_end - generation_start
                
                self.logger.info("-" * 80)
                if first_token_time:
                    first_token_delay = first_token_time - generation_start
                else:
                    first_token_delay = total_time
                generation_only_time = max(0.001,generation_end - (first_token_time or generation_start))
                speed = chunk_count / generation_only_time
                self.logger.info(
                    f"   GENERATION STATS\n"
                    f"   Total time       : {total_time:.2f} sec\n"
                    f"   First token      : {first_token_delay:.2f} sec\n"
                    f"   Chunks           : {chunk_count}\n"
                    f"   Approx speed     : {speed:.2f} chunks/sec\n"
                    f"   Output chars     : {len(raw)}"
                )

                self.logger.info("=" * 80)
                

                try:
                    sample = json.loads(raw)
                except json.JSONDecodeError:
                    self.configs.stats["json_failed"] += 1
                    self.logger.info(f"JSON parsing failed: index={index}, retry={retry + 1}")
                    continue

                valid, _ = self._validate_structure(sample)
                if not valid:
                    self.configs.stats["validation_failed"] += 1
                    self.logger.info(f"Structure validation failed: index={index}, retry={retry + 1}")
                    continue

                sample = self._normalize_sample(sample)
                valid, _ = self._validate_language(sample)
                if not valid:
                    self.configs.stats["language_failed"] += 1
                    self.logger.info(f"Language validation failed: index={index}, retry={retry + 1}")
                    continue

                if self._quality_score(sample) < self.configs.min_quality_score:
                    self.configs.stats["quality_failed"] += 1
                    self.logger.info(f"Quality validation failed: index={index}, retry={retry + 1}")
                    continue

                if not self._judge(sample):
                    self.configs.stats["quality_failed"] += 1
                    self.logger.info(f"Quality judge rejected sample: index={index}, retry={retry + 1}")
                    continue

                self.logger.info(f"Sample accepted: index={index}, retry={retry + 1}")
                return sample

            except Exception as exc:
                self.configs.stats["generation_failed"] += 1
                self.logger.info(f"Generation error: index={index}, retry={retry + 1}, error={exc}")
                self.logger.warning("Generation failure index=%s retry=%s error=%s", index, retry + 1, exc)

        self.logger.info(f"Sample generation failed after {self.configs.retry_count} retries: index={index}")
        return {}

    def _output_dir(self) -> str:
        directory = os.path.dirname(self.configs.output_path) or "."
        os.makedirs(directory, exist_ok=True)
        return directory

    def _checkpoint_dir(self) -> str:
        directory = os.path.join(self._output_dir(), ".checkpoints")
        os.makedirs(directory, exist_ok=True)
        return directory

    def _checkpoint_path(self) -> str:
        return os.path.join(self._checkpoint_dir(), "state.json")

    def _shard_path(self, index: int) -> str:
        base = os.path.splitext(os.path.basename(self.configs.output_path))[0]
        return os.path.join(self._output_dir(), f"{base}-{index:06d}.parquet")

    def _existing_shards(self) -> List[str]:
        base = os.path.splitext(os.path.basename(self.configs.output_path))[0]
        pattern = re.compile(rf"^{re.escape(base)}-\d{{6}}\.parquet$")
        return sorted(os.path.join(self._output_dir(), name) for name in os.listdir(self._output_dir()) if pattern.fullmatch(name))

    def _save_shard(self, samples: List[Dict[str, Any]], index: int) -> None:
        if not samples:
            return
        path = self._shard_path(index)
        temporary = f"{path}.tmp"
        self.logger.info(f"Saving shard {index}: {path}")
        Dataset.from_list(samples).to_parquet(temporary)
        os.replace(temporary, path)
        self.logger.info(f"Shard {index} saved successfully: {len(samples)} samples")

    def _save_checkpoint(self, next_index: int) -> None:
        state = {"next_index": next_index, "accepted": self.accepted, "attempts": self.attempts, "stats": self.configs.stats, "signatures": list(self.signatures), "user_signatures": list(self.user_signatures)}
        temporary = f"{self._checkpoint_path()}.tmp"
        self.logger.info(f"Saving checkpoint: accepted={self.accepted}, attempts={self.attempts}, next_index={next_index}")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(temporary, self._checkpoint_path())

    def run(self) -> None:
        self._validate_config()

        self.start_time = time.time()

        self.logger.info("=" * 80)
        self.logger.info("Starting synthetic dataset generation")
        self.logger.info(f"Target samples: {self.configs.total_samples}")
        self.logger.info(f"Language: {self.configs.language}")
        self.logger.info(f"Output: {self.configs.output_path}")
        self.logger.info("=" * 80)

        self.load_model()
        self.load_judge_model()

        existing_shards = self._existing_shards()

        if existing_shards:
            self.logger.info(f"Found {len(existing_shards)} existing shard(s)")

        current_shard: List[Dict[str, Any]] = []
        shard_index = len(existing_shards)
        next_index = self.accepted

        while self.accepted < self.configs.total_samples and self.attempts < self.max_attempts:
            self.attempts += 1
            self.configs.stats["attempts"] = self.attempts

            sample = self._generate_sample(self.attempts)

            if not sample:
                continue

            signature = self._signature(sample)
            user_signature = self._user_signature(sample)

            if signature in self.signatures or user_signature in self.user_signatures:
                self.configs.stats["duplicate_failed"] += 1
                self.logger.info(f"Duplicate sample rejected: index={self.attempts}")
                continue

            self.signatures.add(signature)
            self.user_signatures.add(user_signature)

            current_shard.append(sample)

            self.accepted += 1
            self.configs.stats["accepted"] = self.accepted
            next_index = self.accepted

            self.logger.info(
                f"Accepted: {self.accepted}/{self.configs.total_samples} "
                f"| Attempts: {self.attempts}/{self.max_attempts}"
            )

            if len(current_shard) >= self.configs.shard_size:
                self._save_shard(current_shard, shard_index)
                shard_index += 1
                current_shard = []

            if self.accepted % self.configs.checkpoint_interval == 0:
                self._save_checkpoint(next_index)

        if current_shard:
            self._save_shard(current_shard, shard_index)

        self._save_checkpoint(self.accepted)

        elapsed = time.time() - self.start_time

        self.logger.info("=" * 80)
        self.logger.info("Generation finished")
        self.logger.info(f"Accepted samples: {self.accepted}")
        self.logger.info(f"Total attempts: {self.attempts}")
        self.logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        self.logger.info(f"Stats: {self.configs.stats}")
        self.logger.info("=" * 80)