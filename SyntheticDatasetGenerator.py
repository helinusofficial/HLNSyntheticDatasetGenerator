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
        self.model_path = self.configs.model_path
        self.output_path = self.configs.output_path
        self.total_samples = int(self.configs.total_samples)
        self.n_ctx = int(self.configs.n_ctx)
        self.n_threads = int(self.configs.n_threads)
        self.n_batch = int(self.configs.n_batch)
        self.seed = int(self.configs.seed)
        self.language = self.configs.language.lower().strip()
        self.n_gpu_layers = int(self.configs.n_gpu_layers)
        self.max_tokens = int(self.configs.max_tokens)
        self.shard_size = int(self.configs.shard_size)
        self.checkpoint_interval = int(self.configs.checkpoint_interval)
        self.max_attempts = max(1, self.total_samples * int(self.configs.max_attempts_multiplier))
        self.min_user_words = int(self.configs.min_user_words)
        self.max_user_words = int(self.configs.max_user_words)
        self.min_assistant_words = int(self.configs.min_assistant_words)
        self.max_assistant_words = int(self.configs.max_assistant_words)
        self.min_quality_score = int(self.configs.min_quality_score)
        self.temperature = float(self.configs.temperature)
        self.top_p = float(self.configs.top_p)
        self.min_p = float(self.configs.min_p)
        self.repeat_penalty = float(self.configs.repeat_penalty)
        self.retry_count = int(self.configs.retry_count)
        self.enable_quality_judge = bool(self.configs.enable_quality_judge)
        self.judge_model_path = self.configs.judge_model_path
        self.random = random.Random(self.seed)
        self.llm = None
        self.judge_llm = None
        self.start_time = None
        self.accepted = 0
        self.attempts = 0
        self.signatures: Set[str] = set()
        self.user_signatures: Set[str] = set()
        self.logger = logging.getLogger("SyntheticDatasetGenerator")
        self.stats = {"attempts": 0, "accepted": 0, "generation_failed": 0, "json_failed": 0, "validation_failed": 0, "language_failed": 0, "quality_failed": 0, "duplicate_failed": 0}

        self.language_configs = {
            "fa": {
                "name": "Persian",
                "native": "فارسی",
                "script_min": 0.58,
                "prompt": "تمام محتوای سؤال کاربر و پاسخ دستیار باید به فارسی طبیعی، روان، حرفه‌ای و بومی نوشته شود. ساختار جمله‌ها باید شبیه نوشته و گفتار طبیعی یک فارسی‌زبان باشد و نباید ترجمه تحت‌اللفظی از انگلیسی به فارسی باشد. از نیم‌فاصله فارسی در ترکیبات مناسب مانند «می‌شود»، «می‌کند»، «نرم‌افزارها»، «داده‌ها»، «بهینه‌سازی» و موارد مشابه استفاده کن. از حروف فارسی «ی» و «ک» استفاده کن و از حروف عربی «ي» و «ك» استفاده نکن. از علائم نگارشی فارسی مانند «،»، «؛»، «؟» و «»» در جای مناسب استفاده کن. واژه‌های انگلیسی فقط در مواردی مانند نام فناوری، نام محصول، کد، نام زبان برنامه‌نویسی، مخفف، استاندارد یا اصطلاح تخصصی رایج مجاز هستند.",
                "tasks": ["پرسش و پاسخ", "توضیح مفهوم", "مقایسه", "حل مسئله", "استدلال", "خلاصه‌سازی", "دسته‌بندی", "ترجمه", "بازنویسی", "عیب‌یابی", "آموزش مرحله‌به‌مرحله", "تصمیم‌گیری", "تحلیل مفهوم", "ارائه مثال", "راهنمای عملی", "تحلیل علت و معلول", "بررسی مزایا و معایب", "تحلیل سناریو", "تحلیل خطا", "ارائه پیشنهاد", "ارزیابی", "تفسیر", "طراحی راهکار", "برنامه‌ریزی"],
                "styles": ["کوتاه و دقیق", "توضیحی و کامل", "مرحله‌به‌مرحله", "آموزشی برای مبتدی", "فنی و تخصصی", "عملی", "تحلیلی", "مقایسه‌ای", "عیب‌یابی", "مبتنی بر سناریو", "مبتنی بر استدلال", "مبتنی بر مثال", "مختصر اما کامل", "ساده و قابل فهم", "پیشرفته و تخصصی"],
                "audiences": ["کاربر عمومی", "مبتدی", "دانش‌آموز", "دانشجو", "برنامه‌نویس", "مهندس", "پژوهشگر", "مدیر", "کارشناس کسب‌وکار", "متخصص فنی", "کاربر حرفه‌ای"],
                "question_styles": ["پرسش مستقیم", "پرسش مبتنی بر سناریو", "پرسش مسئله‌محور", "پرسش چگونه", "پرسش چرا", "پرسش اگر", "پرسش مقایسه‌ای", "پرسش عیب‌یابی", "پرسش مفهومی", "درخواست عملی", "پرسش چندبخشی", "پرسش تصمیم‌محور"],
                "bad_patterns": ["به عنوان یک مدل زبانی", "به عنوان هوش مصنوعی", "به عنوان یک دستیار هوش مصنوعی", "امیدوارم این پاسخ مفید باشد", "اگر سؤال دیگری دارید", "در صورت داشتن هرگونه سؤال دیگر", "من نمی‌توانم به اینترنت دسترسی داشته باشم"]
            },
            "en": {
                "name": "English",
                "native": "English",
                "script_min": 0.65,
                "prompt": "All user and assistant content must be written in natural, fluent, idiomatic English. Avoid literal translations, unnatural phrasing and repetitive templates.",
                "tasks": ["Question answering", "Explanation", "Comparison", "Problem solving", "Reasoning", "Summarization", "Classification", "Translation", "Rewriting", "Troubleshooting", "Step-by-step instruction", "Decision making", "Concept analysis", "Example generation", "Practical guidance", "Cause and effect analysis", "Advantages and disadvantages", "Scenario analysis", "Error analysis", "Evaluation"],
                "styles": ["Short and precise", "Detailed explanatory", "Step-by-step", "Educational", "Technical", "Practical", "Analytical", "Comparative", "Troubleshooting", "Scenario-based", "Reasoning-focused", "Example-driven", "Concise but complete", "Beginner-friendly", "Expert-level"],
                "audiences": ["General user", "Beginner", "Student", "Developer", "Engineer", "Researcher", "Manager", "Business professional", "Technical professional", "Experienced practitioner"],
                "question_styles": ["Direct question", "Scenario-based question", "Problem-based question", "How-to question", "Why question", "What-if question", "Comparison question", "Troubleshooting question", "Conceptual question", "Practical request", "Multi-part question", "Decision-oriented question"],
                "bad_patterns": ["as an ai", "as an ai language model", "i hope this helps", "if you have any further questions", "i cannot browse the internet"]
            }
        }

        self.use_mmap = self.configs.load_model_use_mmap,
        self.use_mlock = self.configs.load_model_use_mlock
        self.verbose = self.configs.load_model_verbose

        if self.language not in self.language_configs:
            raise ValueError(f"Unsupported language: {self.language}")

        self.selected_lang_config = self.language_configs[self.language]
        self.topics = self.configs.topics[self.language]
        self.tasks = self.selected_lang_config["tasks"]
        self.styles = self.selected_lang_config["styles"]
        self.audiences = self.selected_lang_config["audiences"]
        self.question_styles = self.selected_lang_config["question_styles"]
        self.export_final = self.configs.export_final
        self.cleanup_shards = self.configs.cleanup_shards

        self.multi_turn = bool(self.configs.multi_turn)
        self.min_turns = int(self.configs.min_turns)
        self.max_turns = int(self.configs.max_turns)

        if not isinstance(self.topics, list) or not self.topics:
            raise ValueError("Topics configuration must be a non-empty list")

        self.topics = [str(topic).strip() for topic in self.topics if str(topic).strip()]

    def _validate_config(self) -> None:
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self.total_samples <= 0:
            raise ValueError("total_samples must be greater than zero")
        if self.n_ctx <= 0 or self.n_threads <= 0 or self.n_batch <= 0:
            raise ValueError("n_ctx, n_threads and n_batch must be greater than zero")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be greater than zero")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than zero")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be between 0 and 1")
        if self.min_user_words <= 0 or self.max_user_words < self.min_user_words:
            raise ValueError("Invalid user word limits")
        if self.min_assistant_words <= 0 or self.max_assistant_words < self.min_assistant_words:
            raise ValueError("Invalid assistant word limits")
        if self.enable_quality_judge and not self.judge_model_path:
            raise ValueError("judge_model_path is required when enable_quality_judge=True")
        if self.multi_turn:
            if self.min_turns <= 1:
                raise ValueError("multi_turn requires min_turns > 1")

            if self.max_turns < self.min_turns:
                raise ValueError("max_turns must be greater than or equal to min_turns")
        else:
            self.min_turns = 1
            self.max_turns = 1

    def _system_prompt(self) -> str:
        result= f"""You are a professional synthetic instruction-tuning dataset generator.

    Do not use reasoning or thinking mode.
    Do not generate <think> or </think> tags.
    Answer directly without hidden reasoning.

    Your target language is {self.selected_lang_config['name']}.
    {self.selected_lang_config['prompt']}

    Generate realistic, diverse, accurate, useful and natural user-assistant conversations.

    Avoid:
    - artificial prompts
    - repetitive templates
    - generic filler
    - fabricated information
    - unnecessary verbosity
    - meta commentary
    - references to the dataset
    - references to generation instructions
    - statements about being an AI
    - reasoning traces
    - chain-of-thought

    For medical topics provide general educational information only and never diagnose a person, prescribe treatment or invent clinical facts.

    Return only valid JSON."""
        self.logger.info(f"_system_prompt={result}")
        return result

    def load_model(self) -> None:
        self.logger.info(f"Loading generation model: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            use_mmap=self.configs.load_model_use_mmap,
            use_mlock=self.configs.load_model_use_mlock,
            verbose=self.configs.load_model_verbose,
            seed=self.seed
        )
        self.logger.info("Generation model loaded successfully")

    def load_judge_model(self) -> None:
        if not self.enable_quality_judge:
            return
        if not os.path.isfile(self.judge_model_path):
            raise FileNotFoundError(f"Judge model not found: {self.judge_model_path}")
        self.logger.info(f"Loading judge model: {self.judge_model_path}")
        self.judge_llm = Llama(model_path=self.judge_model_path,
                               n_ctx=self.n_ctx, n_threads=self.n_threads,
                               n_batch=self.n_batch, n_gpu_layers=self.n_gpu_layers,
                               use_mmap=True, use_mlock=False, verbose=True, seed=self.seed + 1000000)
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

    def _latin_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        latin = sum(1 for c in letters if c.isascii() and c.isalpha())
        return latin / len(letters)

    def _arabic_character_ratio(self, text: str) -> float:
        chars = [c for c in text if c.isalpha()]
        if not chars:
            return 0.0
        arabic = sum(1 for c in chars if c in "يىكؤإأة")
        return arabic / len(chars)

    def _persian_spacing_score(self, text: str) -> float:
        if self.language != "fa":
            return 1.0
        words = self._words(text)
        if len(words) < 20:
            return 1.0
        common_compounds = ["می", "نمی", "ها", "های", "تر", "ترین", "شده", "شوند", "کند", "کنند"]
        possible = 0
        correct = 0
        for word in words:
            for prefix in ["می", "نمی"]:
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    possible += 1
                    if "\u200c" in word:
                        correct += 1
            for suffix in ["ها", "های", "تر", "ترین"]:
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
        if self.language != "fa":
            return False
        words = self._words(text)
        if not words:
            return True
        latin_words = sum(1 for word in words if re.search(r"[a-zA-Z]", word))
        return latin_words / len(words) > 0.28

    def _has_invalid_persian_characters(self, text: str) -> bool:
        if self.language != "fa":
            return False
        return any(char in text for char in ["ي", "ك", "ى", "ة"])

    def _has_bad_punctuation(self, text: str) -> bool:
        if self.language != "fa":
            return False
        if "؟؟؟" in text or "!!!" in text:
            return True
        if text.count("...") > 4:
            return True
        if re.search(r" {2,}", text):
            return True
        return False

    def _language_quality(self, text: str) -> float:
        if self.language != "fa":
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

        if self.multi_turn:
            intro_fa = "یک نمونه مکالمه چندمرحله‌ای باکیفیت برای دیتاست آموزش و فاین‌تیون مدل زبانی تولید کن."
            intro_en = "Generate a high-quality multi-turn instruction-tuning example."

            turn_instruction_fa = f"""
    یک گفت‌وگوی چندمرحله‌ای تولید کن.

    تعداد نوبت‌های گفت‌وگو باید بین {self.min_turns} و {self.max_turns} نوبت باشد.
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

    The conversation must contain between {self.min_turns} and {self.max_turns} turns.
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

        if self.language == "fa":
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

    {"در حالت چندمرحله‌ای، messages باید با همین الگو ادامه پیدا کند: user → assistant → user → assistant → ..." if self.multi_turn else "در حالت تک‌مرحله‌ای دقیقاً فقط دو پیام تولید کن: user → assistant."}
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

    {"For multi-turn mode, continue the same pattern: user → assistant → user → assistant → ..." if self.multi_turn else "For single-turn mode, return exactly two messages: user → assistant."}
    """

    def _validate_structure(self, sample: Any) -> Tuple[bool, str]:
        if not isinstance(sample, dict):
            return False, "not_object"

        messages = sample.get("messages")

        if not isinstance(messages, list):
            return False, "invalid_messages"

        if self.multi_turn:
            turns = len(messages) // 2

            if turns < self.min_turns or turns > self.max_turns:
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

        if total_user_words < self.min_user_words:
            return False, "user_too_short"

        if total_user_words > self.max_user_words:
            return False, "user_too_long"

        if total_assistant_words < self.min_assistant_words:
            return False, "assistant_too_short"

        if total_assistant_words > self.max_assistant_words:
            return False, "assistant_too_long"

        return True, "ok"

    def _validate_language(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        messages = sample["messages"]

        for i, message in enumerate(messages):
            text = message["content"]

            if self.language == "fa":
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

        if self.language == "fa":
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

            if self.language == "fa":
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
        if not self.enable_quality_judge:
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
            result = self.judge_llm.create_chat_completion(messages=[{"role": "system", "content": "You are a strict dataset quality evaluator. Return only valid JSON."}, {"role": "user", "content": prompt}], temperature=0.1, top_p=0.9, max_tokens=200, response_format={"type": "json_object"}, seed=self.seed + self.accepted)
            judgment = json.loads(result["choices"][0]["message"]["content"].strip())
            return bool(judgment.get("acceptable", False)) and int(judgment.get("score", 0)) >= self.min_quality_score
        except Exception as exc:
            self.logger.warning("Judge failure: %s", exc)
            return False

    def _generate_sample(self, index: int) -> Dict[str, Any]:
        topic = self.random.choice(self.topics)
        task = self.random.choice(self.tasks)
        style = self.random.choice(self.styles)
        difficulty = self.random.choice(["مبتدی", "متوسط", "پیشرفته", "تخصصی"] if self.language == "fa" else ["Beginner", "Intermediate", "Advanced", "Expert"])
        audience = self.random.choice(self.audiences)
        question_style = self.random.choice(self.question_styles)
        prompt = self._build_prompt(topic, task, style, difficulty, audience, question_style, index)
        self.logger.info(f"Generating sample: index={index}, topic={topic}, task={task}, retry_count={self.retry_count}")

        for retry in range(self.retry_count):
            try:
                self.logger.info(f"Generation attempt: index={index}, retry={retry + 1}/{self.retry_count}")

                temperature = min(
                    0.95,
                    max(0.55, self.temperature + self.random.uniform(-0.08, 0.08))
                )

                
                self.logger.info("=" * 80)
                self.logger.info(f"START GENERATION | index={index} | retry={retry + 1}/{self.retry_count}")
                self.logger.info("=" * 80)

                generation_start = time.time()

                stream = self.llm.create_chat_completion(
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
                    temperature=temperature,
                    top_p=self.top_p,
                    min_p=self.min_p,
                    repeat_penalty=self.repeat_penalty,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    seed=self.seed + index * 100 + retry,
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
                    self.stats["json_failed"] += 1
                    self.logger.info(f"JSON parsing failed: index={index}, retry={retry + 1}")
                    continue

                valid, _ = self._validate_structure(sample)
                if not valid:
                    self.stats["validation_failed"] += 1
                    self.logger.info(f"Structure validation failed: index={index}, retry={retry + 1}")
                    continue

                sample = self._normalize_sample(sample)
                valid, _ = self._validate_language(sample)
                if not valid:
                    self.stats["language_failed"] += 1
                    self.logger.info(f"Language validation failed: index={index}, retry={retry + 1}")
                    continue

                if self._quality_score(sample) < self.min_quality_score:
                    self.stats["quality_failed"] += 1
                    self.logger.info(f"Quality validation failed: index={index}, retry={retry + 1}")
                    continue

                if not self._judge(sample):
                    self.stats["quality_failed"] += 1
                    self.logger.info(f"Quality judge rejected sample: index={index}, retry={retry + 1}")
                    continue

                self.logger.info(f"Sample accepted: index={index}, retry={retry + 1}")
                return sample

            except Exception as exc:
                self.stats["generation_failed"] += 1
                self.logger.info(f"Generation error: index={index}, retry={retry + 1}, error={exc}")
                self.logger.warning("Generation failure index=%s retry=%s error=%s", index, retry + 1, exc)

        self.logger.info(f"Sample generation failed after {self.retry_count} retries: index={index}")
        return {}

    def _output_dir(self) -> str:
        directory = os.path.dirname(self.output_path) or "."
        os.makedirs(directory, exist_ok=True)
        return directory

    def _checkpoint_dir(self) -> str:
        directory = os.path.join(self._output_dir(), ".checkpoints")
        os.makedirs(directory, exist_ok=True)
        return directory

    def _checkpoint_path(self) -> str:
        return os.path.join(self._checkpoint_dir(), "state.json")

    def _shard_path(self, index: int) -> str:
        base = os.path.splitext(os.path.basename(self.output_path))[0]
        return os.path.join(self._output_dir(), f"{base}-{index:06d}.parquet")

    def _existing_shards(self) -> List[str]:
        base = os.path.splitext(os.path.basename(self.output_path))[0]
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
        state = {"next_index": next_index, "accepted": self.accepted, "attempts": self.attempts, "stats": self.stats, "signatures": list(self.signatures), "user_signatures": list(self.user_signatures)}
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
        self.logger.info(f"Target samples: {self.total_samples}")
        self.logger.info(f"Language: {self.language}")
        self.logger.info(f"Output: {self.output_path}")
        self.logger.info("=" * 80)

        self.load_model()
        self.load_judge_model()

        existing_shards = self._existing_shards()

        if existing_shards:
            self.logger.info(f"Found {len(existing_shards)} existing shard(s)")

        current_shard: List[Dict[str, Any]] = []
        shard_index = len(existing_shards)
        next_index = self.accepted

        while self.accepted < self.total_samples and self.attempts < self.max_attempts:
            self.attempts += 1
            self.stats["attempts"] = self.attempts

            sample = self._generate_sample(self.attempts)

            if not sample:
                continue

            signature = self._signature(sample)
            user_signature = self._user_signature(sample)

            if signature in self.signatures or user_signature in self.user_signatures:
                self.stats["duplicate_failed"] += 1
                self.logger.info(f"Duplicate sample rejected: index={self.attempts}")
                continue

            self.signatures.add(signature)
            self.user_signatures.add(user_signature)

            current_shard.append(sample)

            self.accepted += 1
            self.stats["accepted"] = self.accepted
            next_index = self.accepted

            self.logger.info(
                f"Accepted: {self.accepted}/{self.total_samples} "
                f"| Attempts: {self.attempts}/{self.max_attempts}"
            )

            if len(current_shard) >= self.shard_size:
                self._save_shard(current_shard, shard_index)
                shard_index += 1
                current_shard = []

            if self.accepted % self.checkpoint_interval == 0:
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
        self.logger.info(f"Stats: {self.stats}")
        self.logger.info("=" * 80)