import hashlib
import json
import logging
import os
import random
import re
import shutil
import time
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple
from datasets import Dataset, load_dataset
from llama_cpp import Llama


class SyntheticDatasetGenerator:

    def __init__(self, model_path: str, output_path: str, total_samples: int, n_ctx: int, n_threads: int, n_batch: int,
                 seed: int, language: str, n_gpu_layers: int, max_tokens: int, shard_size: int,
                 checkpoint_interval: int, max_attempts_multiplier: int, min_user_words: int,
                 max_user_words: int, min_assistant_words: int, max_assistant_words: int,
                 min_quality_score: int, temperature: float, top_p: float, min_p: float,
                 repeat_penalty: float, retry_count: int, enable_quality_judge: bool,
                 judge_model_path: Optional[str], keep_shards: bool, export_final: bool, cleanup_shards: bool):
        self.model_path = model_path
        self.output_path = output_path
        self.total_samples = int(total_samples)
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.n_batch = int(n_batch)
        self.seed = int(seed)
        self.language = language.lower().strip()
        self.n_gpu_layers = int(n_gpu_layers)
        self.max_tokens = int(max_tokens)
        self.shard_size = int(shard_size)
        self.checkpoint_interval = int(checkpoint_interval)
        self.max_attempts = max(1, self.total_samples * int(max_attempts_multiplier))
        self.min_user_words = int(min_user_words)
        self.max_user_words = int(max_user_words)
        self.min_assistant_words = int(min_assistant_words)
        self.max_assistant_words = int(max_assistant_words)
        self.min_quality_score = int(min_quality_score)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.repeat_penalty = float(repeat_penalty)
        self.retry_count = int(retry_count)
        self.enable_quality_judge = bool(enable_quality_judge)
        self.judge_model_path = judge_model_path
        self.keep_shards = bool(keep_shards)
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
                "topics": ["هوش مصنوعی", "یادگیری ماشین", "یادگیری عمیق", "پردازش زبان طبیعی", "بینایی ماشین", "مدل‌های زبانی بزرگ", "هوش مصنوعی مولد", "مهندسی پرامپت", "برنامه‌نویسی", "پایتون", "توسعه نرم‌افزار", "مهندسی نرم‌افزار", "الگوریتم‌ها و ساختمان داده", "پایگاه داده", "رایانش ابری", "دوآپس", "سیستم‌عامل", "شبکه‌های کامپیوتری", "امنیت سایبری", "توسعه وب", "توسعه اپلیکیشن موبایل", "علم داده", "تحلیل داده", "آمار", "ریاضیات", "فیزیک", "شیمی", "زیست‌شناسی", "پزشکی عمومی", "سلامت دیجیتال", "فناوری سلامت", "کسب‌وکار", "مدیریت", "اقتصاد", "بازاریابی", "آموزش", "تاریخ", "جغرافیا", "ترجمه", "خلاصه‌سازی متن", "تحلیل متن", "نگارش", "استدلال", "حل مسئله", "تصمیم‌گیری", "مقایسه مفاهیم", "پرسش و پاسخ عمومی", "توسعه فردی", "کارآفرینی", "مدیریت پروژه", "تجربه کاربری", "طراحی محصول", "سئو", "تولید محتوا"],
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
                "topics": ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Large Language Models", "Generative AI", "Prompt Engineering", "Programming", "Python", "Software Development", "Software Engineering", "Databases", "Cloud Computing", "Cybersecurity", "Data Science", "Statistics", "Mathematics", "Medicine", "Digital Health", "Business", "Management", "Economics", "Marketing", "Education", "History", "Geography", "Writing", "Reasoning", "Problem Solving", "Decision Making", "General Question Answering"],
                "tasks": ["Question answering", "Explanation", "Comparison", "Problem solving", "Reasoning", "Summarization", "Classification", "Translation", "Rewriting", "Troubleshooting", "Step-by-step instruction", "Decision making", "Concept analysis", "Example generation", "Practical guidance", "Cause and effect analysis", "Advantages and disadvantages", "Scenario analysis", "Error analysis", "Evaluation"],
                "styles": ["Short and precise", "Detailed explanatory", "Step-by-step", "Educational", "Technical", "Practical", "Analytical", "Comparative", "Troubleshooting", "Scenario-based", "Reasoning-focused", "Example-driven", "Concise but complete", "Beginner-friendly", "Expert-level"],
                "audiences": ["General user", "Beginner", "Student", "Developer", "Engineer", "Researcher", "Manager", "Business professional", "Technical professional", "Experienced practitioner"],
                "question_styles": ["Direct question", "Scenario-based question", "Problem-based question", "How-to question", "Why question", "What-if question", "Comparison question", "Troubleshooting question", "Conceptual question", "Practical request", "Multi-part question", "Decision-oriented question"],
                "bad_patterns": ["as an ai", "as an ai language model", "i hope this helps", "if you have any further questions", "i cannot browse the internet"]
            },
            "de": {
                "name": "German",
                "native": "Deutsch",
                "script_min": 0.65,
                "prompt": "Alle Benutzer- und Assistententexte müssen in natürlichem, idiomatischem Deutsch verfasst sein. Vermeide wörtliche Übersetzungen, unnatürliche Formulierungen und wiederholte Antwortmuster.",
                "topics": ["Künstliche Intelligenz", "Maschinelles Lernen", "Deep Learning", "Programmierung", "Python", "Softwareentwicklung", "Softwaretechnik", "Datenbanken", "Cybersicherheit", "Data Science", "Statistik", "Mathematik", "Medizin", "Digitale Gesundheit", "Wirtschaft", "Management", "Marketing", "Bildung", "Geschichte", "Geografie", "Textanalyse", "Zusammenfassung", "Problemlösung", "Logisches Denken", "Entscheidungsfindung"],
                "tasks": ["Frage und Antwort", "Erklärung", "Vergleich", "Problemlösung", "Schlussfolgerung", "Zusammenfassung", "Klassifikation", "Übersetzung", "Umschreibung", "Fehlerbehebung", "Schritt-für-Schritt-Anleitung", "Entscheidungshilfe", "Konzeptanalyse", "Beispielerstellung", "Praktische Anleitung"],
                "styles": ["Kurz und präzise", "Ausführlich erklärend", "Schritt für Schritt", "Lehrreich", "Technisch", "Praktisch", "Analytisch", "Vergleichend", "Fehlerbehebung", "Szenariobasiert", "Beispielorientiert", "Einfach verständlich", "Expertenniveau"],
                "audiences": ["Allgemeiner Nutzer", "Anfänger", "Student", "Entwickler", "Ingenieur", "Forscher", "Manager", "Geschäftskunde", "Technischer Fachmann"],
                "question_styles": ["Direkte Frage", "Szenariobasierte Frage", "Problembasierte Frage", "Wie-Frage", "Warum-Frage", "Was-wäre-wenn-Frage", "Vergleichsfrage", "Fehlerbehebungsfrage", "Konzeptionelle Frage", "Praktische Anfrage", "Mehrteilige Frage"],
                "bad_patterns": ["als ki", "als künstliche intelligenz", "ich hoffe, das hilft", "wenn sie weitere fragen haben"]
            }
        }

        if self.language not in self.language_configs:
            raise ValueError(f"Unsupported language: {self.language}")

        self.config = self.language_configs[self.language]
        self.topics = self.config["topics"]
        self.tasks = self.config["tasks"]
        self.styles = self.config["styles"]
        self.audiences = self.config["audiences"]
        self.question_styles = self.config["question_styles"]
        self.export_final=export_final
        self.cleanup_shards=cleanup_shards

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

    def _system_prompt(self) -> str:
        return f"You are a professional synthetic instruction-tuning dataset generator. Your target language is {self.config['name']}. {self.config['prompt']} Generate realistic, diverse, accurate, useful and natural user-assistant conversations. Avoid artificial prompts, repetitive templates, generic filler, fabricated information, unnecessary verbosity, meta commentary, references to the dataset, references to generation instructions, and statements about being an AI. For medical topics provide general educational information only and never diagnose a person, prescribe treatment or invent clinical facts. Return only valid JSON."

    def load_model(self) -> None:
        self.llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, n_batch=self.n_batch, n_gpu_layers=self.n_gpu_layers, use_mmap=True, use_mlock=False, verbose=False, seed=self.seed)

    def load_judge_model(self) -> None:
        if not self.enable_quality_judge:
            return
        if not os.path.isfile(self.judge_model_path):
            raise FileNotFoundError(f"Judge model not found: {self.judge_model_path}")
        self.judge_llm = Llama(model_path=self.judge_model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, n_batch=self.n_batch, n_gpu_layers=self.n_gpu_layers, use_mmap=True, use_mlock=False, verbose=False, seed=self.seed + 1000000)

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
        return any(self._normalize_text(pattern) in normalized for pattern in self.config["bad_patterns"])

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

    def _build_prompt(self, topic: str, task: str, style: str, difficulty: str, audience: str, question_style: str, index: int) -> str:
        if self.language == "fa":
            return f"""یک نمونه باکیفیت برای دیتاست آموزش و فاین‌تیون مدل زبانی تولید کن.

زبان هدف: فارسی
موضوع: {topic}
نوع کار: {task}
سطح دشواری: {difficulty}
مخاطب: {audience}
سبک پاسخ: {style}
نوع سؤال: {question_style}
شناسه تنوع: {index}

سؤال کاربر باید کاملاً طبیعی و شبیه سؤالی باشد که یک فارسی‌زبان واقعی در یک موقعیت واقعی می‌پرسد.
پاسخ دستیار باید دقیق، مفید، مرتبط، روان و متناسب با سؤال باشد.
از ترجمه تحت‌اللفظی از انگلیسی خودداری کن.
از ساختارهای تکراری و کلیشه‌ای استفاده نکن.
پاسخ‌ها را با عباراتی مانند «حتماً»، «البته»، «به طور کلی»، «امیدوارم این پاسخ مفید باشد» یا عبارت‌های مشابه به شکل تکراری شروع یا تمام نکن.
در متن فارسی از «ی» و «ک» فارسی استفاده کن.
از نیم‌فاصله در موارد مناسب مانند «می‌شود»، «می‌کند»، «داده‌ها»، «نرم‌افزارها»، «بهینه‌سازی» و موارد مشابه استفاده کن.
از علائم «،»، «؛»، «؟» و «»» به شکل طبیعی استفاده کن.
واژه‌های انگلیسی فقط زمانی استفاده شوند که اصطلاح تخصصی، نام فناوری، نام محصول، کد، زبان برنامه‌نویسی یا نام خاص باشند.
برای موضوعات پزشکی فقط اطلاعات عمومی و آموزشی ارائه کن و تشخیص، نسخه یا تصمیم درمانی شخصی ارائه نکن.
برای مسائل استدلالی، نتیجه و منطق لازم برای درک پاسخ را ارائه کن اما زنجیره تفکر خصوصی را افشا نکن.
خروجی فقط JSON معتبر باشد.

ساختار دقیق خروجی:
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
}}"""
        return f"""Generate exactly one high-quality instruction-tuning example.

Target language: {self.config['name']}
Topic: {topic}
Task type: {task}
Difficulty: {difficulty}
Audience: {audience}
Response style: {style}
Question style: {question_style}
Variation ID: {index}

The user request must be realistic and natural.
The assistant response must be accurate, useful, relevant and complete.
Avoid repetitive structures, generic filler, artificial benchmark prompts and meta commentary.
Return only valid JSON with exactly two messages: user and assistant."""

    def _validate_structure(self, sample: Any) -> Tuple[bool, str]:
        if not isinstance(sample, dict):
            return False, "not_object"
        messages = sample.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            return False, "invalid_message_count"
        if not isinstance(messages[0], dict) or not isinstance(messages[1], dict):
            return False, "invalid_message_objects"
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            return False, "invalid_roles"
        if set(messages[0].keys()) != {"role", "content"} or set(messages[1].keys()) != {"role", "content"}:
            return False, "invalid_keys"
        user = messages[0].get("content")
        assistant = messages[1].get("content")
        if not isinstance(user, str) or not isinstance(assistant, str):
            return False, "invalid_content_type"
        if not user.strip() or not assistant.strip():
            return False, "empty_content"
        if self._word_count(user) < self.min_user_words:
            return False, "user_too_short"
        if self._word_count(user) > self.max_user_words:
            return False, "user_too_long"
        if self._word_count(assistant) < self.min_assistant_words:
            return False, "assistant_too_short"
        if self._word_count(assistant) > self.max_assistant_words:
            return False, "assistant_too_long"
        if self._normalize_text(user) == self._normalize_text(assistant):
            return False, "identical_messages"
        return True, "ok"

    def _validate_language(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        user = sample["messages"][0]["content"]
        assistant = sample["messages"][1]["content"]
        if self.language == "fa":
            if self._persian_letter_ratio(user) < 0.58:
                return False, "user_not_persian"
            if self._persian_letter_ratio(assistant) < 0.58:
                return False, "assistant_not_persian"
            if self._arabic_character_ratio(user) > 0.03:
                return False, "user_arabic_characters"
            if self._arabic_character_ratio(assistant) > 0.03:
                return False, "assistant_arabic_characters"
            if self._has_excessive_latin(user):
                return False, "user_excessive_latin"
            if self._has_excessive_latin(assistant):
                return False, "assistant_excessive_latin"
            if self._has_invalid_persian_characters(user) or self._has_invalid_persian_characters(assistant):
                return False, "invalid_persian_characters"
        return True, "ok"

    def _quality_score(self, sample: Dict[str, Any]) -> int:
        user = sample["messages"][0]["content"]
        assistant = sample["messages"][1]["content"]
        score = 100
        assistant_words = self._word_count(assistant)
        unique_ratio = len(set(self._words(assistant))) / max(1, assistant_words)
        if self._repetition_ratio(assistant) > 0.22:
            score -= 15
        if self._sentence_repetition_ratio(assistant) > 0.15:
            score -= 15
        if unique_ratio < 0.42:
            score -= 12
        if self._contains_bad_pattern(assistant):
            score -= 30
        if self._has_bad_punctuation(assistant):
            score -= 8
        if self.language == "fa":
            if self._language_quality(user) < 0.72:
                score -= 10
            if self._language_quality(assistant) < 0.72:
                score -= 10
            if assistant_words > 60 and self._persian_spacing_score(assistant) < 0.25:
                score -= 3
        if self._word_count(user) < 8:
            score -= 5
        if assistant_words < 30:
            score -= 8
        if assistant_words > 750:
            score -= 5
        return max(0, min(100, score))

    def _normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        user = sample["messages"][0]["content"].strip()
        assistant = sample["messages"][1]["content"].strip()
        if self.language == "fa":
            user = self._normalize_persian_text(user)
            assistant = self._normalize_persian_text(assistant)
        else:
            user = unicodedata.normalize("NFKC", user)
            assistant = unicodedata.normalize("NFKC", assistant)
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}

    def _signature(self, sample: Dict[str, Any]) -> str:
        user = self._normalize_text(sample["messages"][0]["content"])
        assistant = self._normalize_text(sample["messages"][1]["content"])
        return hashlib.sha256(f"{user}\n{assistant}".encode("utf-8")).hexdigest()

    def _user_signature(self, sample: Dict[str, Any]) -> str:
        user = self._normalize_text(sample["messages"][0]["content"])
        return hashlib.sha256(user.encode("utf-8")).hexdigest()

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
        for retry in range(self.retry_count):
            try:
                temperature = min(0.95, max(0.55, self.temperature + self.random.uniform(-0.08, 0.08)))
                result = self.llm.create_chat_completion(messages=[{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": prompt}], temperature=temperature, top_p=self.top_p, min_p=self.min_p, repeat_penalty=self.repeat_penalty, max_tokens=self.max_tokens, response_format={"type": "json_object"}, seed=self.seed + index * 100 + retry)
                raw = result["choices"][0]["message"]["content"].strip()
                try:
                    sample = json.loads(raw)
                except json.JSONDecodeError:
                    self.stats["json_failed"] += 1
                    continue
                valid, _ = self._validate_structure(sample)
                if not valid:
                    self.stats["validation_failed"] += 1
                    continue
                sample = self._normalize_sample(sample)
                valid, _ = self._validate_language(sample)
                if not valid:
                    self.stats["language_failed"] += 1
                    continue
                if self._quality_score(sample) < self.min_quality_score:
                    self.stats["quality_failed"] += 1
                    continue
                if not self._judge(sample):
                    self.stats["quality_failed"] += 1
                    continue
                return sample
            except Exception as exc:
                self.stats["generation_failed"] += 1
                self.logger.warning("Generation failure index=%s retry=%s error=%s", index, retry + 1, exc)
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
        Dataset.from_list(samples).to_parquet(temporary)
        os.replace(temporary, path)

    def _save_checkpoint(self, next_index: int) -> None:
        state = {"next_index": next_index, "accepted": self.accepted, "attempts": self.attempts, "stats": self.stats, "signatures": list(self.signatures), "user_signatures": list(self.user_signatures)}
        temporary = f"{self._checkpoint_path()}.tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False)
        os.replace(temporary, self._checkpoint_path())

    def _load_checkpoint(self) -> int:
        path = self._checkpoint_path()
        if not os.path.isfile(path):
            return 0
        with open(path, "r", encoding="utf-8") as file:
            state = json.load(file)
        self.accepted = int(state.get("accepted", 0))
        self.attempts = int(state.get("attempts", 0))
        self.stats.update(state.get("stats", {}))
        self.signatures = set(state.get("signatures", []))
        self.user_signatures = set(state.get("user_signatures", []))
        return int(state.get("next_index", 0))

    def _load_existing_dedup_state(self) -> None:
        for path in self._existing_shards():
            try:
                dataset = load_dataset("parquet", data_files=path, split="train")
                for sample in dataset:
                    self.signatures.add(self._signature(sample))
                    self.user_signatures.add(self._user_signature(sample))
            except Exception as exc:
                self.logger.warning("Failed to load shard %s: %s", path, exc)

    def _progress(self) -> None:
        elapsed = max(0.001, time.time() - self.start_time)
        rate = self.accepted / elapsed * 60
        print(f"تولید {self.accepted}/{self.total_samples} | تلاش {self.attempts} | سرعت {rate:.2f}/دقیقه | JSON نامعتبر {self.stats['json_failed']} | اعتبارسنجی {self.stats['validation_failed']} | زبان {self.stats['language_failed']} | کیفیت {self.stats['quality_failed']} | تکراری {self.stats['duplicate_failed']}", end="\r", flush=True)

    def _generate(self) -> List[str]:
        self._validate_config()
        if self.llm is None:
            self.load_model()
        if self.enable_quality_judge and self.judge_llm is None:
            self.load_judge_model()
        self.start_time = time.time()
        next_index = self._load_checkpoint()
        self._load_existing_dedup_state()
        shard_index = len(self._existing_shards())
        buffer = []
        while self.accepted < self.total_samples:
            if self.attempts >= self.max_attempts:
                raise RuntimeError(f"Maximum attempts reached. accepted={self.accepted}, target={self.total_samples}, attempts={self.attempts}, stats={self.stats}")
            self.attempts += 1
            self.stats["attempts"] = self.attempts
            sample = self._generate_sample(next_index)
            next_index += 1
            if not sample:
                if self.attempts % self.checkpoint_interval == 0:
                    self._save_checkpoint(next_index)
                continue
            signature = self._signature(sample)
            user_signature = self._user_signature(sample)
            if signature in self.signatures or user_signature in self.user_signatures:
                self.stats["duplicate_failed"] += 1
                if self.attempts % self.checkpoint_interval == 0:
                    self._save_checkpoint(next_index)
                continue
            self.signatures.add(signature)
            self.user_signatures.add(user_signature)
            buffer.append(sample)
            self.accepted += 1
            self.stats["accepted"] = self.accepted
            if len(buffer) >= self.shard_size:
                self._save_shard(buffer, shard_index)
                shard_index += 1
                buffer = []
            if self.accepted % self.checkpoint_interval == 0:
                self._save_checkpoint(next_index)
            self._progress()
        if buffer:
            self._save_shard(buffer, shard_index)
        self._save_checkpoint(next_index)
        print()
        return self._existing_shards()

    def _export_final(self) -> str:
        shards = self._existing_shards()
        if not shards:
            raise RuntimeError("No dataset shards found")
        datasets = [load_dataset("parquet", data_files=path, split="train") for path in shards]
        columns = datasets[0].column_names
        merged = {column: [] for column in columns}
        for dataset in datasets:
            for column in columns:
                merged[column].extend(dataset[column])
        temporary = f"{self.output_path}.tmp"
        Dataset.from_dict(merged).to_parquet(temporary)
        os.replace(temporary, self.output_path)
        return self.output_path

    def _cleanup(self, remove_shards: bool, remove_checkpoint ) -> None:
        if remove_shards:
            for path in self._existing_shards():
                if os.path.isfile(path):
                    os.remove(path)
        if remove_checkpoint and os.path.isdir(self._checkpoint_dir()):
            shutil.rmtree(self._checkpoint_dir())

    def _get_stats(self) -> Dict[str, Any]:
        elapsed = max(0.001, time.time() - self.start_time) if self.start_time else 0.0
        result = dict(self.stats)
        result["elapsed_seconds"] = elapsed
        result["samples_per_minute"] = self.accepted / elapsed * 60 if elapsed else 0.0
        result["acceptance_rate"] = self.accepted / max(1, self.attempts)
        return result

    def run(self) -> str:
        self._generate()
        if self.export_final:
            result = self._export_final()
            if self.cleanup_shards:
                self._cleanup(self.remove_shards)
            print(f"Dataset saved: {result}")
            print(f"Samples: {self.accepted}")
            print(f"Stats: {json.dumps(self._get_stats(), ensure_ascii=False)}")
            return result
        print(f"Dataset shards saved in: {self._output_dir()}")
        print(f"Samples: {self.accepted}")
        print(f"Stats: {json.dumps(self._get_stats(), ensure_ascii=False)}")
        return self._output_dir()