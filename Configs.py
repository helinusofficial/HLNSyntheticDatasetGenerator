
class SyntheticDatasetConfig:
    def __init__(self, logger):
        self.logger=logger

        self.model_path = r"D:\Downloads\qwen2.5-3b-instruct-q4_k_m.gguf"
        self.output_path = r"./dataset/synthetic.parquet"
        self.total_samples = 1
        self.n_ctx = 2048
        self.n_threads = 8
        self.n_batch = 256
        self.max_tokens = 256

        self.n_gpu_layers = 0
        self.seed = 42
        self.language = "fa"

        self.shard_size = 100
        self.checkpoint_interval = 50
        self.max_attempts_multiplier = 3
        self.min_user_words = 3
        self.max_user_words = 100
        self.min_assistant_words = 10
        self.max_assistant_words = 300
        self.min_quality_score = 65
        self.temperature = 0.75
        self.top_p = 0.9
        self.min_p = 0.05
        self.repeat_penalty = 1.08
        self.retry_count = 1
        self.enable_quality_judge = False
        self.judge_model_path = None
        self.export_final = True
        self.cleanup_shards = False

        self.min_turns = 1
        self.max_turns = 1
        self.multi_turn = False

        self.topics = {
        "fa": [
            "هوش مصنوعی",
            "یادگیری ماشین",
            "یادگیری عمیق",
            "پردازش زبان طبیعی",
            "بینایی ماشین",
            "مدل‌های زبانی بزرگ",
            "هوش مصنوعی مولد",
            "مهندسی پرامپت",
            "رباتیک",
            "برنامه‌نویسی",
            "پایتون",
            "توسعه نرم‌افزار",
            "مهندسی نرم‌افزار",
            "الگوریتم‌ها و ساختمان داده",
            "پایگاه داده",
            "رایانش ابری",
            "دوآپس",
            "سیستم‌عامل",
            "شبکه‌های کامپیوتری",
            "امنیت سایبری",
            "توسعه وب",
            "توسعه اپلیکیشن موبایل",
            "علم داده",
            "تحلیل داده",
            "آمار",
            "ریاضیات",
            "فیزیک",
            "شیمی",
            "زیست‌شناسی",
            "پزشکی عمومی",
            "سلامت دیجیتال",
            "فناوری سلامت",
            "زیست‌فناوری",
            "کسب‌وکار",
            "مدیریت",
            "اقتصاد",
            "بازاریابی",
            "فروش",
            "کارآفرینی",
            "مدیریت پروژه",
            "تجربه کاربری",
            "طراحی محصول",
            "آموزش",
            "روان‌شناسی",
            "فلسفه",
            "تاریخ",
            "جغرافیا",
            "حقوق",
            "ترجمه",
            "زبان‌شناسی",
            "نگارش",
            "تولید محتوا",
            "سئو",
            "محیط زیست",
            "انرژی",
            "اینترنت اشیا"
        ],

        "en": [
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Natural Language Processing",
            "Computer Vision",
            "Large Language Models",
            "Generative AI",
            "Prompt Engineering",
            "Robotics",
            "Programming",
            "Python",
            "Software Development",
            "Software Engineering",
            "Algorithms and Data Structures",
            "Databases",
            "Cloud Computing",
            "DevOps",
            "Operating Systems",
            "Computer Networks",
            "Cybersecurity",
            "Web Development",
            "Mobile Application Development",
            "Data Science",
            "Data Analysis",
            "Statistics",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "General Medicine",
            "Digital Health",
            "Health Technology",
            "Biotechnology",
            "Business",
            "Management",
            "Economics",
            "Marketing",
            "Sales",
            "Entrepreneurship",
            "Project Management",
            "User Experience",
            "Product Design",
            "Education",
            "Psychology",
            "Philosophy",
            "History",
            "Geography",
            "Law",
            "Translation",
            "Linguistics",
            "Writing",
            "Content Creation",
            "SEO",
            "Environment",
            "Energy",
            "Internet of Things"
        ]
    }

        self._logged = False

        self.load_model_use_mmap=True,
        self.load_model_use_mlock=False,
        self.load_model_verbose=True,
        self.language_configs = {
            "fa": {
                "name": "Persian",
                "native": "فارسی",
                "script_min": 0.58,
                "prompt": "تمام محتوای سؤال کاربر و پاسخ دستیار باید به فارسی طبیعی، روان، حرفه‌ای و بومی نوشته شود. ساختار جمله‌ها باید شبیه نوشته و گفتار طبیعی یک فارسی‌زبان باشد و نباید ترجمه تحت‌اللفظی از انگلیسی به فارسی باشد. از نیم‌فاصله فارسی در ترکیبات مناسب مانند «می‌شود»، «می‌کند»، «نرم‌افزارها»، «داده‌ها»، «بهینه‌سازی» و موارد مشابه استفاده کن. از حروف فارسی «ی» و «ک» استفاده کن و از حروف عربی «ي» و «ك» استفاده نکن. از علائم نگارشی فارسی مانند «،»، «؛»، «؟» و «»» در جای مناسب استفاده کن. واژه‌های انگلیسی فقط در مواردی مانند نام فناوری، نام محصول، کد، نام زبان برنامه‌نویسی، مخفف، استاندارد یا اصطلاح تخصصی رایج مجاز هستند.",
                "tasks": ["پرسش و پاسخ", "توضیح مفهوم", "مقایسه", "حل مسئله", "استدلال", "خلاصه‌سازی", "دسته‌بندی",
                          "ترجمه", "بازنویسی", "عیب‌یابی", "آموزش مرحله‌به‌مرحله", "تصمیم‌گیری", "تحلیل مفهوم",
                          "ارائه مثال", "راهنمای عملی", "تحلیل علت و معلول", "بررسی مزایا و معایب", "تحلیل سناریو",
                          "تحلیل خطا", "ارائه پیشنهاد", "ارزیابی", "تفسیر", "طراحی راهکار", "برنامه‌ریزی"],
                "styles": ["کوتاه و دقیق", "توضیحی و کامل", "مرحله‌به‌مرحله", "آموزشی برای مبتدی", "فنی و تخصصی",
                           "عملی", "تحلیلی", "مقایسه‌ای", "عیب‌یابی", "مبتنی بر سناریو", "مبتنی بر استدلال",
                           "مبتنی بر مثال", "مختصر اما کامل", "ساده و قابل فهم", "پیشرفته و تخصصی"],
                "audiences": ["کاربر عمومی", "مبتدی", "دانش‌آموز", "دانشجو", "برنامه‌نویس", "مهندس", "پژوهشگر", "مدیر",
                              "کارشناس کسب‌وکار", "متخصص فنی", "کاربر حرفه‌ای"],
                "question_styles": ["پرسش مستقیم", "پرسش مبتنی بر سناریو", "پرسش مسئله‌محور", "پرسش چگونه", "پرسش چرا",
                                    "پرسش اگر", "پرسش مقایسه‌ای", "پرسش عیب‌یابی", "پرسش مفهومی", "درخواست عملی",
                                    "پرسش چندبخشی", "پرسش تصمیم‌محور"],
                "bad_patterns": ["به عنوان یک مدل زبانی", "به عنوان هوش مصنوعی", "به عنوان یک دستیار هوش مصنوعی",
                                 "امیدوارم این پاسخ مفید باشد", "اگر سؤال دیگری دارید",
                                 "در صورت داشتن هرگونه سؤال دیگر", "من نمی‌توانم به اینترنت دسترسی داشته باشم"]
            },
            "en": {
                "name": "English",
                "native": "English",
                "script_min": 0.65,
                "prompt": "All user and assistant content must be written in natural, fluent, idiomatic English. Avoid literal translations, unnatural phrasing and repetitive templates.",
                "tasks": ["Question answering", "Explanation", "Comparison", "Problem solving", "Reasoning",
                          "Summarization", "Classification", "Translation", "Rewriting", "Troubleshooting",
                          "Step-by-step instruction", "Decision making", "Concept analysis", "Example generation",
                          "Practical guidance", "Cause and effect analysis", "Advantages and disadvantages",
                          "Scenario analysis", "Error analysis", "Evaluation"],
                "styles": ["Short and precise", "Detailed explanatory", "Step-by-step", "Educational", "Technical",
                           "Practical", "Analytical", "Comparative", "Troubleshooting", "Scenario-based",
                           "Reasoning-focused", "Example-driven", "Concise but complete", "Beginner-friendly",
                           "Expert-level"],
                "audiences": ["General user", "Beginner", "Student", "Developer", "Engineer", "Researcher", "Manager",
                              "Business professional", "Technical professional", "Experienced practitioner"],
                "question_styles": ["Direct question", "Scenario-based question", "Problem-based question",
                                    "How-to question", "Why question", "What-if question", "Comparison question",
                                    "Troubleshooting question", "Conceptual question", "Practical request",
                                    "Multi-part question", "Decision-oriented question"],
                "bad_patterns": ["as an ai", "as an ai language model", "i hope this helps",
                                 "if you have any further questions", "i cannot browse the internet"]
            }
        }
        self.stats = {"attempts": 0, "accepted": 0, "generation_failed": 0, "json_failed": 0, "validation_failed": 0,
                      "language_failed": 0, "quality_failed": 0, "duplicate_failed": 0}
        self.system_prompt = """
        You are a professional synthetic instruction-tuning dataset generator.

        Do not use reasoning or thinking mode.
        Do not generate <think> or </think> tags.
        Answer directly without hidden reasoning.

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

        Return only valid JSON.
        """
        self.config_text = f"""\n{"=" * 70}
 SyntheticDatasetConfig Configuration
 {"=" * 70}
 Model_Path                       : {self.model_path}
 Output_Path                      : {self.output_path}
 Total_Samples                    : {self.total_samples}
 N_Context                        : {self.n_ctx}
 N_Threads                        : {self.n_threads}
 N_Batch                          : {self.n_batch}
 Max_Tokens                       : {self.max_tokens}
 N_GPU_Layers                     : {self.n_gpu_layers}
 Seed                             : {self.seed}
 Language                         : {self.language}
 Shard_Size                       : {self.shard_size}
 Checkpoint_Interval              : {self.checkpoint_interval}
 Max_Attempts_Multiplier          : {self.max_attempts_multiplier}
 Min_User_Words                   : {self.min_user_words}
 Max_User_Words                   : {self.max_user_words}
 Min_Assistant_Words              : {self.min_assistant_words}
 Max_Assistant_Words              : {self.max_assistant_words}
 Min_Quality_Score                : {self.min_quality_score}
 Temperature                      : {self.temperature}
 Top_P                            : {self.top_p}
 Min_P                            : {self.min_p}
 Repeat_Penalty                   : {self.repeat_penalty}
 Retry_Count                      : {self.retry_count}
 Enable_Quality_Judge             : {self.enable_quality_judge}
 Judge_Model_Path                 : {self.judge_model_path}
 Export_Final                     : {self.export_final}
 Cleanup_Shards                   : {self.cleanup_shards}
 Min_Turns                        : {self.min_turns}
 Max_Turns                        : {self.max_turns}
 Multi_Turn                       : {self.multi_turn}
 Topics_Languages                 : {list(self.topics.keys())}
 Topics_Count_FA                  : {len(self.topics.get("fa", []))}
 Topics_Count_EN                  : {len(self.topics.get("en", []))}

 load_model_use_mmap              : {self.load_model_use_mmap}
 load_model_use_mlock             : {self.load_model_use_mlock}
 load_model_verbose               : {self.load_model_verbose}"""

    def log(self):
        if self._logged:
            return
        self.logger.info(self.config_text)
        self._logged = True