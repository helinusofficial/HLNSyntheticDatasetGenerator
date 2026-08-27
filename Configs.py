
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
        self.config_text = f"""
        {"=" * 70}
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
        {"-" * 103}
        """

    def log(self):
        if self._logged:
            return
        self.logger.info(self.config_text)
        self._logged = True