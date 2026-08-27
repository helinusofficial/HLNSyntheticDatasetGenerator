class SyntheticDatasetConfig:
    model_path = r"D:\Downloads\qwen2.5-3b-instruct-q4_k_m.gguf"
    output_path = r"./dataset/synthetic.parquet"
    total_samples = 1
    n_ctx = 2048
    n_threads = 8
    n_batch = 256
    max_tokens = 256

    n_gpu_layers = 0
    seed = 42
    language = "fa"

    shard_size = 100
    checkpoint_interval = 50
    max_attempts_multiplier = 3
    min_user_words = 3
    max_user_words = 100
    min_assistant_words = 10
    max_assistant_words = 300
    min_quality_score = 65
    temperature = 0.75
    top_p = 0.9
    min_p = 0.05
    repeat_penalty = 1.08
    retry_count = 1
    enable_quality_judge = False
    judge_model_path = None
    export_final = True
    cleanup_shards = False

    min_turns = 1
    max_turns = 1
    multi_turn = False

    topics = {
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