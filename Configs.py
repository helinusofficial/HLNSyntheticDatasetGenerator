import re

class SyntheticDatasetConfig:
    def __init__(self, logger):
        self.logger = logger

        self.model_path = r"D:\Downloads\qwen2.5-3b-instruct-q4_k_m.gguf"
        self.output_path = r"./dataset/synthetic.parquet"
        self.total_samples = 10
        self.n_ctx = 2048
        self.n_threads = 4
        self.n_batch = 1024
        self.max_tokens = 512

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
        self.retry_count = 3
        self.enable_quality_judge = False
        self.judge_model_path = None
        self.export_final = True
        self.cleanup_shards = False

        self.min_turns = 2
        self.max_turns = 3
        self.multi_turn = True

        self.topics = {
            "fa": [
                # مکالمات روزمره
                "احوالپرسی",
                "معرفی خود",
                "معرفی دیگران",
                "آشنایی با یک فرد جدید",
                "گفت‌وگوی دوستانه",
                "گفت‌وگوی خانوادگی",
                "صحبت درباره روزمره",
                "صحبت درباره برنامه روزانه",
                "صحبت درباره علایق",
                "صحبت درباره سرگرمی‌ها",
                "صحبت درباره اوقات فراغت",
                "صحبت درباره کار",
                "صحبت درباره تحصیل",
                "صحبت درباره زندگی شخصی",
                "صحبت درباره برنامه‌های آینده",
                "صحبت درباره خاطرات",
                "صحبت درباره تجربه‌های گذشته",
                "صحبت درباره تصمیم‌های شخصی",
                "صحبت درباره ترجیحات",
                "صحبت درباره عادت‌ها",

                # تعامل اجتماعی
                "شروع یک گفت‌وگوی جدید",
                "ادامه دادن یک گفت‌وگو",
                "پایان دادن مؤدبانه به مکالمه",
                "پرسیدن سؤال از طرف مقابل",
                "پاسخ دادن به سؤال شخصی",
                "بیان نظر شخصی",
                "موافقت با نظر دیگران",
                "مخالفت محترمانه",
                "ابراز تعجب",
                "ابراز خوشحالی",
                "ابراز ناراحتی",
                "ابراز نگرانی",
                "ابراز علاقه",
                "ابراز بی‌علاقگی",
                "تشکر کردن",
                "عذرخواهی کردن",
                "پذیرش عذرخواهی",
                "تبریک گفتن",
                "همدردی کردن",
                "دلگرم کردن",
                "تعریف کردن از دیگران",
                "پاسخ به تعریف",
                "دعوت کردن",
                "پذیرش دعوت",
                "رد کردن دعوت",
                "پیشنهاد دادن",
                "پذیرش پیشنهاد",
                "رد کردن پیشنهاد",

                # درخواست و کمک
                "درخواست کمک",
                "درخواست اطلاعات",
                "درخواست توضیح",
                "درخواست توضیح بیشتر",
                "درخواست مثال",
                "درخواست راهنمایی",
                "درخواست پیشنهاد",
                "درخواست انجام یک کار",
                "درخواست تغییر یک کار",
                "درخواست اصلاح",
                "درخواست تأیید",
                "درخواست نظر",
                "درخواست بازخورد",
                "درخواست روشن شدن موضوع",
                "درخواست تکرار",
                "درخواست ساده‌تر توضیح دادن",
                "درخواست جزئیات بیشتر",
                "درخواست خلاصه کردن",

                # پرسش و پاسخ مکالمه‌ای
                "پرسش مستقیم",
                "پرسش تکمیلی",
                "پرسش پیگیری",
                "پرسش مرتبط با پاسخ قبلی",
                "پرسش درباره یک موضوع قبلی",
                "پرسش چندبخشی",
                "پرسش غیرمستقیم",
                "پرسش مبهم",
                "رفع ابهام",
                "اصلاح سوءتفاهم",
                "تغییر سؤال",
                "بازگشت به موضوع قبلی",
                "ادامه دادن بر اساس پاسخ قبلی",

                # حل مسئله
                "بیان یک مشکل",
                "شرح یک مشکل",
                "درخواست راه‌حل",
                "پیدا کردن علت مشکل",
                "عیب‌یابی",
                "بررسی چند راه‌حل",
                "انتخاب بین چند گزینه",
                "تصمیم‌گیری",
                "درخواست توصیه",
                "بررسی مزایا و معایب",
                "بررسی پیامدهای یک تصمیم",
                "حل اختلاف",
                "حل یک سوءتفاهم",
                "رسیدن به توافق",
                "مذاکره",
                "چانه‌زنی",

                # مکالمات کاری
                "گفت‌وگوی کاری",
                "گفت‌وگو با همکار",
                "گفت‌وگو با مدیر",
                "گفت‌وگو با کارمند",
                "گفت‌وگو با مشتری",
                "گفت‌وگو با کارفرما",
                "درخواست از همکار",
                "پیگیری یک کار",
                "گزارش وضعیت کار",
                "هماهنگی کاری",
                "هماهنگی جلسه",
                "تغییر زمان جلسه",
                "لغو جلسه",
                "درخواست مرخصی",
                "تقسیم وظایف",
                "تحویل کار",
                "دریافت بازخورد کاری",
                "پاسخ به انتقاد کاری",
                "ارائه پیشنهاد کاری",
                "مذاکره کاری",
                "حل اختلاف در محیط کار",
                "گفت‌وگوی استخدامی",
                "مصاحبه شغلی",
                "مکالمه با مشتری ناراضی",

                # خرید و خدمات
                "خرید محصول",
                "پرس‌وجو درباره محصول",
                "پرس‌وجو درباره قیمت",
                "پرس‌وجو درباره ویژگی‌ها",
                "مقایسه محصولات",
                "انتخاب محصول",
                "درخواست تخفیف",
                "ثبت سفارش",
                "پیگیری سفارش",
                "مشکل در سفارش",
                "مشکل در پرداخت",
                "درخواست بازگشت کالا",
                "تعویض کالا",
                "شکایت از خدمات",
                "پاسخ به شکایت",
                "درخواست پشتیبانی",
                "گفت‌وگو با پشتیبانی",
                "رزرو خدمات",
                "لغو رزرو",
                "تغییر رزرو",

                # موقعیت‌های واقعی
                "مکالمه در فروشگاه",
                "مکالمه در رستوران",
                "مکالمه در کافه",
                "مکالمه در هتل",
                "مکالمه در فرودگاه",
                "مکالمه در ایستگاه",
                "مکالمه در تاکسی",
                "مکالمه در بانک",
                "مکالمه در اداره",
                "مکالمه در دانشگاه",
                "مکالمه در مدرسه",
                "مکالمه در محل کار",
                "مکالمه تلفنی",
                "مکالمه آنلاین",
                "مکالمه با پشتیبانی فنی",
                "مکالمه برای رزرو",
                "مکالمه برای دریافت اطلاعات",
                "مکالمه هنگام سفر",

                # سفر
                "برنامه‌ریزی سفر",
                "انتخاب مقصد",
                "رزرو هتل",
                "رزرو بلیت",
                "پرسیدن مسیر",
                "پرسیدن آدرس",
                "گم شدن در مسیر",
                "مشکل در سفر",
                "تغییر برنامه سفر",
                "لغو سفر",
                "مکالمه با کارکنان هتل",
                "مکالمه در فرودگاه",
                "مکالمه هنگام ورود به هتل",

                # آموزش
                "پرسیدن سؤال آموزشی",
                "درخواست آموزش",
                "درخواست آموزش ساده",
                "درخواست آموزش مرحله‌به‌مرحله",
                "پرسیدن سؤال بعد از آموزش",
                "رفع ابهام آموزشی",
                "اصلاح یک پاسخ",
                "بررسی یک پاسخ",
                "تمرین مهارت",
                "تمرین مکالمه",
                "درخواست مثال آموزشی",
                "توضیح یک موضوع برای مبتدی",

                # فناوری و کار با ابزار
                "کمک برای استفاده از نرم‌افزار",
                "عیب‌یابی نرم‌افزار",
                "مشکل در کامپیوتر",
                "مشکل در تلفن همراه",
                "مشکل در اینترنت",
                "کمک برای تنظیمات",
                "کمک برای نصب برنامه",
                "کمک برای پیدا کردن خطا",
                "درخواست راهنمایی فنی",
                "گفت‌وگوی کاربر و پشتیبان فنی",

                # مکالمات مبتنی بر context
                "ارجاع به صحبت قبلی",
                "ارجاع به پاسخ قبلی",
                "ادامه یک درخواست قبلی",
                "اصلاح درخواست قبلی",
                "اضافه کردن اطلاعات جدید",
                "حذف بخشی از درخواست",
                "تغییر بخشی از درخواست",
                "تغییر هدف در ادامه مکالمه",
                "درخواست توضیح درباره پاسخ قبلی",
                "درخواست جزئیات درباره موضوع قبلی",
                "مقایسه با گزینه قبلی",
                "بازگشت به یک موضوع قبلی",
                "جمع‌بندی مکالمه",
                "رسیدن به نتیجه",
                "ادامه مکالمه پس از وقفه",

                # موقعیت‌های احساسی و انسانی
                "صحبت با فرد ناراحت",
                "صحبت با فرد نگران",
                "دلجویی",
                "همدردی",
                "آرام کردن یک فرد",
                "تشویق کردن",
                "حمایت در یک موقعیت دشوار",
                "پاسخ به خبر خوب",
                "پاسخ به خبر بد",
                "پاسخ به انتقاد",
                "پاسخ به ناراحتی",
                "مدیریت یک گفت‌وگوی حساس",

                # زبان و نوشتار
                "اصلاح جمله",
                "اصلاح متن",
                "بازنویسی جمله",
                "بازنویسی متن",
                "تغییر لحن",
                "ساده‌سازی متن",
                "رسمی کردن متن",
                "غیررسمی کردن متن",
                "پیشنهاد عبارت مناسب",
                "پیدا کردن واژه مناسب",
                "توضیح معنی عبارت",
                "ترجمه در قالب مکالمه",
                "پاسخ مناسب به یک پیام",
                "نوشتن پاسخ کوتاه",
                "نوشتن پاسخ مؤدبانه",

                # گفت‌وگوهای باز و آزاد
                "گفت‌وگوی آزاد",
                "گفت‌وگوی غیررسمی",
                "گفت‌وگوی رسمی",
                "گفت‌وگوی نیمه‌رسمی",
                "صحبت درباره یک موضوع عمومی",
                "بحث دوستانه",
                "گفت‌وگوی نظر محور",
                "گفت‌وگوی تجربه محور",
                "گفت‌وگوی سناریو محور",
                "گفت‌وگوی مسئله محور"
            ],

            "en": [
                "Greetings",
                "Introducing yourself",
                "Introducing someone else",
                "Meeting someone new",
                "Casual conversation",
                "Family conversation",
                "Talking about daily life",
                "Talking about routines",
                "Talking about interests",
                "Talking about hobbies",
                "Talking about free time",
                "Talking about work",
                "Talking about education",
                "Talking about personal life",
                "Talking about future plans",
                "Talking about memories",
                "Talking about past experiences",
                "Talking about personal decisions",
                "Talking about preferences",
                "Talking about habits",

                "Starting a conversation",
                "Continuing a conversation",
                "Ending a conversation politely",
                "Asking someone a question",
                "Answering a personal question",
                "Expressing an opinion",
                "Agreeing with someone",
                "Disagreeing politely",
                "Expressing surprise",
                "Expressing happiness",
                "Expressing sadness",
                "Expressing concern",
                "Expressing interest",
                "Expressing disinterest",
                "Thanking",
                "Apologizing",
                "Accepting an apology",
                "Congratulating",
                "Showing sympathy",
                "Encouraging someone",
                "Complimenting someone",
                "Responding to a compliment",
                "Inviting someone",
                "Accepting an invitation",
                "Declining an invitation",
                "Making a suggestion",
                "Accepting a suggestion",
                "Rejecting a suggestion",

                "Asking for help",
                "Asking for information",
                "Asking for an explanation",
                "Asking for more explanation",
                "Asking for an example",
                "Asking for guidance",
                "Asking for a recommendation",
                "Making a request",
                "Requesting a change",
                "Requesting a correction",
                "Asking for confirmation",
                "Asking for an opinion",
                "Asking for feedback",
                "Asking for clarification",
                "Asking someone to repeat",
                "Asking for a simpler explanation",
                "Requesting more details",
                "Asking for a summary",

                "Direct question",
                "Follow-up question",
                "Related question",
                "Question based on a previous answer",
                "Question about a previous topic",
                "Multi-part question",
                "Indirect question",
                "Ambiguous question",
                "Clarification",
                "Resolving a misunderstanding",
                "Changing a question",
                "Returning to a previous topic",
                "Continuing from a previous answer",

                "Describing a problem",
                "Explaining a problem",
                "Asking for a solution",
                "Finding the cause of a problem",
                "Troubleshooting",
                "Evaluating multiple solutions",
                "Choosing between options",
                "Decision making",
                "Asking for advice",
                "Comparing advantages and disadvantages",
                "Evaluating consequences",
                "Resolving a disagreement",
                "Resolving a misunderstanding",
                "Reaching an agreement",
                "Negotiation",
                "Bargaining",

                "Work conversation",
                "Coworker conversation",
                "Manager and employee conversation",
                "Employee and manager conversation",
                "Customer conversation",
                "Employer conversation",
                "Making a request at work",
                "Following up on work",
                "Reporting progress",
                "Work coordination",
                "Scheduling a meeting",
                "Rescheduling a meeting",
                "Canceling a meeting",
                "Requesting time off",
                "Delegating tasks",
                "Handing over work",
                "Receiving work feedback",
                "Responding to criticism",
                "Making a work proposal",
                "Business negotiation",
                "Resolving workplace conflict",
                "Job interview",
                "Customer complaint conversation",

                "Buying a product",
                "Product inquiry",
                "Price inquiry",
                "Asking about features",
                "Comparing products",
                "Choosing a product",
                "Asking for a discount",
                "Placing an order",
                "Following up on an order",
                "Order problem",
                "Payment problem",
                "Returning a product",
                "Exchanging a product",
                "Complaining about a service",
                "Responding to a complaint",
                "Requesting customer support",
                "Customer support conversation",
                "Booking a service",
                "Canceling a booking",
                "Changing a booking",

                "Store conversation",
                "Restaurant conversation",
                "Cafe conversation",
                "Hotel conversation",
                "Airport conversation",
                "Station conversation",
                "Taxi conversation",
                "Bank conversation",
                "Office conversation",
                "University conversation",
                "School conversation",
                "Workplace conversation",
                "Phone conversation",
                "Online conversation",
                "Technical support conversation",
                "Booking conversation",
                "Information desk conversation",
                "Travel conversation",

                "Planning a trip",
                "Choosing a destination",
                "Booking a hotel",
                "Booking a ticket",
                "Asking for directions",
                "Asking for an address",
                "Getting lost",
                "Travel problem",
                "Changing travel plans",
                "Canceling a trip",
                "Talking with hotel staff",
                "Airport conversation",
                "Hotel check-in conversation",

                "Asking an educational question",
                "Requesting instruction",
                "Asking for a simple explanation",
                "Requesting step-by-step instruction",
                "Asking a follow-up after learning",
                "Resolving a learning misunderstanding",
                "Correcting an answer",
                "Checking an answer",
                "Practicing a skill",
                "Conversation practice",
                "Asking for an educational example",
                "Explaining something to a beginner",

                "Getting help with software",
                "Software troubleshooting",
                "Computer problem",
                "Phone problem",
                "Internet problem",
                "Getting help with settings",
                "Installing an application",
                "Finding an error",
                "Technical guidance",
                "User and technical support conversation",

                "Referring to previous conversation",
                "Referring to a previous answer",
                "Continuing a previous request",
                "Correcting a previous request",
                "Adding new information",
                "Removing part of a request",
                "Changing part of a request",
                "Changing the goal during a conversation",
                "Asking about a previous answer",
                "Requesting details about a previous topic",
                "Comparing with a previous option",
                "Returning to an earlier topic",
                "Summarizing a conversation",
                "Reaching a conclusion",
                "Continuing after an interruption",

                "Talking with someone who is upset",
                "Talking with someone who is worried",
                "Comforting someone",
                "Showing empathy",
                "Calming someone down",
                "Encouraging someone",
                "Supporting someone in a difficult situation",
                "Responding to good news",
                "Responding to bad news",
                "Responding to criticism",
                "Responding to frustration",
                "Handling a sensitive conversation",

                "Sentence correction",
                "Text correction",
                "Sentence rewriting",
                "Text rewriting",
                "Changing tone",
                "Simplifying text",
                "Making text more formal",
                "Making text more casual",
                "Suggesting a suitable phrase",
                "Finding the right word",
                "Explaining an expression",
                "Conversational translation",
                "Writing a suitable reply",
                "Writing a short reply",
                "Writing a polite reply",

                "Open-ended conversation",
                "Informal conversation",
                "Formal conversation",
                "Semi-formal conversation",
                "General conversation",
                "Friendly discussion",
                "Opinion-based conversation",
                "Experience-based conversation",
                "Scenario-based conversation",
                "Problem-based conversation"
            ]
        }

        self._logged = False

        self.load_model_use_mmap = True
        self.load_model_use_mlock = False
        self.load_model_verbose = False

        self.judge_llm_use_mmap=True
        self.judge_llm_use_mlock=False
        self.judge_llm_verbose=False

        self.language_configs = {
            "fa": {
                "name": "Persian",
                "native": "فارسی",
                "script_min": 0.58,
                "prompt": "تمام محتوای سؤال کاربر و پاسخ دستیار باید به فارسی طبیعی، روان، حرفه‌ای و بومی نوشته شود. لحن مکالمه باید خودمانی، صمیمی و طبیعی باشد؛ مانند گفت‌وگوی روزمره بین دو فارسی‌زبان. از لحن رسمی، اداری، کتابی و خشک خودداری کن. از محاوره طبیعی استفاده کن، اما از عامیانه‌نویسی افراطی، اصطلاحات عجیب و لحن غیرحرفه‌ای پرهیز کن. ساختار جمله‌ها باید شبیه گفتار طبیعی یک فارسی‌زبان باشد و نباید ترجمه تحت‌اللفظی از انگلیسی به فارسی باشد. از نیم‌فاصله فارسی در ترکیبات مناسب مانند «می‌شود»، «می‌کند»، «نرم‌افزارها»، «داده‌ها»، «بهینه‌سازی» و موارد مشابه استفاده کن. از حروف فارسی «ی» و «ک» استفاده کن و از حروف عربی «ي» و «ك» استفاده نکن. از علائم نگارشی فارسی مانند «،»، «؛»، «؟» و «»» در جای مناسب استفاده کن. واژه‌های انگلیسی فقط در مواردی مانند نام فناوری، نام محصول، کد، نام زبان برنامه‌نویسی، مخفف، استاندارد یا اصطلاح تخصصی رایج مجاز هستند.",
                "tasks": [
                    "شروع مکالمه",
                    "ادامه مکالمه",
                    "پاسخ به پیام",
                    "پاسخ به سؤال",
                    "پرسیدن سؤال",
                    "پرسیدن سؤال تکمیلی",
                    "پرسیدن سؤال پیگیری",
                    "درخواست اطلاعات",
                    "درخواست کمک",
                    "درخواست توضیح",
                    "درخواست توضیح بیشتر",
                    "درخواست مثال",
                    "درخواست راهنمایی",
                    "درخواست پیشنهاد",
                    "درخواست توصیه",
                    "درخواست نظر",
                    "درخواست تأیید",
                    "درخواست تکرار",
                    "درخواست ساده‌سازی",
                    "درخواست جزئیات",
                    "رفع ابهام",
                    "رفع سوءتفاهم",
                    "پیگیری درخواست",
                    "اصلاح درخواست قبلی",
                    "تغییر درخواست",
                    "اضافه کردن اطلاعات",
                    "حذف اطلاعات از درخواست",
                    "ارجاع به صحبت قبلی",
                    "ارجاع به پاسخ قبلی",
                    "ادامه بر اساس context قبلی",
                    "بیان یک مشکل",
                    "شرح یک موقعیت",
                    "درخواست راه‌حل",
                    "عیب‌یابی",
                    "حل مسئله",
                    "بررسی راه‌حل‌ها",
                    "انتخاب بین گزینه‌ها",
                    "تصمیم‌گیری",
                    "مقایسه گزینه‌ها",
                    "بررسی مزایا و معایب",
                    "ارزیابی پیامدها",
                    "درخواست بازخورد",
                    "ارائه بازخورد",
                    "بیان نظر",
                    "بیان ترجیح",
                    "بیان موافقت",
                    "بیان مخالفت",
                    "مخالفت محترمانه",
                    "رسیدن به توافق",
                    "مذاکره",
                    "چانه‌زنی",
                    "پیشنهاد دادن",
                    "پذیرش پیشنهاد",
                    "رد کردن پیشنهاد",
                    "تشکر کردن",
                    "عذرخواهی کردن",
                    "پذیرش عذرخواهی",
                    "تبریک گفتن",
                    "همدردی کردن",
                    "دلگرم کردن",
                    "دعوت کردن",
                    "پذیرش دعوت",
                    "رد کردن دعوت",
                    "آرام کردن مخاطب",
                    "پاسخ همدلانه",
                    "پاسخ به انتقاد",
                    "پاسخ به شکایت",
                    "مدیریت گفت‌وگوی حساس",
                    "هماهنگی",
                    "برنامه‌ریزی",
                    "آموزش",
                    "آموزش مرحله‌به‌مرحله",
                    "تمرین",
                    "اصلاح زبان",
                    "اصلاح جمله",
                    "بازنویسی",
                    "تغییر لحن",
                    "ساده‌سازی",
                    "ترجمه در قالب مکالمه",
                    "نوشتن پاسخ مناسب",
                    "خلاصه‌سازی مکالمه",
                    "جمع‌بندی مکالمه",
                    "رسیدن به نتیجه",
                    "ادامه دادن یک سناریو",
                    "حل اختلاف",
                    "پشتیبانی",
                    "پشتیبانی فنی",
                    "گفت‌وگوی خدماتی",
                    "گفت‌وگوی کاری",
                    "گفت‌وگوی اجتماعی",
                    "گفت‌وگوی دوستانه",
                    "گفت‌وگوی رسمی",
                    "گفت‌وگوی غیررسمی",
                    "گفت‌وگوی آزاد"
                ],
                "styles": [
                    "محاوره‌ای",
                    "طبیعی و روزمره",
                    "دوستانه",
                    "صمیمی",
                    "گرم و دوستانه",
                    "غیررسمی",
                    "رسمی",
                    "نیمه‌رسمی",
                    "محترمانه",
                    "کوتاه و طبیعی",
                    "کوتاه و دقیق",
                    "مختصر",
                    "توضیحی",
                    "کامل و توضیحی",
                    "مستقیم",
                    "غیرمستقیم",
                    "آرام و صبور",
                    "همدلانه",
                    "حمایتی",
                    "دلگرم‌کننده",
                    "حرفه‌ای",
                    "عملی",
                    "ساده و قابل فهم",
                    "آموزشی",
                    "فنی",
                    "فنی و تخصصی",
                    "تحلیلی",
                    "مقایسه‌ای",
                    "مبتنی بر سناریو",
                    "مبتنی بر context",
                    "مبتنی بر گفت‌وگو",
                    "مبتنی بر تجربه",
                    "قاطع",
                    "انعطاف‌پذیر",
                    "خنثی",
                    "جدی",
                    "مثبت",
                    "واقع‌گرایانه",
                    "طبیعی و خودمانی",
                    "مؤدبانه و رسمی",
                    "مؤدبانه و دوستانه"
                ],
                "audiences": [
                    "کاربر عمومی",
                    "کاربر مبتدی",
                    "کاربر حرفه‌ای",
                    "فرد غریبه",
                    "آشنای جدید",
                    "دوست",
                    "عضو خانواده",
                    "همکار",
                    "مدیر",
                    "کارمند",
                    "مشتری",
                    "فروشنده",
                    "پشتیبان",
                    "کارشناس",
                    "کارشناس خدمات",
                    "کارشناس فنی",
                    "معلم",
                    "دانش‌آموز",
                    "دانشجو",
                    "استاد",
                    "مسافر",
                    "میزبان",
                    "مهمان",
                    "پزشک",
                    "پرستار",
                    "مخاطب رسمی",
                    "مخاطب غیررسمی",
                    "مخاطب حرفه‌ای",
                    "مخاطب فنی",
                    "مدیر کسب‌وکار",
                    "صاحب کسب‌وکار",
                    "مشتری ناراضی",
                    "فرد ناراحت",
                    "فرد نگران",
                    "فرد عصبانی",
                    "فرد سردرگم",
                    "فرد کنجکاو",
                    "فرد عجول",
                    "فرد بی‌تجربه",
                    "فرد باتجربه",
                    "برنامه‌نویس",
                    "مهندس",
                    "پژوهشگر",
                    "کارشناس کسب‌وکار"
                ],
                "question_styles": [
                    "احوالپرسی",
                    "شروع مکالمه",
                    "پرسش مستقیم",
                    "پرسش کوتاه",
                    "پرسش محاوره‌ای",
                    "پرسش رسمی",
                    "پرسش غیررسمی",
                    "پرسش دوستانه",
                    "پرسش مؤدبانه",
                    "پرسش تکمیلی",
                    "پرسش پیگیری",
                    "پرسش مرتبط با پاسخ قبلی",
                    "پرسش مبتنی بر context",
                    "پرسش درباره موضوع قبلی",
                    "پرسش چندبخشی",
                    "پرسش غیرمستقیم",
                    "پرسش مبهم",
                    "پرسش روشن‌کننده",
                    "پرسش چگونه",
                    "پرسش چرا",
                    "پرسش اگر",
                    "پرسش فرضی",
                    "پرسش مقایسه‌ای",
                    "پرسش تصمیم‌محور",
                    "پرسش مسئله‌محور",
                    "پرسش سناریومحور",
                    "پرسش تجربه‌محور",
                    "پرسش نظرخواهی",
                    "درخواست مستقیم",
                    "درخواست کوتاه",
                    "درخواست محاوره‌ای",
                    "درخواست مؤدبانه",
                    "درخواست کمک",
                    "درخواست اطلاعات",
                    "درخواست توضیح",
                    "درخواست توضیح بیشتر",
                    "درخواست مثال",
                    "درخواست پیشنهاد",
                    "درخواست راهنمایی",
                    "درخواست نظر",
                    "درخواست تأیید",
                    "درخواست تکرار",
                    "درخواست ساده‌تر توضیح دادن",
                    "درخواست جزئیات بیشتر",
                    "بیان مشکل",
                    "شرح موقعیت",
                    "بیان نیاز",
                    "بیان ترجیح",
                    "بیان نظر",
                    "بیان مخالفت",
                    "بیان موافقت",
                    "اصلاح درخواست قبلی",
                    "تغییر درخواست قبلی",
                    "اضافه کردن اطلاعات جدید",
                    "ارجاع به صحبت قبلی",
                    "ارجاع به پاسخ قبلی",
                    "ادامه موضوع قبلی",
                    "پیام کوتاه",
                    "پیام روزمره",
                    "پیام دوستانه",
                    "پیام کاری",
                    "پیام رسمی",
                    "پیام غیررسمی",
                    "پیام خدماتی",
                    "پیام پشتیبانی",
                    "پیام تلفنی",
                    "پیام آنلاین"
                ],
                "bad_patterns": [
                    "به عنوان یک مدل زبانی",
                    "به عنوان هوش مصنوعی",
                    "به عنوان یک دستیار هوش مصنوعی",
                    "امیدوارم این پاسخ مفید باشد",
                    "اگر سؤال دیگری دارید",
                    "در صورت داشتن هرگونه سؤال دیگر",
                    "من نمی‌توانم به اینترنت دسترسی داشته باشم"
                ]
            },

            "en": {
                "name": "English",
                "native": "English",
                "script_min": 0.65,
                "prompt": "All user and assistant content must be written in natural, fluent, idiomatic English. Avoid literal translations, unnatural phrasing and repetitive templates.",
                "tasks": [
                    "Starting a conversation",
                    "Continuing a conversation",
                    "Responding to a message",
                    "Answering a question",
                    "Asking a question",
                    "Asking a follow-up question",
                    "Asking a related question",
                    "Requesting information",
                    "Asking for help",
                    "Asking for an explanation",
                    "Asking for more explanation",
                    "Asking for an example",
                    "Asking for guidance",
                    "Asking for a recommendation",
                    "Asking for advice",
                    "Asking for an opinion",
                    "Asking for confirmation",
                    "Asking someone to repeat",
                    "Asking for simplification",
                    "Requesting more details",
                    "Clarifying ambiguity",
                    "Resolving a misunderstanding",
                    "Following up on a request",
                    "Correcting a previous request",
                    "Changing a request",
                    "Adding information",
                    "Removing information from a request",
                    "Referring to previous conversation",
                    "Referring to a previous answer",
                    "Continuing from previous context",
                    "Describing a problem",
                    "Describing a situation",
                    "Asking for a solution",
                    "Troubleshooting",
                    "Problem solving",
                    "Evaluating solutions",
                    "Choosing between options",
                    "Decision making",
                    "Comparing options",
                    "Evaluating advantages and disadvantages",
                    "Evaluating consequences",
                    "Asking for feedback",
                    "Giving feedback",
                    "Expressing an opinion",
                    "Expressing a preference",
                    "Expressing agreement",
                    "Expressing disagreement",
                    "Disagreeing politely",
                    "Reaching an agreement",
                    "Negotiation",
                    "Bargaining",
                    "Making a suggestion",
                    "Accepting a suggestion",
                    "Rejecting a suggestion",
                    "Thanking",
                    "Apologizing",
                    "Accepting an apology",
                    "Congratulating",
                    "Showing sympathy",
                    "Encouraging someone",
                    "Inviting someone",
                    "Accepting an invitation",
                    "Declining an invitation",
                    "Calming someone down",
                    "Responding with empathy",
                    "Responding to criticism",
                    "Responding to a complaint",
                    "Handling a sensitive conversation",
                    "Coordination",
                    "Planning",
                    "Teaching",
                    "Step-by-step instruction",
                    "Practice",
                    "Language correction",
                    "Sentence correction",
                    "Rewriting",
                    "Changing tone",
                    "Simplifying",
                    "Conversational translation",
                    "Writing a suitable reply",
                    "Summarizing a conversation",
                    "Wrapping up a conversation",
                    "Reaching a conclusion",
                    "Continuing a scenario",
                    "Conflict resolution",
                    "Customer support",
                    "Technical support",
                    "Service conversation",
                    "Work conversation",
                    "Social conversation",
                    "Friendly conversation",
                    "Formal conversation",
                    "Informal conversation",
                    "Open-ended conversation"
                ],
                "styles": [
                    "Conversational",
                    "Natural and everyday",
                    "Friendly",
                    "Warm and friendly",
                    "Casual",
                    "Informal",
                    "Formal",
                    "Semi-formal",
                    "Polite",
                    "Short and natural",
                    "Short and precise",
                    "Brief",
                    "Explanatory",
                    "Detailed explanatory",
                    "Direct",
                    "Indirect",
                    "Calm and patient",
                    "Empathetic",
                    "Supportive",
                    "Encouraging",
                    "Professional",
                    "Practical",
                    "Simple and clear",
                    "Educational",
                    "Technical",
                    "Technical and specialized",
                    "Analytical",
                    "Comparative",
                    "Scenario-based",
                    "Context-aware",
                    "Conversation-driven",
                    "Experience-based",
                    "Assertive",
                    "Flexible",
                    "Neutral",
                    "Serious",
                    "Positive",
                    "Realistic",
                    "Natural and casual",
                    "Polite and formal",
                    "Polite and friendly"
                ],
                "audiences": [
                    "General user",
                    "Beginner user",
                    "Experienced user",
                    "Stranger",
                    "New acquaintance",
                    "Friend",
                    "Family member",
                    "Coworker",
                    "Manager",
                    "Employee",
                    "Customer",
                    "Seller",
                    "Support agent",
                    "Specialist",
                    "Service representative",
                    "Technical specialist",
                    "Teacher",
                    "Student",
                    "University student",
                    "Professor",
                    "Traveler",
                    "Host",
                    "Guest",
                    "Doctor",
                    "Nurse",
                    "Formal audience",
                    "Informal audience",
                    "Professional audience",
                    "Technical audience",
                    "Business manager",
                    "Business owner",
                    "Dissatisfied customer",
                    "Upset person",
                    "Worried person",
                    "Angry person",
                    "Confused person",
                    "Curious person",
                    "Impatient person",
                    "Inexperienced person",
                    "Experienced practitioner",
                    "Developer",
                    "Engineer",
                    "Researcher",
                    "Business professional"
                ],
                "question_styles": [
                    "Greeting",
                    "Conversation opener",
                    "Direct question",
                    "Short question",
                    "Conversational question",
                    "Formal question",
                    "Informal question",
                    "Friendly question",
                    "Polite question",
                    "Follow-up question",
                    "Related question",
                    "Question based on a previous answer",
                    "Context-based question",
                    "Question about a previous topic",
                    "Multi-part question",
                    "Indirect question",
                    "Ambiguous question",
                    "Clarifying question",
                    "How-to question",
                    "Why question",
                    "What-if question",
                    "Hypothetical question",
                    "Comparison question",
                    "Decision-oriented question",
                    "Problem-based question",
                    "Scenario-based question",
                    "Experience-based question",
                    "Opinion-seeking question",
                    "Direct request",
                    "Short request",
                    "Conversational request",
                    "Polite request",
                    "Request for help",
                    "Request for information",
                    "Request for explanation",
                    "Request for more explanation",
                    "Request for an example",
                    "Request for a suggestion",
                    "Request for guidance",
                    "Request for an opinion",
                    "Request for confirmation",
                    "Request to repeat",
                    "Request for a simpler explanation",
                    "Request for more details",
                    "Problem statement",
                    "Situation description",
                    "Need statement",
                    "Preference statement",
                    "Opinion statement",
                    "Disagreement statement",
                    "Agreement statement",
                    "Correction of a previous request",
                    "Change to a previous request",
                    "Adding new information",
                    "Reference to previous conversation",
                    "Reference to previous answer",
                    "Continuation of previous topic",
                    "Short message",
                    "Everyday message",
                    "Friendly message",
                    "Work message",
                    "Formal message",
                    "Informal message",
                    "Service message",
                    "Support message",
                    "Phone message",
                    "Online message"
                ],
                "bad_patterns": [
                    "as an ai",
                    "as an ai language model",
                    "i hope this helps",
                    "if you have any further questions",
                    "i cannot browse the internet"
                ]
            }
        }

        self.stats = {
            "attempts": 0,
            "accepted": 0,
            "generation_failed": 0,
            "json_failed": 0,
            "validation_failed": 0,
            "language_failed": 0,
            "quality_failed": 0,
            "duplicate_failed": 0
        }

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

        self.intro_fa = "یک نمونه مکالمه چندمرحله‌ای باکیفیت برای دیتاست آموزش و فاین‌تیون مدل زبانی تولید کن."
        self.intro_en = "Generate a high-quality multi-turn instruction-tuning example."

        self.prompt_config = {
            "fa": {
                "intro": {
                    "single": "یک نمونه تک‌مرحله‌ای باکیفیت برای دیتاست آموزش و فاین‌تیون مدل زبانی تولید کن.",
                    "multi": "یک نمونه مکالمه چندمرحله‌ای باکیفیت برای دیتاست آموزش و فاین‌تیون مدل زبانی تولید کن."
                },

                "turn_instruction": {
                    "single": """
        فقط یک نوبت سؤال و پاسخ تولید کن.

        خروجی باید دقیقاً شامل یک پیام user و یک پیام assistant باشد.
        """,

                    "multi": """
        یک گفت‌وگوی چندمرحله‌ای تولید کن.

        تعداد نوبت‌های گفت‌وگو باید بین {min_turns} و {max_turns} نوبت باشد.
        هر نوبت شامل یک پیام user و یک پیام assistant است.

        گفت‌وگو باید پیوستگی معنایی داشته باشد.
        هر پیام user بعدی باید بر اساس پاسخ قبلی یا context مکالمه شکل بگیرد.
        کاربر نباید بدون ارتباط موضوع را تغییر دهد.
        در برخی نوبت‌ها می‌توان از ارجاع‌های طبیعی مانند «این مورد»، «همین راهکار»، «اگر این‌طور باشد» و موارد مشابه استفاده کرد.
        دستیار باید تمام context قبلی مکالمه را در نظر بگیرد.
        از تکرار سؤال یا پاسخ قبلی خودداری کن.
        مکالمه باید شبیه یک گفت‌وگوی واقعی و طبیعی باشد.
        """
                },

                "instructions": """
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
        """,

                "output": """
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
        """,

                "continuation": {
                    "single": "در حالت تک‌مرحله‌ای دقیقاً فقط دو پیام تولید کن: user → assistant.",
                    "multi": "در حالت چندمرحله‌ای، messages باید با همین الگو ادامه پیدا کند: user → assistant → user → assistant → ..."
                }
            },

            "en": {
                "intro": {
                    "single": "Generate a high-quality single-turn instruction-tuning example.",
                    "multi": "Generate a high-quality multi-turn instruction-tuning example."
                },

                "turn_instruction": {
                    "single": """
        Generate exactly one user message followed by one assistant message.

        The output must contain exactly two messages.
        """,

                    "multi": """
        Generate a multi-turn conversation.

        The conversation must contain between {min_turns} and {max_turns} turns.
        Each turn consists of one user message followed by one assistant message.

        The conversation must maintain semantic continuity.
        Each following user message should naturally build on previous answers or conversation context.
        Do not abruptly switch to unrelated topics.
        Some user messages may naturally refer to previous context.
        The assistant must consider the full conversation history when responding.
        Avoid repeating previous questions or answers.
        The conversation must feel realistic and natural.
        """
                },

                "instructions": """
        The user messages must be realistic and natural.
        The assistant responses must be accurate, useful, relevant and complete.

        Avoid repetitive structures, generic filler, artificial benchmark prompts and meta commentary.
        """,

                "output": """
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
        """,

                "continuation": {
                    "single": "For single-turn mode, return exactly two messages: user → assistant.",
                    "multi": "For multi-turn mode, continue the same pattern: user → assistant → user → assistant → ..."
                }
            }
        }

        self.judge_config = {
            "system_prompt": "You are a strict dataset quality evaluator. Return only valid JSON.",

            "user_prompt": """
        این نمونه دیتاست instruction-tuning را از نظر کیفیت بررسی کن.

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
        {sample}
        """,

            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 200,
            "response_format": {
                "type": "json_object"
            }
        }

        self.difficulties = {
            "fa": ["مبتدی", "متوسط", "پیشرفته", "تخصصی"],
            "en": ["Beginner", "Intermediate", "Advanced", "Expert"]
        }

        self.generation_config = {
            "temperature_min": 0.55,
            "temperature_max": 0.95,
            "temperature_variation": 0.08,
            "response_format": {
                "type": "json_object"
            }
        }

        self.lexical_mi = {
            "میان", "میوه", "میهن", "میزان", "میدان", "میعاد", "میانگین",
            "میانه", "میلاد", "میل", "میخ", "میخک", "میگرن", "میگو",
            "میانجی", "میانسال", "میان‌مدت", "میان‌بر", "میانگین"
        }

        self.comparative_exceptions = {
            "بهتر", "بهترین", "بیشتر", "بیشترین", "کمتر", "کمترین",
            "پیشتر", "پیشترین", "دیگر", "آخر", "آخرین", "برتر",
            "برترین", "سوتر", "زودتر", "زودترین"
        }

        self.config_text = self._build_config_text()

    def _build_config_text(self) -> str:
        lines = []

        lines.append("")
        lines.append("=" * 90)
        lines.append("SyntheticDatasetConfig Configuration")
        lines.append("=" * 90)

        lines.append("[MODEL]")
        lines.append(f"Model_Path                : {self.model_path}")
        lines.append(f"N_Context                 : {self.n_ctx}")
        lines.append(f"N_Threads                 : {self.n_threads}")
        lines.append(f"N_Batch                   : {self.n_batch}")
        lines.append(f"Max_Tokens                : {self.max_tokens}")
        lines.append(f"N_GPU_Layers              : {self.n_gpu_layers}")
        lines.append(f"Load_Model_Use_MMap       : {self.load_model_use_mmap}")
        lines.append(f"Load_Model_Use_MLock      : {self.load_model_use_mlock}")
        lines.append(f"Load_Model_Verbose        : {self.load_model_verbose}")
        lines.append(f"judge_llm_use_mmap        : {self.judge_llm_use_mmap}")
        lines.append(f"judge_llm_use_mlock       : {self.judge_llm_use_mlock}")
        lines.append(f"judge_llm_verbose         : {self.judge_llm_verbose}")

        lines.append("")
        lines.append("[OUTPUT]")
        lines.append(f"Output_Path               : {self.output_path}")
        lines.append(f"Export_Final              : {self.export_final}")
        lines.append(f"Cleanup_Shards            : {self.cleanup_shards}")
        lines.append(f"Shard_Size                : {self.shard_size}")
        lines.append(f"Checkpoint_Interval       : {self.checkpoint_interval}")

        lines.append("")
        lines.append("[GENERATION]")
        lines.append(f"Total_Samples             : {self.total_samples}")
        lines.append(f"Seed                      : {self.seed}")
        lines.append(f"Language                  : {self.language}")
        lines.append(f"Temperature               : {self.temperature}")
        lines.append(f"Top_P                     : {self.top_p}")
        lines.append(f"Min_P                     : {self.min_p}")
        lines.append(f"Repeat_Penalty            : {self.repeat_penalty}")
        lines.append(f"Retry_Count               : {self.retry_count}")
        lines.append(f"Max_Attempts_Multiplier   : {self.max_attempts_multiplier}")
        lines.append(f"Generation_Config         : {repr(self.generation_config)}")

        lines.append("")
        lines.append("[VALIDATION]")
        lines.append(f"Min_User_Words            : {self.min_user_words}")
        lines.append(f"Max_User_Words            : {self.max_user_words}")
        lines.append(f"Min_Assistant_Words       : {self.min_assistant_words}")
        lines.append(f"Max_Assistant_Words       : {self.max_assistant_words}")
        lines.append(f"Min_Quality_Score         : {self.min_quality_score}")

        lines.append("")
        lines.append("[MULTI TURN]")
        lines.append(f"Multi_Turn                : {self.multi_turn}")
        lines.append(f"Min_Turns                 : {self.min_turns}")
        lines.append(f"Max_Turns                 : {self.max_turns}")

        lines.append("")
        lines.append("[QUALITY JUDGE]")
        lines.append(f"Enable_Quality_Judge      : {self.enable_quality_judge}")
        lines.append(f"Judge_Model_Path          : {self.judge_model_path}")
        lines.append(f"Judge_Config              : {repr(self.judge_config)}")

        lines.append("")
        lines.append("[TOPICS]")
        lines.append(f"Topics_Languages          : {list(self.topics.keys())}")
        lines.append(f"Topics_Count_FA           : {len(self.topics.get('fa', []))}")
        lines.append(f"Topics_Count_EN           : {len(self.topics.get('en', []))}")

        for language, topics in self.topics.items():
            lines.append(f"Topics_{language.upper()}     : {topics}")

        lines.append("")
        lines.append("[DIFFICULTIES]")
        lines.append(f"Difficulties               : {repr(self.difficulties)}")

        lines.append("")
        lines.append("[LANGUAGE CONFIGS]")

        for language, config in self.language_configs.items():
            lines.append("")
            lines.append(f"Language                  : {language}")
            lines.append(f"Name                      : {config.get('name')}")
            lines.append(f"Native                    : {config.get('native')}")
            lines.append(f"Script_Min                : {config.get('script_min')}")
            lines.append(f"Prompt                    : {config.get('prompt')}")

            lines.append(
                f"Tasks_Count               : {len(config.get('tasks', []))}"
            )
            lines.append(
                f"Styles_Count              : {len(config.get('styles', []))}"
            )
            lines.append(
                f"Audiences_Count           : {len(config.get('audiences', []))}"
            )
            lines.append(
                f"Question_Styles_Count     : {len(config.get('question_styles', []))}"
            )
            lines.append(
                f"Bad_Patterns_Count        : {len(config.get('bad_patterns', []))}"
            )

            lines.append(f"Tasks                     : {config.get('tasks')}")
            lines.append(f"Styles                    : {config.get('styles')}")
            lines.append(f"Audiences                 : {config.get('audiences')}")
            lines.append(
                f"Question_Styles           : {config.get('question_styles')}"
            )
            lines.append(
                f"Bad_Patterns              : {config.get('bad_patterns')}"
            )

        lines.append("")
        lines.append("[SYSTEM PROMPT]")
        lines.append(self.system_prompt)

        lines.append("")
        lines.append("[INTRO PROMPTS]")
        lines.append(f"Intro_FA                  : {self.intro_fa}")
        lines.append(f"Intro_EN                  : {self.intro_en}")

        lines.append("")
        lines.append("[PROMPT CONFIG]")
        lines.append(repr(self.prompt_config))

        lines.append("")
        lines.append("[LEXICAL CONFIG]")
        lines.append(f"Lexical_MI                : {self.lexical_mi}")
        lines.append(f"Comparative_Exceptions    : {self.comparative_exceptions}")

        lines.append("")
        lines.append("[STATS]")
        for key, value in self.stats.items():
            lines.append(f"{key:<25}: {value}")

        lines.append("-" * 113)
        text = "\n".join(lines)
        text = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda m: chr(int(m.group(1), 16)),
            text
        )

        return text

    def log(self):
        if self._logged:
            return

        self.logger.info(self.config_text)
        self._logged = True